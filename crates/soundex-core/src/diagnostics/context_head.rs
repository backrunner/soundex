//! Experimental four-input P20/P24 ABI. Not accepted by the production loader.
//! Uses production channel analysis/detection/synthesis/gating with one ORT call
//! per physical-channel batch, including bypass/silence, and past-only context.

use std::{error::Error, path::Path, time::Instant};

use ndarray::{s, Array4};
use ort::{
    session::{
        builder::GraphOptimizationLevel, HasSelectedOutputs, OutputSelector, RunOptions, Session,
    },
    value::Tensor,
};
use soundex_dsp::{
    limiter::Limiter,
    stft::{SpectralFrame, StftAnalyzer},
};

use crate::{
    channel::ChannelProcessor,
    diagnostics::ablation::{ConditionAblation, FusionAblation},
    processor::{validate_config, GateMixer},
    HighBandGain, SoundExConfig,
};

type Outcome<T> = std::result::Result<T, Box<dyn Error>>;

pub const WARMUP: usize = 80;
// P24's inherited input cache masks the detector condition for its first 64
// hops. This is distinct from the head's 80-hop state/output warmup.
pub const CONDITION_WARMUP: usize = 64;
pub const CONTEXT_BINS: usize = 1538;

/// Preallocated Rust equivalent of training's StreamingContext("past").
pub struct PastContext {
    history: Vec<Vec<f32>>,
    analyzers: [StftAnalyzer; 2],
    spectra: [SpectralFrame; 2],
    held: Vec<f32>,
    hop: usize,
}

impl PastContext {
    pub fn new(channels: usize) -> Self {
        assert!(matches!(channels, 1 | 2));
        Self {
            history: vec![vec![0.0; 2048]; channels],
            analyzers: [StftAnalyzer::new(1024, 128), StftAnalyzer::new(2048, 128)],
            spectra: [SpectralFrame::zeros(513), SpectralFrame::zeros(1025)],
            held: vec![0.0; channels * CONTEXT_BINS],
            hop: 0,
        }
    }

    pub fn step(&mut self, input: &[f32]) -> &[f32] {
        let channels = self.history.len();
        assert_eq!(input.len(), channels * 128);
        for (channel, history) in self.history.iter_mut().enumerate() {
            history.copy_within(128.., 0);
            for (n, value) in history[1920..].iter_mut().enumerate() {
                *value = input[n * channels + channel];
            }
            if self.hop.is_multiple_of(4) {
                let mut offset = channel * CONTEXT_BINS;
                for (analyzer, spectrum) in self.analyzers.iter_mut().zip(&mut self.spectra) {
                    let size = analyzer.fft_size();
                    analyzer.analyze_into(&history[2048 - size..], spectrum);
                    for &db in &spectrum.log_magnitude {
                        // The analyzer floors at 1e-10; preserve true silence.
                        self.held[offset] = if db <= -200.0 {
                            0.0
                        } else {
                            10.0_f32.powf(db / 20.0) * (256.0 / size as f32)
                        };
                        offset += 1;
                    }
                }
            }
        }
        self.hop += 1;
        &self.held
    }

    pub fn reset(&mut self) {
        for history in &mut self.history {
            history.fill(0.0);
        }
        self.held.fill(0.0);
        self.hop = 0;
    }
}

/// Diagnostics only: no callback transport, device, failure recovery or promotion.
pub struct ContextHeadStream {
    config: SoundExConfig,
    channels: Vec<ChannelProcessor>,
    mono: Vec<f32>,
    context: PastContext,
    session: Session,
    features: Tensor<f32>,
    context_input: Tensor<f32>,
    condition: Tensor<f32>,
    state: Tensor<f32>,
    options: RunOptions<HasSelectedOutputs>,
    prediction: Array4<f32>,
    gate: GateMixer,
    dry_limiter: Limiter,
    dry: Vec<f32>,
    wet: Vec<f32>,
    hops: usize,
    failed: bool,
    fusion_ablation: FusionAblation,
    condition_ablation: ConditionAblation,
}

impl ContextHeadStream {
    pub fn new(path: &Path, config: SoundExConfig) -> Outcome<Self> {
        Self::with_ablation(
            path,
            config,
            FusionAblation::Current,
            ConditionAblation::Native,
        )
    }

    pub fn with_ablation(
        path: &Path,
        mut config: SoundExConfig,
        fusion_ablation: FusionAblation,
        condition_ablation: ConditionAblation,
    ) -> Outcome<Self> {
        validate_config(&config)?;
        if config.fft_size != 256 || config.hop_size != 128 {
            return Err("diagnostic context head requires FFT256/hop128".into());
        }
        config.high_band_gain = HighBandGain::Model;
        config.limiter_mode = soundex_dsp::limiter::LimiterMode::SmoothKnee;
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_parallel_execution(false)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        if session
            .metadata()?
            .custom("soundex.experimental")
            .as_deref()
            != Some("p20-context-head-diagnostic")
            || session.metadata()?.custom("soundex.deployable").as_deref() != Some("false")
            || session
                .inputs()
                .iter()
                .map(|x| x.name())
                .collect::<Vec<_>>()
                != ["features", "context", "condition", "state"]
            || session
                .outputs()
                .iter()
                .map(|x| x.name())
                .collect::<Vec<_>>()
                != ["output", "next_state"]
        {
            return Err("expected explicit experimental context-head ABI".into());
        }
        let width: usize = session
            .metadata()?
            .custom("soundex.width")
            .ok_or("missing width")?
            .parse()?;
        if width == 0 || width > 8192 {
            return Err("invalid diagnostic width".into());
        }
        let count = config.channels as usize;
        let shape = [count, 2, 1, 129];
        let mut state = Tensor::<f32>::new(session.allocator(), [count, width * 63])?;
        state.extract_tensor_mut().1.fill(0.0);
        let options = RunOptions::new()?.with_outputs(
            OutputSelector::default()
                .preallocate("output", Tensor::<f32>::new(session.allocator(), shape)?)
                .preallocate(
                    "next_state",
                    Tensor::<f32>::new(session.allocator(), [count, width * 63])?,
                ),
        );
        Ok(Self {
            fusion_ablation,
            condition_ablation,
            channels: (0..count)
                .map(|_| ChannelProcessor::new(&config, 1))
                .collect(),
            mono: vec![0.0; 128],
            context: PastContext::new(count),
            features: Tensor::new(session.allocator(), shape)?,
            context_input: Tensor::new(session.allocator(), [count, 1, CONTEXT_BINS])?,
            condition: Tensor::new(session.allocator(), [count, 1, 3])?,
            state,
            options,
            prediction: Array4::zeros(shape),
            session,
            gate: GateMixer::new(128, 128),
            dry_limiter: Limiter::with_mode(config.limiter_ceiling, 10.0, config.limiter_mode),
            dry: vec![0.0; 128 * count],
            wet: vec![0.0; 128 * count],
            config,
            hops: 0,
            failed: false,
        })
    }

    /// Returns frontend, inference, DSP and total milliseconds, without report IO.
    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Outcome<[f64; 4]> {
        if self.failed {
            return Err("diagnostic stream failed; reset required".into());
        }
        if input.len() != self.dry.len()
            || output.len() != input.len()
            || input.iter().any(|x| !x.is_finite())
        {
            return Err("expected finite complete interleaved hop".into());
        }
        let result = self.step_inner(input, output);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn step_inner(&mut self, input: &[f32], output: &mut [f32]) -> Outcome<[f64; 4]> {
        let start = Instant::now();
        let count = self.channels.len();
        self.context_input
            .extract_tensor_mut()
            .1
            .copy_from_slice(self.context.step(input));
        for (c, channel) in self.channels.iter_mut().enumerate() {
            for (n, value) in self.mono.iter_mut().enumerate() {
                *value = input[n * count + c];
            }
            channel.analyze_hop(&self.mono, &self.config)?;
            let spectrum = channel.observed_spectrum();
            let values = &mut self.features.extract_tensor_mut().1[c * 258..(c + 1) * 258];
            values[..129].copy_from_slice(&spectrum.log_magnitude);
            values[129..].copy_from_slice(&spectrum.phase);
            self.condition.extract_tensor_mut().1[c * 3..(c + 1) * 3].copy_from_slice(&[
                self.config.sample_rate as f32 / 48000.0,
                channel.info().detected_bandwidth_hz / (self.config.sample_rate as f32 / 2.0),
                f32::from(self.hops >= CONDITION_WARMUP && channel.needs_enhancement()),
            ]);
            self.condition_ablation
                .apply(&mut self.condition.extract_tensor_mut().1[c * 3..(c + 1) * 3]);
        }
        let frontend = start.elapsed();
        let inference_start = Instant::now();
        let result = self.session.run_with_options(
            ort::inputs![
                &self.features,
                &self.context_input,
                &self.condition,
                &self.state
            ],
            &self.options,
        )?;
        let (shape, values) = result["output"].try_extract_tensor::<f32>()?;
        let (state_shape, state) = result["next_state"].try_extract_tensor::<f32>()?;
        if shape.as_ref() != [count as i64, 2, 1, 129]
            || state_shape.as_ref() != self.state.shape().as_ref()
            || values.iter().chain(state).any(|x| !x.is_finite())
        {
            return Err("invalid experimental model output/state".into());
        }
        self.prediction
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(values);
        self.state.extract_tensor_mut().1.copy_from_slice(state);
        drop(result);
        let inference = inference_start.elapsed();
        let dsp_start = Instant::now();
        let mut active = false;
        for (c, channel) in self.channels.iter_mut().enumerate() {
            if self.hops < WARMUP {
                channel.suppress_enhancement();
                let observed = &self.features.extract_tensor().1[c * 258..(c + 1) * 258];
                self.prediction
                    .slice_mut(s![c, .., .., ..])
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(observed);
            }
            let mode = if self.hops < WARMUP {
                FusionAblation::Current
            } else {
                self.fusion_ablation
            };
            let info = channel.finish_hop_ablation(
                self.prediction.slice(s![c, .., .., ..]),
                &self.config,
                mode,
            )?;
            active |= !info.bypassed;
            for n in 0..128 {
                self.dry[n * count + c] = channel.dry()[n];
                self.wet[n * count + c] = channel.wet()[n];
            }
        }
        self.dry_limiter.process_buffer(&mut self.dry);
        self.gate
            .mix_interleaved_into(&self.dry, &self.wet, count, active, output);
        for x in output.iter_mut() {
            *x = x.clamp(-self.config.limiter_ceiling, self.config.limiter_ceiling);
        }
        self.hops += 1;
        Ok([
            frontend.as_secs_f64() * 1000.0,
            inference.as_secs_f64() * 1000.0,
            dsp_start.elapsed().as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64() * 1000.0,
        ])
    }

    /// Optional allocating trace outside the measured hop. Never used in cost runs.
    pub fn trace(&self) -> serde_json::Value {
        serde_json::json!({"features": self.features.extract_tensor().1,
            "context": self.context_input.extract_tensor().1,
            "condition": self.condition.extract_tensor().1,
            "prediction": self.prediction.as_slice().unwrap()})
    }

    pub fn enhancing(&self) -> bool {
        self.channels
            .iter()
            .any(ChannelProcessor::needs_enhancement)
    }

    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.reset();
        }
        self.context.reset();
        self.state.extract_tensor_mut().1.fill(0.0);
        self.gate.reset();
        self.hops = 0;
        self.failed = false;
        self.dry.fill(0.0);
        self.wet.fill(0.0);
        self.prediction.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_is_past_only_held_channel_isolated_and_resettable() {
        let mut mono = PastContext::new(1);
        let mut stereo = PastContext::new(2);
        let pcm: Vec<f32> = (0..128).map(|n| (n as f32 * 0.1).sin()).collect();
        let interleaved: Vec<f32> = pcm.iter().flat_map(|&x| [0.0, x]).collect();
        let first = mono.step(&pcm).to_vec();
        let second = stereo.step(&interleaved);
        assert_eq!(&second[CONTEXT_BINS..], first);
        assert!(second[..CONTEXT_BINS].iter().all(|&x| x == 0.0));
        for _ in 0..3 {
            assert_eq!(mono.step(&[0.0; 128]), first);
        }
        assert_ne!(mono.step(&[0.0; 128]), first);
        mono.reset();
        assert_eq!(mono.step(&pcm), first);
        mono.reset();
        assert!(mono.step(&[0.0; 128]).iter().all(|&x| x == 0.0));
    }
}
