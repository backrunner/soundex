//! Public streaming and offline audio enhancement processor.

use ndarray::{s, Array4};
use soundex_dsp::limiter::Limiter;

use crate::channel::ChannelProcessor;
use crate::config::SoundExConfig;
use crate::error::{Result, SoundExError};
use crate::inference::InferenceEngine;
use crate::stream::{latency_hops, CausalHopIter};

/// Information about the most recently processed frame.
#[derive(Debug, Clone, Copy)]
pub struct ProcessInfo {
    /// Whether all channels bypassed model enhancement for this frame.
    pub bypassed: bool,
    /// Lowest detected effective bandwidth across channels, in Hz.
    pub detected_bandwidth_hz: f32,
    /// Applied generated high-band gain, in dB.
    pub enhancement_gain_db: f32,
}

impl Default for ProcessInfo {
    fn default() -> Self {
        Self {
            bypassed: true,
            detected_bandwidth_hz: 0.0,
            enhancement_gain_db: 0.0,
        }
    }
}

/// Sample counts for an arbitrary-chunk streaming call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamProgress {
    /// Interleaved input samples accepted by the call.
    pub consumed: usize,
    /// Interleaved output samples appended by the call.
    pub produced: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Ready,
    FixedHop,
    Chunked,
    Finalized,
    Poisoned,
}

/// Shared sample-domain wet/dry gate for every channel.
struct GateMixer {
    current: f32,
    start: f32,
    target: f32,
    ramp_position: usize,
    ramp_samples: usize,
    release_samples: usize,
    release_remaining: usize,
}

impl GateMixer {
    fn new(ramp_samples: usize, release_samples: usize) -> Self {
        Self {
            current: 0.0,
            start: 0.0,
            target: 0.0,
            ramp_position: ramp_samples,
            ramp_samples: ramp_samples.max(1),
            release_samples,
            release_remaining: 0,
        }
    }

    #[cfg(test)]
    fn mix_interleaved(
        &mut self,
        dry: &[f32],
        wet: &[f32],
        channels: usize,
        request_wet: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0; dry.len()];
        self.mix_interleaved_into(dry, wet, channels, request_wet, &mut output);
        output
    }

    fn mix_interleaved_into(
        &mut self,
        dry: &[f32],
        wet: &[f32],
        channels: usize,
        request_wet: bool,
        output: &mut [f32],
    ) {
        debug_assert_eq!(dry.len(), wet.len());
        debug_assert_eq!(dry.len(), output.len());
        debug_assert_eq!(dry.len() % channels, 0);
        if request_wet {
            self.release_remaining = self.release_samples;
        }

        for ((dry_frame, wet_frame), output_frame) in dry
            .chunks_exact(channels)
            .zip(wet.chunks_exact(channels))
            .zip(output.chunks_exact_mut(channels))
        {
            let hold_wet = request_wet || self.release_remaining > 0;
            if !request_wet && self.release_remaining > 0 {
                self.release_remaining -= 1;
            }
            self.set_target(if hold_wet { 1.0 } else { 0.0 });
            let mix = self.next_mix();
            for ((output_sample, &dry_sample), &wet_sample) in
                output_frame.iter_mut().zip(dry_frame).zip(wet_frame)
            {
                *output_sample = dry_sample * (1.0 - mix) + wet_sample * mix;
            }
        }
    }

    fn set_target(&mut self, target: f32) {
        if target != self.target {
            self.start = self.current;
            self.target = target;
            self.ramp_position = 0;
        }
    }

    fn next_mix(&mut self) -> f32 {
        if self.ramp_position >= self.ramp_samples {
            self.current = self.target;
            return self.current;
        }
        self.ramp_position += 1;
        let progress = self.ramp_position as f32 / self.ramp_samples as f32;
        let raised_cosine = 0.5 - 0.5 * (std::f32::consts::PI * progress).cos();
        self.current = self.start + (self.target - self.start) * raised_cosine;
        self.current
    }

    fn reset(&mut self) {
        self.current = 0.0;
        self.start = 0.0;
        self.target = 0.0;
        self.ramp_position = self.ramp_samples;
        self.release_remaining = 0;
    }
}

/// Streaming-capable SoundEx audio enhancement processor.
pub struct SoundExProcessor {
    config: SoundExConfig,
    engine: InferenceEngine,
    channels: Vec<ChannelProcessor>,
    gate: GateMixer,
    dry_limiter: Limiter,
    last_info: ProcessInfo,
    mode: StreamMode,
    pending_input: Vec<f32>,
    chunk_input_samples: usize,
    chunk_output_samples: usize,
    chunk_latency_to_discard: usize,
    channel_inputs: Vec<Vec<f32>>,
    channel_infos: Vec<ProcessInfo>,
    active_channels: Vec<usize>,
    inference_input: Array4<f32>,
    inference_output: Array4<f32>,
    dry_interleaved: Vec<f32>,
    wet_interleaved: Vec<f32>,
    hop_output: Vec<f32>,
}

impl SoundExProcessor {
    /// Create a processor and load the configured ONNX model.
    pub fn new(config: SoundExConfig) -> Result<Self> {
        validate_config(&config)?;
        let engine = InferenceEngine::load(&config.model_path, &config)?;
        let channels = (0..config.channels)
            .map(|_| ChannelProcessor::new(&config))
            .collect();
        let latency = config.fft_size - config.hop_size;
        let interleaved_latency = latency * config.channels as usize;
        let channel_count = config.channels as usize;
        let hop_size = config.hop_size;
        let hop_samples = hop_size * channel_count;
        let frequency_bins = config.fft_size / 2 + 1;
        let gate = GateMixer::new(config.hop_size.min(128), latency);
        let dry_limiter = Limiter::new(config.limiter_ceiling, 10.0);
        Ok(Self {
            config,
            engine,
            channels,
            gate,
            dry_limiter,
            last_info: ProcessInfo::default(),
            mode: StreamMode::Ready,
            pending_input: Vec::new(),
            chunk_input_samples: 0,
            chunk_output_samples: 0,
            chunk_latency_to_discard: interleaved_latency,
            channel_inputs: vec![vec![0.0; hop_size]; channel_count],
            channel_infos: vec![ProcessInfo::default(); channel_count],
            active_channels: Vec::with_capacity(channel_count),
            inference_input: Array4::zeros((channel_count, 2, 1, frequency_bins)),
            inference_output: Array4::zeros((channel_count, 2, 1, frequency_bins)),
            dry_interleaved: vec![0.0; hop_samples],
            wet_interleaved: vec![0.0; hop_samples],
            hop_output: vec![0.0; hop_samples],
        })
    }

    /// Process one interleaved streaming hop.
    ///
    /// `input` and `output` must both contain `hop_size * channels` samples.
    /// Startup samples, including zeros, are returned directly; call
    /// [`finalize`](Self::finalize) once to retrieve the delayed tail.
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<ProcessInfo> {
        self.ensure_usable()?;
        let expected = self.hop_samples();
        if input.len() != expected || output.len() != expected {
            return Err(SoundExError::InvalidInput(format!(
                "expected {expected} interleaved samples per frame, got input={} output={}",
                input.len(),
                output.len()
            )));
        }
        validate_finite(input)?;
        match self.mode {
            StreamMode::Ready | StreamMode::FixedHop => self.mode = StreamMode::FixedHop,
            StreamMode::Chunked => return Err(SoundExError::StreamModeConflict),
            StreamMode::Finalized | StreamMode::Poisoned => unreachable!(),
        }

        match self.process_hop_internal(input) {
            Ok(info) => {
                output.copy_from_slice(&self.hop_output);
                Ok(info)
            }
            Err(error) => {
                self.mode = StreamMode::Poisoned;
                Err(error)
            }
        }
    }

    /// Consume arbitrary complete interleaved sample frames.
    ///
    /// All input is accepted and partial hops are buffered. Produced samples
    /// are appended to `output`. Startup padding is withheld, so the outputs of
    /// every call followed by [`finalize`](Self::finalize) are same-length and
    /// sample-aligned with [`process_buffer`](Self::process_buffer).
    pub fn process_chunk(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<StreamProgress> {
        self.ensure_usable()?;
        let channel_count = self.channel_count();
        if !input.len().is_multiple_of(channel_count) {
            return Err(SoundExError::InvalidInput(format!(
                "interleaved input length {} is not divisible by {channel_count} channels",
                input.len()
            )));
        }
        validate_finite(input)?;
        match self.mode {
            StreamMode::Ready | StreamMode::Chunked => {}
            StreamMode::FixedHop => return Err(SoundExError::StreamModeConflict),
            StreamMode::Finalized | StreamMode::Poisoned => unreachable!(),
        }
        if input.is_empty() {
            return Ok(StreamProgress {
                consumed: 0,
                produced: 0,
            });
        }
        self.mode = StreamMode::Chunked;
        self.pending_input.extend_from_slice(input);
        self.chunk_input_samples += input.len();

        let mut staged = Vec::new();
        let hop_samples = self.hop_samples();
        while self.pending_input.len() >= hop_samples {
            let hop = self.pending_input[..hop_samples].to_vec();
            match self.process_hop_internal(&hop) {
                Ok(_) => {}
                Err(error) => {
                    self.mode = StreamMode::Poisoned;
                    return Err(error);
                }
            };
            self.pending_input.drain(..hop_samples);
            stage_aligned_output(
                &self.hop_output,
                &mut self.chunk_latency_to_discard,
                &mut staged,
            );
        }

        let produced = staged.len();
        output.extend_from_slice(&staged);
        self.chunk_output_samples += produced;
        Ok(StreamProgress {
            consumed: input.len(),
            produced,
        })
    }

    /// Emit the delayed tail exactly once and finalize the current stream.
    ///
    /// For arbitrary chunks, the appended output completes a same-length,
    /// latency-compensated stream. For fixed-hop calls, it contains the raw
    /// `latency_samples_per_channel * channels` tail because startup padding was
    /// already returned by `process_frame`.
    pub fn finalize(&mut self, output: &mut Vec<f32>) -> Result<StreamProgress> {
        self.ensure_usable()?;
        let mode = self.mode;
        let mut staged = Vec::new();
        let hop_samples = self.hop_samples();

        if mode == StreamMode::Chunked && !self.pending_input.is_empty() {
            let mut padded = vec![0.0; hop_samples];
            padded[..self.pending_input.len()].copy_from_slice(&self.pending_input);
            match self.process_hop_internal(&padded) {
                Ok(_) => {}
                Err(error) => return self.poison(error),
            }
            stage_aligned_output(
                &self.hop_output,
                &mut self.chunk_latency_to_discard,
                &mut staged,
            );
        }

        if matches!(mode, StreamMode::FixedHop | StreamMode::Chunked) {
            let zeros = vec![0.0; hop_samples];
            for _ in 0..latency_hops(self.config.fft_size, self.config.hop_size) {
                match self.process_hop_internal(&zeros) {
                    Ok(_) => {}
                    Err(error) => return self.poison(error),
                }
                if mode == StreamMode::Chunked {
                    stage_aligned_output(
                        &self.hop_output,
                        &mut self.chunk_latency_to_discard,
                        &mut staged,
                    );
                } else {
                    staged.extend_from_slice(&self.hop_output);
                }
            }
        }

        if mode == StreamMode::Chunked {
            let projected = self.chunk_output_samples + staged.len();
            if projected < self.chunk_input_samples {
                return self.poison(SoundExError::InvalidInput(
                    "stream flush produced too few samples".into(),
                ));
            }
            let excess = projected - self.chunk_input_samples;
            if excess > staged.len() {
                return self.poison(SoundExError::InvalidInput(
                    "stream emitted more samples than it consumed".into(),
                ));
            }
            staged.truncate(staged.len() - excess);
            self.chunk_output_samples += staged.len();
            self.pending_input.clear();
        }

        let produced = staged.len();
        output.extend_from_slice(&staged);
        self.mode = StreamMode::Finalized;
        Ok(StreamProgress {
            consumed: 0,
            produced,
        })
    }

    /// Process an interleaved PCM buffer and return a same-length buffer.
    ///
    /// Offline processing uses the same causal hop framing, startup history,
    /// padded final hop, and tail drain as the streaming APIs.
    pub fn process_buffer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.ensure_usable()?;
        if self.mode != StreamMode::Ready {
            return Err(SoundExError::StreamModeConflict);
        }
        let channel_count = self.channel_count();
        if !input.len().is_multiple_of(channel_count) {
            return Err(SoundExError::InvalidInput(format!(
                "interleaved input length {} is not divisible by {channel_count} channels",
                input.len()
            )));
        }
        validate_finite(input)?;
        if input.is_empty() {
            self.reset();
            return Ok(Vec::new());
        }

        self.reset();
        let mut raw = Vec::new();
        for hop in CausalHopIter::new(
            input,
            channel_count,
            self.config.fft_size,
            self.config.hop_size,
        ) {
            match self.process_hop_internal(&hop) {
                Ok(_) => raw.extend_from_slice(&self.hop_output),
                Err(error) => {
                    self.mode = StreamMode::Poisoned;
                    return Err(error);
                }
            }
        }

        let trim = self.latency_samples_per_channel() * channel_count;
        let result = raw
            .get(trim..trim + input.len())
            .ok_or_else(|| {
                SoundExError::InvalidInput("offline flush produced too few samples".into())
            })?
            .to_vec();
        let info = self.last_info;
        self.reset();
        self.last_info = info;
        Ok(result)
    }

    /// Replace the model used for subsequent frames and reset stream state.
    pub fn reload_model(&mut self, model_path: impl Into<std::path::PathBuf>) -> Result<()> {
        let path = model_path.into();
        let engine = InferenceEngine::load(&path, &self.config)?;
        self.engine = engine;
        self.config.model_path = path;
        self.reset();
        Ok(())
    }

    /// Reset all channel, gate, overlap-add, and stream lifecycle state.
    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.reset();
        }
        self.gate.reset();
        self.last_info = ProcessInfo::default();
        self.mode = StreamMode::Ready;
        self.pending_input.clear();
        self.chunk_input_samples = 0;
        self.chunk_output_samples = 0;
        self.chunk_latency_to_discard = self.latency_samples_per_channel() * self.channel_count();
    }

    /// Return whether the latest processed frame bypassed model enhancement.
    pub fn is_bypassed(&self) -> bool {
        self.last_info.bypassed
    }

    /// Return whether a processing failure requires a reset.
    pub fn is_poisoned(&self) -> bool {
        self.mode == StreamMode::Poisoned
    }

    /// Configured samples per channel in one streaming hop.
    pub fn hop_size(&self) -> usize {
        self.config.hop_size
    }

    /// Configured FFT size in samples per channel.
    pub fn fft_size(&self) -> usize {
        self.config.fft_size
    }

    /// Causal STFT delay in samples per channel.
    pub fn latency_samples_per_channel(&self) -> usize {
        self.config.fft_size - self.config.hop_size
    }

    /// Number of ONNX Runtime runs issued since construction or model reload.
    pub fn inference_run_count(&self) -> u64 {
        self.engine.run_count()
    }

    fn process_hop_internal(&mut self, input: &[f32]) -> Result<ProcessInfo> {
        debug_assert_eq!(input.len(), self.hop_samples());
        let channel_count = self.channel_count();
        for channel_index in 0..channel_count {
            for sample in 0..self.config.hop_size {
                self.channel_inputs[channel_index][sample] =
                    input[sample * channel_count + channel_index];
            }
            self.channels[channel_index]
                .analyze_hop(&self.channel_inputs[channel_index], &self.config)?;
        }

        self.active_channels.clear();
        for (channel_index, channel) in self.channels.iter().enumerate() {
            if channel.needs_enhancement() {
                let batch_index = self.active_channels.len();
                channel.write_model_input(&mut self.inference_input, batch_index);
                self.active_channels.push(channel_index);
            }
        }
        let active_count = self.active_channels.len();
        if active_count > 0 {
            self.engine.infer_into(
                self.inference_input.slice(s![..active_count, .., .., ..]),
                self.inference_output
                    .slice_mut(s![..active_count, .., .., ..]),
            )?;
        }

        for channel_index in 0..channel_count {
            let prediction = self
                .active_channels
                .iter()
                .position(|&active_channel| active_channel == channel_index)
                .map(|batch_index| self.inference_output.slice(s![batch_index, .., .., ..]));
            self.channel_infos[channel_index] =
                self.channels[channel_index].finish_hop(prediction, &self.config)?;
        }
        let info = aggregate_info(&self.channel_infos);
        for channel_index in 0..channel_count {
            for sample in 0..self.config.hop_size {
                let output_index = sample * channel_count + channel_index;
                self.dry_interleaved[output_index] = self.channels[channel_index].dry()[sample];
                self.wet_interleaved[output_index] = self.channels[channel_index].wet()[sample];
            }
        }
        // Codec decoders may produce finite PCM beyond full scale. Protect the
        // delayed dry path too; the wet path is already limited per channel.
        // Limiting each path once keeps the raised-cosine blend bounded without
        // compressing the enhanced signal a second time.
        self.dry_limiter.process_buffer(&mut self.dry_interleaved);
        self.gate.mix_interleaved_into(
            &self.dry_interleaved,
            &self.wet_interleaved,
            channel_count,
            !info.bypassed,
            &mut self.hop_output,
        );
        for sample in &mut self.hop_output {
            *sample = sample.clamp(-self.config.limiter_ceiling, self.config.limiter_ceiling);
        }
        self.last_info = info;
        Ok(info)
    }

    fn ensure_usable(&self) -> Result<()> {
        match self.mode {
            StreamMode::Finalized => Err(SoundExError::StreamFinalized),
            StreamMode::Poisoned => Err(SoundExError::ProcessorPoisoned),
            StreamMode::Ready | StreamMode::FixedHop | StreamMode::Chunked => Ok(()),
        }
    }

    fn poison<T>(&mut self, error: SoundExError) -> Result<T> {
        self.mode = StreamMode::Poisoned;
        Err(error)
    }

    fn channel_count(&self) -> usize {
        self.config.channels as usize
    }

    fn hop_samples(&self) -> usize {
        self.config.hop_size * self.channel_count()
    }
}

fn stage_aligned_output(raw: &[f32], latency_to_discard: &mut usize, staged: &mut Vec<f32>) {
    let discard = (*latency_to_discard).min(raw.len());
    *latency_to_discard -= discard;
    staged.extend_from_slice(&raw[discard..]);
}

fn validate_finite(input: &[f32]) -> Result<()> {
    if input.iter().any(|sample| !sample.is_finite()) {
        return Err(SoundExError::InvalidInput(
            "input contains non-finite samples".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_config(config: &SoundExConfig) -> Result<()> {
    if config.sample_rate == 0 {
        return Err(SoundExError::InvalidInput("sample_rate must be > 0".into()));
    }
    if config.channels == 0 || config.channels > 2 {
        return Err(SoundExError::InvalidInput("channels must be 1 or 2".into()));
    }
    if config.fft_size < 2 || !config.fft_size.is_power_of_two() {
        return Err(SoundExError::InvalidInput(
            "fft_size must be a power of two >= 2".into(),
        ));
    }
    if config.hop_size == 0
        || config.hop_size > config.fft_size
        || !config.fft_size.is_multiple_of(config.hop_size)
    {
        return Err(SoundExError::InvalidInput(
            "hop_size must divide fft_size and be in 1..=fft_size".into(),
        ));
    }
    if !config.bypass_threshold_db.is_finite()
        || config.bypass_threshold_db > 0.0
        || !(0.0..=1.0).contains(&config.min_bandwidth_ratio)
    {
        return Err(SoundExError::InvalidInput(
            "invalid bandwidth detector configuration".into(),
        ));
    }
    if !config.crossover_width_hz.is_finite() || config.crossover_width_hz < 0.0 {
        return Err(SoundExError::InvalidInput(
            "crossover_width_hz must be finite and non-negative".into(),
        ));
    }
    if !config.limiter_ceiling.is_finite() || !(0.0..=1.0).contains(&config.limiter_ceiling) {
        return Err(SoundExError::InvalidInput(
            "limiter_ceiling must be in 0..=1".into(),
        ));
    }
    if config.ort_intra_threads == 0 || config.ort_inter_threads == 0 {
        return Err(SoundExError::InvalidInput(
            "ORT thread counts must be positive".into(),
        ));
    }
    Ok(())
}

fn aggregate_info(infos: &[ProcessInfo]) -> ProcessInfo {
    ProcessInfo {
        bypassed: infos.iter().all(|info| info.bypassed),
        detected_bandwidth_hz: infos
            .iter()
            .map(|info| info.detected_bandwidth_hz)
            .fold(f32::INFINITY, f32::min),
        enhancement_gain_db: infos
            .iter()
            .filter(|info| !info.bypassed)
            .map(|info| info.enhancement_gain_db)
            .reduce(f32::max)
            .unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ort-backend")]
    use std::path::PathBuf;

    use super::*;

    #[cfg(feature = "ort-backend")]
    fn identity_model() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx")
    }

    #[test]
    fn gate_uses_one_raised_cosine_ramp_for_all_channels() {
        let ramp_samples = 8;
        let mut gate = GateMixer::new(ramp_samples, 4);
        let dry = vec![0.0; 16 * 2];
        let wet = vec![1.0; 16 * 2];

        let attack = gate.mix_interleaved(&dry, &wet, 2, true);

        for frame in attack.as_chunks::<2>().0 {
            assert_eq!(frame[0].to_bits(), frame[1].to_bits());
        }
        let theoretical_step_bound = std::f32::consts::PI / (2.0 * ramp_samples as f32);
        let max_step = attack
            .as_chunks::<2>()
            .0
            .iter()
            .map(|frame| frame[0])
            .collect::<Vec<_>>()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0, f32::max);
        assert!(max_step <= theoretical_step_bound + 1e-6);

        let release = gate.mix_interleaved(&dry, &wet, 2, false);
        assert!(release[..4 * 2].iter().all(|sample| *sample == 1.0));
        let release_max_step = release
            .as_chunks::<2>()
            .0
            .iter()
            .map(|frame| frame[0])
            .collect::<Vec<_>>()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0, f32::max);
        assert!(release_max_step <= theoretical_step_bound + 1e-6);
    }

    #[test]
    fn aggregate_info_preserves_reported_gain_reduction() {
        let mono = aggregate_info(&[ProcessInfo {
            bypassed: false,
            detected_bandwidth_hz: 12_000.0,
            enhancement_gain_db: -6.0,
        }]);
        assert_eq!(mono.enhancement_gain_db, -6.0);

        let stereo = aggregate_info(&[
            ProcessInfo {
                bypassed: false,
                detected_bandwidth_hz: 12_000.0,
                enhancement_gain_db: -6.0,
            },
            ProcessInfo {
                bypassed: false,
                detected_bandwidth_hz: 14_000.0,
                enhancement_gain_db: -3.0,
            },
        ]);
        assert_eq!(stereo.enhancement_gain_db, -3.0);

        let asymmetric = aggregate_info(&[
            ProcessInfo {
                bypassed: false,
                detected_bandwidth_hz: 12_000.0,
                enhancement_gain_db: -6.0,
            },
            ProcessInfo {
                bypassed: true,
                detected_bandwidth_hz: 22_050.0,
                enhancement_gain_db: 0.0,
            },
        ]);
        assert_eq!(asymmetric.enhancement_gain_db, -6.0);
    }

    #[cfg(feature = "ort-backend")]
    #[test]
    fn batched_stereo_failure_keeps_output_unchanged_and_poisons_processor() {
        let config = SoundExConfig::with_model(identity_model())
            .fft_size(1024)
            .hop_size(512)
            .channels(2);
        let mut processor = SoundExProcessor::new(config).unwrap();
        processor.engine.fail_after_successes(0);
        let input: Vec<f32> = (0..processor.hop_size())
            .flat_map(|sample| {
                let value =
                    0.5 * (2.0 * std::f32::consts::PI * 440.0 * sample as f32 / 44_100.0).sin();
                [value, value]
            })
            .collect();
        let mut output = vec![0.375; input.len()];

        let error = processor.process_frame(&input, &mut output).unwrap_err();

        assert!(matches!(error, SoundExError::Inference(_)));
        assert_eq!(output, vec![0.375; input.len()]);
        assert!(processor.is_poisoned());
        assert!(matches!(
            processor.process_frame(&input, &mut output),
            Err(SoundExError::ProcessorPoisoned)
        ));

        processor.reset();
        processor.process_frame(&input, &mut output).unwrap();
        assert!(!processor.is_poisoned());
    }
}
