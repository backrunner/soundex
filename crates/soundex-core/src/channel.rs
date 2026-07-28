//! Per-channel streaming DSP around one shared batched inference call.

use ndarray::{Array4, ArrayView3};
use soundex_dsp::bandwidth::{BandwidthDetector, BandwidthDetectorConfig};
use soundex_dsp::crossover::CrossoverBlend;
use soundex_dsp::limiter::Limiter;
use soundex_dsp::loudness::{rms, LoudnessMatcher};
use soundex_dsp::phase::PhaseSmoother;
use soundex_dsp::stft::{SpectralFrame, StftAnalyzer, StftSynthesizer};

use crate::config::SoundExConfig;
use crate::error::{Result, SoundExError};
use crate::processor::ProcessInfo;
use crate::stream::StreamBuffer;

pub(crate) struct ChannelProcessor {
    analyzer: StftAnalyzer,
    synthesizer: StftSynthesizer,
    detector: BandwidthDetector,
    loudness: LoudnessMatcher,
    phase_smoother: PhaseSmoother,
    crossover: CrossoverBlend,
    limiter: Limiter,
    input: StreamBuffer,
    original: SpectralFrame,
    synthesis: SpectralFrame,
    original_magnitude: Vec<f32>,
    generated_magnitude: Vec<f32>,
    blended_magnitude: Vec<f32>,
    predicted_phase: Vec<f32>,
    crossover_weights: Vec<f32>,
    dry: Vec<f32>,
    wet: Vec<f32>,
    needs_enhancement: bool,
    detected_bandwidth_hz: f32,
    enhancement_gain_db: f32,
}

impl ChannelProcessor {
    pub(crate) fn new(config: &SoundExConfig) -> Self {
        let detector = BandwidthDetector::new(BandwidthDetectorConfig {
            threshold_db: config.bypass_threshold_db,
            min_bandwidth_ratio: config.min_bandwidth_ratio,
            ..Default::default()
        });
        let bins = config.fft_size / 2 + 1;
        Self {
            analyzer: StftAnalyzer::new(config.fft_size, config.hop_size),
            synthesizer: StftSynthesizer::new(config.fft_size, config.hop_size),
            detector,
            loudness: LoudnessMatcher::new(-6.0, 0.1),
            phase_smoother: PhaseSmoother::new(),
            crossover: CrossoverBlend::new(config.crossover_width_hz),
            limiter: Limiter::new(config.limiter_ceiling, 10.0),
            input: StreamBuffer::with_latency_padding(config.fft_size, config.hop_size),
            original: SpectralFrame::zeros(bins),
            synthesis: SpectralFrame::zeros(bins),
            original_magnitude: vec![0.0; bins],
            generated_magnitude: vec![0.0; bins],
            blended_magnitude: vec![0.0; bins],
            predicted_phase: vec![0.0; bins],
            crossover_weights: vec![0.0; bins],
            dry: vec![0.0; config.hop_size],
            wet: vec![0.0; config.hop_size],
            needs_enhancement: false,
            detected_bandwidth_hz: 0.0,
            enhancement_gain_db: 0.0,
        }
    }

    pub(crate) fn analyze_hop(&mut self, input: &[f32], config: &SoundExConfig) -> Result<()> {
        self.input.push(input)?;
        if !self.input.is_frame_ready() {
            return Err(SoundExError::InvalidInput(
                "internal stream did not receive a complete hop".into(),
            ));
        }

        let frame = self.input.current_frame();
        self.dry.copy_from_slice(&frame[..config.hop_size]);
        self.analyzer.analyze_into(frame, &mut self.original);
        self.detected_bandwidth_hz = self
            .detector
            .detect(&self.original.log_magnitude, config.sample_rate);
        self.needs_enhancement = self
            .detector
            .needs_enhancement(self.detected_bandwidth_hz, config.sample_rate as f32 / 2.0);
        self.enhancement_gain_db = 0.0;
        self.input.advance();
        Ok(())
    }

    pub(crate) fn write_model_input(&self, input: &mut Array4<f32>, batch_index: usize) {
        for bin in 0..self.original.log_magnitude.len() {
            input[[batch_index, 0, 0, bin]] = self.original.log_magnitude[bin];
            input[[batch_index, 1, 0, bin]] = self.original.phase[bin];
        }
    }

    pub(crate) fn finish_hop(
        &mut self,
        prediction: Option<ArrayView3<'_, f32>>,
        config: &SoundExConfig,
    ) -> Result<ProcessInfo> {
        if self.needs_enhancement {
            let prediction = prediction.ok_or_else(|| {
                SoundExError::Inference("active channel has no batched model output".into())
            })?;
            self.enhance(prediction, config)?;
        } else {
            self.synthesis
                .log_magnitude
                .copy_from_slice(&self.original.log_magnitude);
            self.synthesis.phase.copy_from_slice(&self.original.phase);
        }
        self.synthesizer
            .synthesize_into(&self.synthesis, &mut self.wet);
        self.limiter.process_buffer(&mut self.wet);
        Ok(self.info())
    }

    fn enhance(&mut self, prediction: ArrayView3<'_, f32>, config: &SoundExConfig) -> Result<()> {
        let bins = self.original.log_magnitude.len();
        if prediction.shape() != [2, 1, bins] {
            return Err(SoundExError::Inference(format!(
                "channel prediction must have shape [2, 1, {bins}], got {:?}",
                prediction.shape()
            )));
        }
        for bin in 0..bins {
            self.original_magnitude[bin] = 10.0f32.powf(self.original.log_magnitude[bin] / 20.0);
            self.generated_magnitude[bin] =
                10.0f32.powf(prediction[[0, 0, bin]].clamp(-200.0, 100.0) / 20.0);
            self.predicted_phase[bin] = prediction[[1, 0, bin]];
        }

        let bin_resolution = config.sample_rate as f32 / config.fft_size as f32;
        let cutoff_bin = ((self.detected_bandwidth_hz / bin_resolution).round() as usize)
            .clamp(1, bins.saturating_sub(1));
        let reference_bins = ((config.crossover_width_hz / bin_resolution).round() as usize)
            .max(1)
            .min(cutoff_bin);
        let low_rms = rms(&self.original_magnitude[cutoff_bin - reference_bins..cutoff_bin]);
        let high_rms = rms(&self.generated_magnitude[cutoff_bin..]);
        let gain = self.loudness.compute_gain(low_rms, high_rms);
        self.crossover.compute_weights_into(
            self.detected_bandwidth_hz,
            bin_resolution,
            &mut self.crossover_weights,
        );
        apply_generated_gain(&mut self.generated_magnitude, &self.crossover_weights, gain);
        self.crossover
            .apply_into(
                &self.original_magnitude,
                &self.generated_magnitude,
                &self.crossover_weights,
                &mut self.blended_magnitude,
            )
            .map_err(|error| SoundExError::Dsp(error.to_string()))?;
        self.phase_smoother
            .smooth_into(
                &self.original.phase,
                &self.predicted_phase,
                &self.crossover_weights,
                &mut self.synthesis.phase,
            )
            .map_err(|error| SoundExError::Dsp(error.to_string()))?;
        for bin in 0..bins {
            self.synthesis.log_magnitude[bin] = if self.crossover_weights[bin] == 0.0 {
                self.original.log_magnitude[bin]
            } else {
                20.0 * self.blended_magnitude[bin].max(1e-10).log10()
            };
        }
        self.enhancement_gain_db = 20.0 * gain.max(1e-10).log10();
        Ok(())
    }

    pub(crate) fn needs_enhancement(&self) -> bool {
        self.needs_enhancement
    }

    pub(crate) fn dry(&self) -> &[f32] {
        &self.dry
    }

    pub(crate) fn wet(&self) -> &[f32] {
        &self.wet
    }

    pub(crate) fn info(&self) -> ProcessInfo {
        ProcessInfo {
            bypassed: !self.needs_enhancement,
            detected_bandwidth_hz: self.detected_bandwidth_hz,
            enhancement_gain_db: self.enhancement_gain_db,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.synthesizer.reset();
        self.detector.reset();
        self.loudness.reset();
        self.input.reset_with_latency_padding();
        self.needs_enhancement = false;
        self.detected_bandwidth_hz = 0.0;
        self.enhancement_gain_db = 0.0;
        self.dry.fill(0.0);
        self.wet.fill(0.0);
    }
}

fn apply_generated_gain(generated_magnitude: &mut [f32], weights: &[f32], gain: f32) {
    debug_assert_eq!(generated_magnitude.len(), weights.len());
    for (value, &weight) in generated_magnitude.iter_mut().zip(weights) {
        if weight > 0.0 {
            *value *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_gain_covers_every_nonzero_crossover_weight() {
        let mut generated = vec![2.0; 4];
        let weights = vec![0.0, 0.1, 0.5, 1.0];

        apply_generated_gain(&mut generated, &weights, 3.0);

        assert_eq!(generated, vec![2.0, 6.0, 6.0, 6.0]);
        let blended = CrossoverBlend::new(1000.0)
            .apply(&[1.0; 4], &generated, &weights)
            .unwrap();
        assert_eq!(blended, vec![1.0, 1.5, 3.5, 6.0]);
    }

    #[test]
    fn zero_weight_db_bins_are_copied_without_round_trip_error() {
        let original_db = [-17.123_457, -9.75];
        let blended = [10.0f32.powf(original_db[0] / 20.0), 1.0];
        let weights = [0.0, 1.0];
        let result: Vec<f32> = original_db
            .iter()
            .zip(blended)
            .zip(weights)
            .map(|((&original, magnitude), weight)| {
                if weight == 0.0 {
                    original
                } else {
                    20.0 * magnitude.max(1e-10).log10()
                }
            })
            .collect();

        assert_eq!(result[0].to_bits(), original_db[0].to_bits());
    }
}
