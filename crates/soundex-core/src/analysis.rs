//! Model-free bandwidth analysis used by dry-run workflows.

use soundex_dsp::bandwidth::{BandwidthDetector, BandwidthDetectorConfig};
use soundex_dsp::stft::StftAnalyzer;

use crate::config::SoundExConfig;
use crate::error::{Result, SoundExError};
use crate::processor::validate_config;
use crate::stream::{CausalHopIter, StreamBuffer};

/// Summary of frame-level bandwidth gate decisions.
#[derive(Debug, Clone, Copy)]
pub struct AnalysisInfo {
    /// Number of hop-spaced spectral frames analyzed.
    pub frames_analyzed: usize,
    /// Number of frames for which at least one channel needs enhancement.
    pub frames_needing_enhancement: usize,
    /// Mean detected bandwidth across analyzed frames, in Hz.
    pub detected_bandwidth_hz: f32,
}

impl AnalysisInfo {
    /// Fraction of frames that need enhancement, in the inclusive range 0..=1.
    pub fn enhancement_ratio(&self) -> f32 {
        if self.frames_analyzed == 0 {
            0.0
        } else {
            self.frames_needing_enhancement as f32 / self.frames_analyzed as f32
        }
    }

    /// Whether any analyzed frame needs enhancement.
    pub fn needs_enhancement(&self) -> bool {
        self.frames_needing_enhancement > 0
    }
}

/// Analyze an interleaved PCM buffer without loading or running a model.
pub fn analyze_buffer(config: &SoundExConfig, input: &[f32]) -> Result<AnalysisInfo> {
    validate_config(config)?;
    let channel_count = config.channels as usize;
    if !input.len().is_multiple_of(channel_count) {
        return Err(SoundExError::InvalidInput(format!(
            "interleaved input length {} is not divisible by {} channels",
            input.len(),
            channel_count
        )));
    }
    if input.iter().any(|sample| !sample.is_finite()) {
        return Err(SoundExError::InvalidInput(
            "input contains non-finite samples".into(),
        ));
    }
    let samples_per_channel = input.len() / channel_count;
    if samples_per_channel == 0 {
        return Ok(AnalysisInfo {
            frames_analyzed: 0,
            frames_needing_enhancement: 0,
            detected_bandwidth_hz: 0.0,
        });
    }

    let mut analyzers: Vec<_> = (0..channel_count)
        .map(|_| StftAnalyzer::new(config.fft_size, config.hop_size))
        .collect();
    let mut frame_buffers: Vec<_> = (0..channel_count)
        .map(|_| StreamBuffer::with_latency_padding(config.fft_size, config.hop_size))
        .collect();
    let detector_config = BandwidthDetectorConfig {
        threshold_db: config.bypass_threshold_db,
        min_bandwidth_ratio: config.min_bandwidth_ratio,
        ..Default::default()
    };
    let mut detectors: Vec<_> = (0..channel_count)
        .map(|_| BandwidthDetector::new(detector_config.clone()))
        .collect();

    let mut frames_analyzed = 0;
    let mut frames_needing_enhancement = 0;
    let mut bandwidth_sum = 0.0;
    for hop in CausalHopIter::new(input, channel_count, config.fft_size, config.hop_size) {
        let mut frame_bandwidth = f32::INFINITY;
        let mut frame_needs_enhancement = false;
        for channel in 0..channel_count {
            let channel_hop: Vec<f32> = (0..config.hop_size)
                .map(|sample| hop[sample * channel_count + channel])
                .collect();
            frame_buffers[channel].push(&channel_hop)?;
            let spectral = analyzers[channel].analyze(frame_buffers[channel].current_frame());
            let bandwidth = detectors[channel].detect(&spectral.log_magnitude, config.sample_rate);
            frame_bandwidth = frame_bandwidth.min(bandwidth);
            frame_needs_enhancement |=
                detectors[channel].needs_enhancement(bandwidth, config.sample_rate as f32 / 2.0);
            frame_buffers[channel].advance();
        }
        frames_analyzed += 1;
        bandwidth_sum += frame_bandwidth;
        frames_needing_enhancement += usize::from(frame_needs_enhancement);
    }

    Ok(AnalysisInfo {
        frames_analyzed,
        frames_needing_enhancement,
        detected_bandwidth_hz: bandwidth_sum / frames_analyzed as f32,
    })
}

#[cfg(all(test, feature = "ort-backend"))]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::processor::SoundExProcessor;

    fn identity_model() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx")
    }

    #[test]
    fn dry_run_decisions_match_causal_processing_frames() {
        let input: Vec<f32> = (0..3333)
            .map(|sample| {
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * sample as f32 / 44_100.0).sin()
            })
            .collect();
        let config = SoundExConfig::with_model(identity_model())
            .fft_size(1024)
            .hop_size(512);
        let expected = analyze_buffer(&config, &input).unwrap();
        let mut processor = SoundExProcessor::new(config.clone()).unwrap();
        let mut frames = 0;
        let mut needing_enhancement = 0;
        let mut bandwidth_sum = 0.0;
        for hop in CausalHopIter::new(&input, 1, config.fft_size, config.hop_size) {
            let mut output = vec![0.0; config.hop_size];
            let info = processor.process_frame(&hop, &mut output).unwrap();
            frames += 1;
            needing_enhancement += usize::from(!info.bypassed);
            bandwidth_sum += info.detected_bandwidth_hz;
        }

        assert_eq!(frames, expected.frames_analyzed);
        assert_eq!(needing_enhancement, expected.frames_needing_enhancement);
        assert!((bandwidth_sum / frames as f32 - expected.detected_bandwidth_hz).abs() < 1e-3);
    }
}
