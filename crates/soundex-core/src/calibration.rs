//! Offline calibration observations from the actual per-channel streaming analyzer.
//! Collection allocates and is never called by the audio callback.

use crate::{
    channel::ChannelProcessor,
    config::SoundExConfig,
    error::{Result, SoundExError},
    processor::validate_config,
    stream::{CausalHopIter, StreamBuffer},
};
use soundex_dsp::gain_calibration::{gain_features, GainCalibration, GAIN_FEATURES};
use soundex_dsp::loudness::rms;
use soundex_dsp::stft::StftAnalyzer;

pub struct GainObservation {
    pub hop: usize,
    pub channel: usize,
    pub cutoff_hz: f32,
    pub active: bool,
    pub features: [f32; GAIN_FEATURES],
    pub edge_rms: f32,
    pub clean_high_rms: f32,
}

/// Same padding, detector, silence reset and history readiness as production.
/// The clean signal is used only for the offline label, never for features or gates.
pub fn collect_gain_observations(
    config: &SoundExConfig,
    degraded: &[f32],
    clean: &[f32],
    context: usize,
) -> Result<Vec<GainObservation>> {
    validate_config(config)?;
    if ![1, 9].contains(&context)
        || degraded.len() != clean.len()
        || !degraded.len().is_multiple_of(config.channels as usize)
        || degraded.iter().chain(clean).any(|x| !x.is_finite())
    {
        return Err(SoundExError::InvalidInput(
            "invalid aligned calibration pair/context".into(),
        ));
    }
    let channels = config.channels as usize;
    let mut processors: Vec<_> = (0..channels)
        .map(|_| ChannelProcessor::new(config, context))
        .collect();
    let mut reference_buffers: Vec<_> = (0..channels)
        .map(|_| StreamBuffer::with_latency_padding(config.fft_size, config.hop_size))
        .collect();
    let mut analyzers: Vec<_> = (0..channels)
        .map(|_| StftAnalyzer::new(config.fft_size, config.hop_size))
        .collect();
    let hops = CausalHopIter::new(degraded, channels, config.fft_size, config.hop_size).zip(
        CausalHopIter::new(clean, channels, config.fft_size, config.hop_size),
    );
    let mut result = Vec::new();
    for (hop, (input, target)) in hops.enumerate() {
        for channel in 0..channels {
            let input: Vec<f32> = input
                .iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .collect();
            let target: Vec<f32> = target
                .iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .collect();
            let processor = &mut processors[channel];
            processor.analyze_hop(&input, config)?;
            let info = processor.info();
            reference_buffers[channel].push(&target)?;
            let clean_frame =
                analyzers[channel].analyze(reference_buffers[channel].current_frame());
            reference_buffers[channel].advance();
            let original: Vec<f32> = processor
                .observed_spectrum()
                .log_magnitude
                .iter()
                .map(|x| 10.0_f32.powf(x / 20.0))
                .collect();
            let clean_magnitude: Vec<f32> = clean_frame
                .log_magnitude
                .iter()
                .map(|x| 10.0_f32.powf(x / 20.0))
                .collect();
            let resolution = config.sample_rate as f32 / config.fft_size as f32;
            let cutoff = ((info.detected_bandwidth_hz / resolution).round() as usize)
                .clamp(1, original.len() - 1);
            let reference = ((config.crossover_width_hz / resolution).round() as usize)
                .max(1)
                .min(cutoff);
            result.push(GainObservation {
                hop,
                channel,
                cutoff_hz: info.detected_bandwidth_hz,
                active: !info.bypassed,
                features: gain_features(&original, cutoff, reference, config.sample_rate),
                edge_rms: rms(&original[cutoff - reference..cutoff]),
                clean_high_rms: rms(&clean_magnitude[cutoff..]),
            });
        }
    }
    Ok(result)
}

/// Strict numeric contract. Metadata may carry local provenance; the caller binds
/// the whole artifact hash in its run receipt. No file access in the audio callback.
pub fn read_gain_calibration(path: impl AsRef<std::path::Path>) -> Result<GainCalibration> {
    let bytes = std::fs::read(path).map_err(|e| SoundExError::InvalidInput(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| SoundExError::InvalidInput(e.to_string()))?;
    let invalid = || SoundExError::InvalidInput("invalid gain calibration v1 contract".into());
    if value["schema_version"] != 1
        || value["feature_contract"] != "soundex-edge-gain-v1"
        || value["fft_size"] != 256
        || value["hop_size"] != 128
        || value["crossover_width_hz"] != 1000
        || value["base_ratio_db"] != -6
        || value["correction_limit_db"] != 6
    {
        return Err(invalid());
    }
    let numbers = value["coefficients"].as_array().ok_or_else(invalid)?;
    if numbers.len() != GAIN_FEATURES {
        return Err(invalid());
    }
    let mut coefficients = [0.0; GAIN_FEATURES];
    for (slot, number) in coefficients.iter_mut().zip(numbers) {
        *slot = number.as_f64().ok_or_else(invalid)? as f32;
    }
    GainCalibration::new(coefficients).map_err(|e| SoundExError::InvalidInput(e.into()))
}

#[cfg(all(test, feature = "ort-backend"))]
mod tests {
    use super::*;
    use crate::SoundExProcessor;

    #[test]
    fn observations_follow_real_channel_silence_reset_and_context() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/low-latency-identity.onnx");
        let config = SoundExConfig::with_model(model).sample_rate(48000);
        let input: Vec<f32> = (0..8192)
            .map(|i| {
                if (2048..4096).contains(&i) {
                    0.0
                } else {
                    0.2 * (i as f32 * 0.11).sin()
                }
            })
            .collect();
        let trace = collect_gain_observations(&config, &input, &input, 1).unwrap();
        let mut processor = SoundExProcessor::new(config.clone()).unwrap();
        for (row, hop) in trace.iter().zip(CausalHopIter::new(&input, 1, 256, 128)) {
            let info = processor.process_frame(&hop, &mut [0.0; 128]).unwrap();
            assert_eq!(row.active, !info.bypassed);
            assert_eq!(row.cutoff_hz, info.detected_bandwidth_hz);
        }
        let history = collect_gain_observations(&config, &input, &input, 9).unwrap();
        assert!(history[..8].iter().all(|x| !x.active));
        assert!(history.iter().any(|x| x.active));
        let changed =
            collect_gain_observations(&config, &input, &vec![0.0; input.len()], 9).unwrap();
        for (a, b) in history.iter().zip(changed) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.active, b.active);
        }
    }
}
