//! Offline clean-reference oracles. These allocate and must never run in a callback.
//! Only the model prediction is replaced; detection, gain, fusion, OLA, gate and
//! limiting remain the production path. Oracle output is not deployable quality.

use crate::{
    processor::validate_config,
    stream::{CausalHopIter, StreamBuffer},
    ProcessInfo, Result, SoundExConfig, SoundExError, SoundExProcessor,
};
use ndarray::Array4;
use soundex_dsp::stft::{SpectralFrame, StftAnalyzer};

#[derive(Clone, Copy)]
pub enum OracleSpectrum {
    Observed,
    CleanMagnitude,
    CleanPhase,
    CleanComplex,
}

pub struct OracleOutput {
    pub audio: Vec<f32>,
    pub hops: Vec<ProcessInfo>,
}

/// Same input and reference sample clock, padding and tail; no reference-based gates.
pub fn process_oracle(
    config: SoundExConfig,
    degraded: &[f32],
    reference: &[f32],
    spectrum: OracleSpectrum,
) -> Result<OracleOutput> {
    validate_config(&config)?;
    let channels = config.channels as usize;
    if degraded.len() != reference.len()
        || !degraded.len().is_multiple_of(channels)
        || degraded.iter().chain(reference).any(|x| !x.is_finite())
    {
        return Err(SoundExError::InvalidInput(
            "oracle needs a finite aligned pair".into(),
        ));
    }
    let mut processor = SoundExProcessor::new(config.clone())?;
    let mut buffers: Vec<_> = (0..2 * channels)
        .map(|_| StreamBuffer::with_latency_padding(config.fft_size, config.hop_size))
        .collect();
    let mut analyzer = StftAnalyzer::new(config.fft_size, config.hop_size);
    let mut frames = vec![SpectralFrame::zeros(config.fft_size / 2 + 1); 2 * channels];
    let mut prediction = Array4::zeros((channels, 2, 1, config.fft_size / 2 + 1));
    let mut output = vec![0.0; config.hop_size * channels];
    let mut mono = vec![0.0; config.hop_size];
    let mut raw = Vec::new();
    let mut hops = Vec::new();
    for (input, clean) in
        CausalHopIter::new(degraded, channels, config.fft_size, config.hop_size).zip(
            CausalHopIter::new(reference, channels, config.fft_size, config.hop_size),
        )
    {
        for (source, pcm) in [&input, &clean].into_iter().enumerate() {
            for channel in 0..channels {
                let index = source * channels + channel;
                for (sample, value) in mono.iter_mut().enumerate() {
                    *value = pcm[sample * channels + channel];
                }
                buffers[index].push(&mono)?;
                analyzer.analyze_into(buffers[index].current_frame(), &mut frames[index]);
                buffers[index].advance();
            }
        }
        for channel in 0..channels {
            let observed = &frames[channel];
            let clean = &frames[channels + channel];
            let magnitude = if matches!(
                spectrum,
                OracleSpectrum::CleanMagnitude | OracleSpectrum::CleanComplex
            ) {
                &clean.log_magnitude
            } else {
                &observed.log_magnitude
            };
            let phase = if matches!(
                spectrum,
                OracleSpectrum::CleanPhase | OracleSpectrum::CleanComplex
            ) {
                &clean.phase
            } else {
                &observed.phase
            };
            for bin in 0..magnitude.len() {
                prediction[[channel, 0, 0, bin]] = magnitude[bin];
                prediction[[channel, 1, 0, bin]] = phase[bin];
            }
        }
        hops.push(processor.process_oracle_frame(&input, &mut output, prediction.view())?);
        raw.extend_from_slice(&output);
    }
    let trim = processor.latency_samples_per_channel() * channels;
    let audio = if degraded.is_empty() {
        Vec::new()
    } else {
        raw.get(trim..trim + degraded.len())
            .ok_or_else(|| SoundExError::InvalidInput("oracle tail is incomplete".into()))?
            .to_vec()
    };
    Ok(OracleOutput { audio, hops })
}

#[cfg(all(test, feature = "ort-backend"))]
mod tests {
    use super::*;
    fn config(channels: u16) -> SoundExConfig {
        let mut config = SoundExConfig::with_model(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/low-latency-identity.onnx"),
        )
        .channels(channels);
        config.min_bandwidth_ratio = 1.0;
        config
    }
    #[test]
    fn observed_replay_equals_production_including_partial_tail_and_silent_left() {
        for channels in [1, 2] {
            let input: Vec<f32> = (0..4093)
                .flat_map(|n| {
                    let value = 0.2 * (n as f32 * 0.057).sin();
                    if channels == 1 {
                        vec![value]
                    } else {
                        vec![0.0, value]
                    }
                })
                .collect();
            let config = config(channels);
            let normal = SoundExProcessor::new(config.clone())
                .unwrap()
                .process_buffer(&input)
                .unwrap();
            let oracle = process_oracle(config, &input, &input, OracleSpectrum::Observed).unwrap();
            assert_eq!(normal, oracle.audio);
            assert!(oracle.hops.iter().any(|h| !h.bypassed));
            if channels == 2 {
                assert!(oracle.audio.iter().step_by(2).all(|&x| x == 0.0));
            }
        }
    }
    #[test]
    fn oracle_preserves_silence_and_rejects_invalid_pairs() {
        let silence = vec![0.0; 257];
        let result = process_oracle(
            config(1),
            &silence,
            &vec![0.5; 257],
            OracleSpectrum::CleanComplex,
        )
        .unwrap();
        assert!(result.audio.iter().all(|&x| x == 0.0));
        assert!(result.hops.iter().all(|h| h.bypassed));
        assert!(process_oracle(config(1), &[0.0], &[], OracleSpectrum::CleanComplex).is_err());
        assert!(
            process_oracle(config(1), &[f32::NAN], &[0.0], OracleSpectrum::CleanComplex).is_err()
        );
    }
}
