#![cfg(feature = "ort-backend")]

use std::path::PathBuf;

use soundex_core::{SoundExConfig, SoundExProcessor};
use soundex_dsp::limiter::Limiter;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx")
}

fn tone(frequency: f32, length: usize) -> Vec<f32> {
    (0..length)
        .map(|sample| {
            0.5 * (2.0 * std::f32::consts::PI * frequency * sample as f32 / 44_100.0).sin()
        })
        .collect()
}

fn forced_gate_config(channels: u16, enhance: bool) -> SoundExConfig {
    let mut config = SoundExConfig::with_model(model_path())
        .fft_size(1024)
        .hop_size(512)
        .channels(channels);
    config.min_bandwidth_ratio = if enhance { 1.0 } else { 0.0 };
    config
}

fn interleave(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .flat_map(|(&left, &right)| [left, right])
        .collect()
}

#[test]
fn full_bandwidth_mono_is_exactly_bypassed() {
    let input = tone(20_000.0, 4096);
    let config = SoundExConfig::with_model(model_path())
        .fft_size(1024)
        .hop_size(512);
    let mut processor = SoundExProcessor::new(config).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    assert_eq!(output, input);
    assert!(processor.is_bypassed());
}

#[test]
fn full_bandwidth_stereo_preserves_interleaving() {
    let left = tone(20_000.0, 4096);
    let right = tone(19_000.0, 4096);
    let input: Vec<f32> = left
        .iter()
        .zip(&right)
        .flat_map(|(left, right)| [*left, *right])
        .collect();
    let config = SoundExConfig::with_model(model_path())
        .fft_size(1024)
        .hop_size(512)
        .channels(2);
    let mut processor = SoundExProcessor::new(config).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    assert_eq!(output, input);
}

#[test]
fn decoded_overfullscale_bypass_is_limited_and_chunk_invariant() {
    for channels in [1, 2] {
        let input: Vec<f32> = (0..4097)
            .flat_map(|index| {
                let value = 1.3 * (2.0 * std::f32::consts::PI * index as f32 / 37.0).sin();
                [value, -value].into_iter().take(channels as usize)
            })
            .collect();
        let config = forced_gate_config(channels, false);
        let limiter = Limiter::new(config.limiter_ceiling, 10.0);
        let expected: Vec<f32> = input
            .iter()
            .map(|&sample| limiter.process_sample(sample))
            .collect();
        let mut offline = SoundExProcessor::new(config.clone()).unwrap();
        let output = offline.process_buffer(&input).unwrap();
        assert_eq!(output, expected);
        assert!(output
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 0.95));
        assert_eq!(offline.inference_run_count(), 0);

        let mut streamed = SoundExProcessor::new(config).unwrap();
        let mut chunk_output = Vec::new();
        let mut position = 0;
        for frames in [31, 512, 997, 7].into_iter().cycle() {
            let end = (position + frames * channels as usize).min(input.len());
            streamed
                .process_chunk(&input[position..end], &mut chunk_output)
                .unwrap();
            position = end;
            if position == input.len() {
                break;
            }
        }
        streamed.finalize(&mut chunk_output).unwrap();
        assert_eq!(chunk_output, output);
    }
}

#[test]
fn low_bandwidth_input_runs_inference_and_remains_finite() {
    let input = tone(440.0, 4096);
    let config = SoundExConfig::with_model(model_path())
        .fft_size(1024)
        .hop_size(512);
    let mut processor = SoundExProcessor::new(config).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    assert_eq!(output.len(), input.len());
    assert!(output.iter().all(|sample| sample.is_finite()));
    assert!(output.iter().all(|sample| sample.abs() <= 0.95));
}

#[test]
fn active_channel_count_maps_to_at_most_one_session_run_per_hop() {
    let mono_input = tone(440.0, 512);
    let mut mono = SoundExProcessor::new(forced_gate_config(1, true)).unwrap();
    let mut mono_output = vec![0.0; mono_input.len()];
    mono.process_frame(&mono_input, &mut mono_output).unwrap();
    assert_eq!(mono.inference_run_count(), 1);

    let stereo_input = interleave(&mono_input, &tone(880.0, 512));
    let mut stereo = SoundExProcessor::new(forced_gate_config(2, true)).unwrap();
    let mut stereo_output = vec![0.0; stereo_input.len()];
    stereo
        .process_frame(&stereo_input, &mut stereo_output)
        .unwrap();
    assert_eq!(stereo.inference_run_count(), 1);

    let mut bypassed = SoundExProcessor::new(forced_gate_config(2, false)).unwrap();
    bypassed
        .process_frame(&stereo_input, &mut stereo_output)
        .unwrap();
    assert_eq!(bypassed.inference_run_count(), 0);
}

#[test]
fn one_active_stereo_channel_uses_one_single_item_batch() {
    let input = interleave(&tone(440.0, 512), &tone(20_000.0, 512));
    let mut processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512)
            .channels(2),
    )
    .unwrap();
    let mut output = vec![0.0; input.len()];

    let info = processor.process_frame(&input, &mut output).unwrap();

    assert!(!info.bypassed);
    assert_eq!(processor.inference_run_count(), 1);
}

#[test]
fn batched_stereo_matches_two_mono_processors() {
    let samples = 4096;
    let left = tone(440.0, samples);
    let right = tone(880.0, samples);
    let stereo_input = interleave(&left, &right);
    let mut stereo = SoundExProcessor::new(forced_gate_config(2, true)).unwrap();
    let mut left_mono = SoundExProcessor::new(forced_gate_config(1, true)).unwrap();
    let mut right_mono = SoundExProcessor::new(forced_gate_config(1, true)).unwrap();
    let mut stereo_output = vec![0.0; 1024];
    let mut left_output = vec![0.0; 512];
    let mut right_output = vec![0.0; 512];

    for hop in 0..samples / 512 {
        let mono_range = hop * 512..(hop + 1) * 512;
        let stereo_range = hop * 1024..(hop + 1) * 1024;
        stereo
            .process_frame(&stereo_input[stereo_range], &mut stereo_output)
            .unwrap();
        left_mono
            .process_frame(&left[mono_range.clone()], &mut left_output)
            .unwrap();
        right_mono
            .process_frame(&right[mono_range], &mut right_output)
            .unwrap();

        for sample in 0..512 {
            assert!((stereo_output[sample * 2] - left_output[sample]).abs() <= 1e-5);
            assert!((stereo_output[sample * 2 + 1] - right_output[sample]).abs() <= 1e-5);
        }
    }

    assert_eq!(stereo.inference_run_count(), 8);
    assert_eq!(left_mono.inference_run_count(), 8);
    assert_eq!(right_mono.inference_run_count(), 8);
}

#[test]
fn antiphase_stereo_remains_antiphase_during_enhancement() {
    let left = tone(440.0, 4096);
    let right: Vec<f32> = left.iter().map(|sample| -*sample).collect();
    let input = interleave(&left, &right);
    let mut processor = SoundExProcessor::new(forced_gate_config(2, true)).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    for frame in output.as_chunks::<2>().0 {
        assert!((frame[0] + frame[1]).abs() <= 1e-5);
    }
}

#[test]
fn asymmetric_transient_does_not_leak_between_channels() {
    let mut left = vec![0.0; 4096];
    left[1024] = 0.5;
    let right = vec![0.0; left.len()];
    let input = interleave(&left, &right);
    let mut processor = SoundExProcessor::new(forced_gate_config(2, true)).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    assert!(output
        .as_chunks::<2>()
        .0
        .iter()
        .any(|frame| frame[0].abs() > 1e-4));
    assert!(output
        .as_chunks::<2>()
        .0
        .iter()
        .all(|frame| frame[1].abs() <= 1e-7));
}

#[test]
fn streaming_api_exposes_causal_latency() {
    let input = tone(20_000.0, 1024);
    let mut processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
    let mut output = vec![0.0; processor.hop_size()];

    processor.process_frame(&input[..512], &mut output).unwrap();
    assert_eq!(output, vec![0.0; processor.hop_size()]);

    processor.process_frame(&input[512..], &mut output).unwrap();
    assert_eq!(output, input[..512]);
}
