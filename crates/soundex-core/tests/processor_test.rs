#![cfg(feature = "ort-backend")]

use std::path::PathBuf;

use soundex_core::{analyze_buffer, SoundExConfig, SoundExError, SoundExProcessor};

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
    let mut config = SoundExConfig::with_model(model_path()).channels(channels);
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
    let config = SoundExConfig::with_model(model_path());
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
    let config = SoundExConfig::with_model(model_path()).channels(2);
    let mut processor = SoundExProcessor::new(config).unwrap();

    let output = processor.process_buffer(&input).unwrap();

    assert_eq!(output, input);
}

#[test]
fn low_bandwidth_input_runs_inference_and_remains_finite() {
    let input = tone(440.0, 4096);
    let config = SoundExConfig::with_model(model_path());
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
    let mut processor =
        SoundExProcessor::new(SoundExConfig::with_model(model_path()).channels(2)).unwrap();
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

    for frame in output.chunks_exact(2) {
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

    assert!(output.chunks_exact(2).any(|frame| frame[0].abs() > 1e-4));
    assert!(output.chunks_exact(2).all(|frame| frame[1].abs() <= 1e-7));
}

#[test]
fn streaming_api_exposes_causal_latency() {
    let input = tone(20_000.0, 1024);
    let mut processor = SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
    let mut output = vec![0.0; processor.hop_size()];

    processor.process_frame(&input[..512], &mut output).unwrap();
    assert_eq!(output, vec![0.0; processor.hop_size()]);

    processor.process_frame(&input[512..], &mut output).unwrap();
    assert_eq!(output, input[..512]);
}

#[test]
fn analysis_distinguishes_low_and_full_bandwidth_tones() {
    let config = SoundExConfig::default();
    let low = analyze_buffer(&config, &tone(440.0, 4096)).unwrap();
    let full = analyze_buffer(&config, &tone(20_000.0, 4096)).unwrap();

    assert!(low.needs_enhancement());
    assert!(!full.needs_enhancement());
    assert!(low.detected_bandwidth_hz < full.detected_bandwidth_hz);
}

#[test]
fn invalid_config_is_rejected_before_model_loading() {
    let mut config = SoundExConfig::with_model("missing.onnx");
    config.channels = 3;
    assert!(matches!(
        SoundExProcessor::new(config),
        Err(SoundExError::InvalidInput(_))
    ));
}

#[test]
fn missing_model_is_an_error() {
    let error = SoundExProcessor::new(SoundExConfig::with_model("missing.onnx"))
        .err()
        .expect("missing model should fail");
    assert!(matches!(error, SoundExError::ModelNotFound(_)));
}

fn stream_in_chunks(input: &[f32], chunk_size: usize) -> Vec<f32> {
    let mut processor = SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
    let mut output = Vec::new();
    let mut consumed = 0;
    for chunk in input.chunks(chunk_size) {
        let progress = processor.process_chunk(chunk, &mut output).unwrap();
        assert_eq!(progress.consumed, chunk.len());
        consumed += progress.consumed;
    }
    let progress = processor.finalize(&mut output).unwrap();
    assert_eq!(progress.consumed, 0);
    assert_eq!(consumed, input.len());
    assert_eq!(output.len(), input.len());
    output
}

#[test]
fn arbitrary_chunking_and_finalize_match_offline_processing() {
    let input = tone(440.0, 3333);
    let mut offline_processor =
        SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
    let expected = offline_processor.process_buffer(&input).unwrap();

    for chunk_size in [1, 127, 512, 997] {
        let actual = stream_in_chunks(&input, chunk_size);
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual_sample, &expected_sample)) in actual.iter().zip(&expected).enumerate()
        {
            assert!(
                (actual_sample - expected_sample).abs() <= 1e-6,
                "chunk {chunk_size} differs at sample {index}: {actual_sample} vs {expected_sample}"
            );
        }
    }
}

#[test]
fn short_inputs_use_the_same_causal_tail_framing() {
    for length in [1, 511, 512, 2047] {
        let input = tone(440.0, length);
        let streamed = stream_in_chunks(&input, 17);
        let mut offline_processor =
            SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
        let offline = offline_processor.process_buffer(&input).unwrap();

        assert_eq!(streamed.len(), length);
        assert_eq!(streamed, offline);

        let analysis = analyze_buffer(&SoundExConfig::default(), &input).unwrap();
        assert_eq!(analysis.frames_analyzed, length.div_ceil(512) + 1);
    }
}

#[test]
fn finalize_is_single_use_and_reset_starts_a_new_stream() {
    let input = tone(440.0, 37);
    let mut processor = SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
    let mut output = Vec::new();
    processor.process_chunk(&input, &mut output).unwrap();
    processor.finalize(&mut output).unwrap();

    assert!(matches!(
        processor.finalize(&mut output),
        Err(SoundExError::StreamFinalized)
    ));
    assert!(matches!(
        processor.process_chunk(&input, &mut output),
        Err(SoundExError::StreamFinalized)
    ));

    processor.reset();
    let mut second_output = Vec::new();
    processor.process_chunk(&input, &mut second_output).unwrap();
    processor.finalize(&mut second_output).unwrap();
    assert_eq!(second_output.len(), input.len());
}

#[test]
fn fixed_hop_finalize_emits_exact_reported_tail() {
    let mut processor = SoundExProcessor::new(SoundExConfig::with_model(model_path())).unwrap();
    assert_eq!(processor.latency_samples_per_channel(), 512);
    let input = tone(440.0, processor.hop_size());
    let mut frame_output = vec![0.0; processor.hop_size()];
    processor.process_frame(&input, &mut frame_output).unwrap();
    let mut tail = Vec::new();

    let progress = processor.finalize(&mut tail).unwrap();

    assert_eq!(progress.produced, processor.latency_samples_per_channel());
    assert_eq!(tail.len(), processor.latency_samples_per_channel());
}

#[test]
fn stereo_chunking_counts_interleaved_latency_correctly_before_reset() {
    let left = tone(440.0, 1703);
    let right = tone(880.0, 1703);
    let input: Vec<f32> = left
        .iter()
        .zip(&right)
        .flat_map(|(&left, &right)| [left, right])
        .collect();
    let config = SoundExConfig::with_model(model_path()).channels(2);
    let mut offline_processor = SoundExProcessor::new(config.clone()).unwrap();
    let expected = offline_processor.process_buffer(&input).unwrap();
    let mut streaming_processor = SoundExProcessor::new(config).unwrap();
    let mut actual = Vec::new();
    for chunk in input.chunks(74) {
        streaming_processor
            .process_chunk(chunk, &mut actual)
            .unwrap();
    }
    streaming_processor.finalize(&mut actual).unwrap();

    assert_eq!(actual.len(), input.len());
    for (&actual_sample, &expected_sample) in actual.iter().zip(&expected) {
        assert!((actual_sample - expected_sample).abs() <= 1e-6);
    }
}
