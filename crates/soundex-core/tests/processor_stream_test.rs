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
    let mut processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
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
    let mut offline_processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
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
        let mut offline_processor = SoundExProcessor::new(
            SoundExConfig::with_model(model_path())
                .fft_size(1024)
                .hop_size(512),
        )
        .unwrap();
        let offline = offline_processor.process_buffer(&input).unwrap();

        assert_eq!(streamed.len(), length);
        assert_eq!(streamed, offline);

        let analysis = analyze_buffer(
            &SoundExConfig::default().fft_size(1024).hop_size(512),
            &input,
        )
        .unwrap();
        assert_eq!(analysis.frames_analyzed, length.div_ceil(512) + 1);
    }
}

#[test]
fn finalize_is_single_use_and_reset_starts_a_new_stream() {
    let input = tone(440.0, 37);
    let mut processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
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
    let mut processor = SoundExProcessor::new(
        SoundExConfig::with_model(model_path())
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
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
    let config = SoundExConfig::with_model(model_path())
        .fft_size(1024)
        .hop_size(512)
        .channels(2);
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
