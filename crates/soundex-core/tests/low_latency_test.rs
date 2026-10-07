//! Low-latency defaults, stream alignment, and legacy model rejection.
#![cfg(feature = "ort-backend")]

use soundex_core::{SoundExConfig, SoundExError, SoundExProcessor};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn default_stream_has_exactly_128_samples_of_delay() {
    for rate in [44100, 48000] {
        for channels in [1, 2] {
            let mut config = SoundExConfig::with_model(fixture("low-latency-identity.onnx"))
                .sample_rate(rate)
                .channels(channels);
            config.min_bandwidth_ratio = 0.0;
            let mut processor = SoundExProcessor::new(config).unwrap();
            assert_eq!(processor.fft_size(), 256);
            assert_eq!(processor.hop_size(), 128);
            assert_eq!(processor.latency_samples_per_channel(), 128);
            let mut input = vec![0.0; 128 * channels as usize];
            input[37 * channels as usize] = 0.25;
            let mut output = vec![0.0; input.len()];
            processor.process_frame(&input, &mut output).unwrap();
            assert!(output.iter().all(|&sample| sample == 0.0));
            processor
                .process_frame(&vec![0.0; input.len()], &mut output)
                .unwrap();
            assert_eq!(output, input);
        }
    }
}

#[test]
fn arbitrary_chunks_match_fixed_hop_aligned_output() {
    let mut config = SoundExConfig::with_model(fixture("low-latency-identity.onnx"));
    config.min_bandwidth_ratio = 1.0;
    let input: Vec<f32> = (0..1537).map(|i| 0.2 * (i as f32 * 0.15).sin()).collect();
    let expected = SoundExProcessor::new(config.clone())
        .unwrap()
        .process_buffer(&input)
        .unwrap();
    for chunk in [1, 31, 127, 128, 257] {
        let mut processor = SoundExProcessor::new(config.clone()).unwrap();
        let mut output = Vec::new();
        for part in input.chunks(chunk) {
            processor.process_chunk(part, &mut output).unwrap();
        }
        processor.finalize(&mut output).unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn legacy_models_need_explicit_matching_frame_configuration() {
    let result = SoundExProcessor::new(SoundExConfig::with_model(fixture("identity.onnx")));
    assert!(matches!(result, Err(SoundExError::ModelContract(_))));
    SoundExProcessor::new(
        SoundExConfig::with_model(fixture("identity.onnx"))
            .fft_size(1024)
            .hop_size(512),
    )
    .unwrap();
}
