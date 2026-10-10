use soundex_core::{EnhancementMode, SoundExConfig, SoundExProcessor};

fn configuration(mode: EnhancementMode, rate: u32, channels: u16) -> SoundExConfig {
    let mut config = SoundExConfig::with_model(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/low-latency-identity.onnx"),
    )
    .sample_rate(rate)
    .channels(channels)
    .enhancement_mode(mode);
    config.min_bandwidth_ratio = 1.0;
    config
}

fn signal(frames: usize, channels: usize, rate: u32) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let t = i as f32 / rate as f32;
            let active = i < frames / 2;
            let value = if active {
                (1..40)
                    .map(|k| {
                        0.005 * (std::f32::consts::TAU * (200.0 * k as f32) * t + k as f32).sin()
                    })
                    .sum()
            } else {
                0.0
            };
            (0..channels).map(move |channel| if channel == 0 { value } else { 0.0 })
        })
        .collect()
}

#[test]
fn spectral_path_requires_no_model_or_backend_and_resets_deterministically() {
    let config = SoundExConfig::with_model("does-not-exist.onnx")
        .enhancement_mode(EnhancementMode::Spectral);
    let mut processor = SoundExProcessor::new(config).unwrap();
    let input = signal(5003, 1, 44100);
    let first = processor.process_buffer(&input).unwrap();
    let second = processor.process_buffer(&input).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), input.len());
    assert_eq!(processor.inference_run_count(), 0);
    assert_eq!(processor.latency_samples_per_channel(), 128);
    assert!(first.iter().all(|v| v.is_finite() && v.abs() <= 0.95));
}

#[test]
fn spectral_stream_keeps_silent_channel_silent_and_chunks_aligned() {
    for rate in [44100, 48000] {
        let config = configuration(EnhancementMode::Spectral, rate, 2);
        let input = signal(5003, 2, rate);
        let expected = SoundExProcessor::new(config.clone())
            .unwrap()
            .process_buffer(&input)
            .unwrap();
        for chunk_frames in [1, 31, 128, 997] {
            let mut processor = SoundExProcessor::new(config.clone()).unwrap();
            let mut output = Vec::new();
            for chunk in input.chunks(chunk_frames * 2) {
                processor.process_chunk(chunk, &mut output).unwrap();
            }
            processor.finalize(&mut output).unwrap();
            assert_eq!(output, expected);
            assert!(output.as_chunks::<2>().0.iter().all(|v| v[1] == 0.0));
            assert!(output[4000 * 2..].iter().all(|v| v.abs() < 1e-8));
        }
    }
}

#[test]
#[cfg(feature = "ort-backend")]
fn hybrid_chunks_remain_exact_and_model_is_invoked() {
    for rate in [44100, 48000] {
        let config = configuration(EnhancementMode::Hybrid, rate, 2);
        let input = signal(5003, 2, rate);
        let expected = SoundExProcessor::new(config.clone())
            .unwrap()
            .process_buffer(&input)
            .unwrap();
        let mut processor = SoundExProcessor::new(config).unwrap();
        let mut output = Vec::new();
        for chunk in input.chunks(31 * 2) {
            processor.process_chunk(chunk, &mut output).unwrap();
        }
        processor.finalize(&mut output).unwrap();
        assert_eq!(expected, output);
        assert!(processor.inference_run_count() > 0);
        assert!(output.as_chunks::<2>().0.iter().all(|v| v[1] == 0.0));
        assert!(output[4000 * 2..].iter().all(|v| v.abs() < 1e-8));
    }
}
