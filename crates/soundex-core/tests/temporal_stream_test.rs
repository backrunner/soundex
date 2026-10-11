#![cfg(feature = "ort-backend")]

use ndarray::Array4;
use soundex_core::{inference::InferenceEngine, SoundExConfig, SoundExProcessor};
use std::path::PathBuf;

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn config() -> SoundExConfig {
    let mut config = SoundExConfig::with_model(path("temporal-probe.onnx")).channels(2);
    config.min_bandwidth_ratio = 1.0;
    config
}

fn audio() -> Vec<f32> {
    (0..128 * 40)
        .flat_map(|n| {
            let t = n as f32 / 44100.0;
            let left = if n < 128 * 10 {
                0.0
            } else {
                (t * 2500.0).sin() * 0.1
            };
            let right = if n >= 128 * 25 {
                0.0
            } else {
                (t * 3700.0).sin() * 0.13
            };
            [left, right]
        })
        .collect()
}

#[test]
fn engine_preserves_history_axis_batch_identity_and_single_frame_output() {
    let mut engine = InferenceEngine::load(&path("temporal-probe.onnx"), &config()).unwrap();
    assert_eq!(engine.context_frames(), 9);
    for batch in [2, 1, 2] {
        let input = Array4::from_shape_fn((batch, 2, 9, 129), |(b, c, t, f)| {
            b as f32 + c as f32 * 0.1 + t as f32 * 0.02 + f as f32 * 0.001
        });
        let result = engine.infer(&input).unwrap();
        assert_eq!(result.shape(), &[batch, 2, 1, 129]);
        for b in 0..batch {
            for c in 0..2 {
                for f in 0..129 {
                    let expected = 0.1 * input[[b, c, 0, f]] + 0.9 * input[[b, c, 8, f]];
                    assert!((result[[b, c, 0, f]] - expected).abs() < 1e-6);
                }
            }
        }
    }
    assert!(engine.infer(&Array4::zeros((1, 2, 1, 129))).is_err());
}

#[test]
fn history_warmup_returns_dry_audio_and_reset_restores_cold_start() {
    let mut processor = SoundExProcessor::new(config()).unwrap();
    let mut dry_config = config();
    dry_config.min_bandwidth_ratio = 0.0;
    let mut dry = SoundExProcessor::new(dry_config).unwrap();
    let hop = vec![0.1; 256];
    let mut output = vec![0.0; 256];
    let mut expected = output.clone();
    for _ in 0..8 {
        processor.process_frame(&hop, &mut output).unwrap();
        dry.process_frame(&hop, &mut expected).unwrap();
        assert_eq!(processor.inference_run_count(), 0);
        assert_eq!(output, expected);
    }
    processor.process_frame(&hop, &mut output).unwrap();
    assert_eq!(processor.inference_run_count(), 1);
    processor.reset();
    let before = processor.inference_run_count();
    let mut fresh = SoundExProcessor::new(config()).unwrap();
    for hop in audio().as_chunks::<256>().0 {
        processor.process_frame(hop, &mut output).unwrap();
        fresh.process_frame(hop, &mut expected).unwrap();
        assert_eq!(output, expected);
    }
    assert_eq!(
        processor.inference_run_count() - before,
        fresh.inference_run_count()
    );
    assert_eq!(processor.latency_samples_per_channel(), 128);
}

#[test]
fn temporal_stream_chunks_channel_switching_and_reload_match_fresh_processing() {
    let input = audio();
    let expected = SoundExProcessor::new(config())
        .unwrap()
        .process_buffer(&input)
        .unwrap();
    for chunk_size in [2, 62, 254, 1022] {
        let mut processor = SoundExProcessor::new(config()).unwrap();
        let mut output = Vec::new();
        for chunk in input.chunks(chunk_size) {
            processor.process_chunk(chunk, &mut output).unwrap();
        }
        processor.finalize(&mut output).unwrap();
        assert_eq!(output, expected);
        assert!(output.iter().all(|v| v.is_finite() && v.abs() < 1.0));
        // The silent physical channel stays silent as the active batch slot changes.
        assert!(output[..128 * 9 * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .all(|s| s[0] == 0.0));
        assert!(output[128 * 28 * 2..]
            .as_chunks::<2>()
            .0
            .iter()
            .all(|s| s[1] == 0.0));
        processor
            .reload_model(path("low-latency-identity.onnx"))
            .unwrap();
        let mut single = config();
        single.model_path = path("low-latency-identity.onnx");
        assert_eq!(
            processor.process_buffer(&input).unwrap(),
            SoundExProcessor::new(single)
                .unwrap()
                .process_buffer(&input)
                .unwrap()
        );
        processor.reload_model(path("temporal-probe.onnx")).unwrap();
        assert_eq!(processor.process_buffer(&input).unwrap(), expected);
    }
}
