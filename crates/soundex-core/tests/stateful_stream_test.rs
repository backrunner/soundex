#![cfg(feature = "ort-backend")]

use ndarray::Array4;
use soundex_core::{inference::InferenceEngine, HighBandGain, SoundExConfig, SoundExProcessor};
use std::path::PathBuf;

fn config(channels: u16) -> SoundExConfig {
    let mut c = SoundExConfig::with_model(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/stateful-probe.onnx"),
    )
    .channels(channels);
    c.high_band_gain = HighBandGain::Model;
    c.min_bandwidth_ratio = 1.0;
    c
}

#[test]
fn explicit_state_tracks_physical_channels_and_reset() {
    let c = config(2);
    let mut engine = InferenceEngine::load(&c.model_path, &c).unwrap();
    assert!(engine.is_stateful());
    assert_eq!(engine.warmup_frames(), 64);
    let mut expected = [0.0_f32; 2];
    for hop in 0..100 {
        if hop == 70 {
            engine.reset_state();
            expected.fill(0.0);
        }
        let input = Array4::from_shape_fn((2, 2, 1, 129), |(b, _, _, _)| {
            if (hop / 20) % 2 == b {
                -80.0
            } else {
                -20.0
            }
        });
        let result = engine.infer(&input).unwrap();
        for b in 0..2 {
            assert!((result[[b, 0, 0, 10]] - input[[b, 0, 0, 10]] - expected[b]).abs() < 1e-5);
            assert_eq!(result[[b, 1, 0, 10]], input[[b, 1, 0, 10]]);
            expected[b] = expected[b] * 0.9 + input[[b, 0, 0, 10]] * 0.01;
        }
    }
    let mut invalid = c.clone();
    invalid.high_band_gain = HighBandGain::MatchEdge;
    assert!(InferenceEngine::load(&c.model_path, &invalid).is_err());
}

#[test]
fn state_advances_during_silence_and_bypass_and_warmup_is_dry() {
    let mut p = SoundExProcessor::new(config(2)).unwrap();
    let mut dry_config = config(2);
    dry_config.min_bandwidth_ratio = 0.0;
    let mut dry = SoundExProcessor::new(dry_config).unwrap();
    let mut output = [0.0; 256];
    let mut reference = [0.0; 256];
    for hop in 0..70 {
        let input = std::array::from_fn::<_, 256, _>(|n| {
            if hop < 8 {
                0.0
            } else {
                0.1 * ((hop * 128 + n / 2) as f32 * 0.11).sin()
            }
        });
        let info = p.process_frame(&input, &mut output).unwrap();
        dry.process_frame(&input, &mut reference).unwrap();
        assert_eq!(p.inference_run_count(), hop as u64 + 1);
        assert_eq!(dry.inference_run_count(), hop as u64 + 1);
        if hop < 64 {
            assert!(info.bypassed);
            assert_eq!(output, reference);
        } else {
            assert!(!info.bypassed);
        }
    }
    assert_eq!(p.latency_samples_per_channel(), 128);
}

#[test]
fn channel_switching_chunks_reset_and_reload_preserve_timeline() {
    let input: Vec<f32> = (0..128 * 160 + 37)
        .flat_map(|n| {
            [
                if n < 128 * 75 {
                    0.0
                } else {
                    0.1 * (n as f32 * 0.11).sin()
                },
                if n > 128 * 110 {
                    0.0
                } else {
                    0.1 * (n as f32 * 0.17).sin()
                },
            ]
        })
        .collect();
    let c = config(2);
    let expected = SoundExProcessor::new(c.clone())
        .unwrap()
        .process_buffer(&input)
        .unwrap();
    for size in [2, 62, 254, 1022] {
        let mut p = SoundExProcessor::new(c.clone()).unwrap();
        let mut output = Vec::new();
        for chunk in input.chunks(size) {
            p.process_chunk(chunk, &mut output).unwrap();
        }
        p.finalize(&mut output).unwrap();
        assert_eq!(output, expected);
        for reset in [true, false] {
            if reset {
                p.reset();
            } else {
                p.reload_model(&c.model_path).unwrap();
            }
            assert_eq!(p.process_buffer(&input).unwrap(), expected);
        }
    }
    // Physical-channel memory must also agree with two independent mono processors.
    for channel in 0..2 {
        let mono: Vec<_> = input
            .as_chunks::<2>()
            .0
            .iter()
            .map(|f| f[channel])
            .collect();
        let actual = SoundExProcessor::new(config(1))
            .unwrap()
            .process_buffer(&mono)
            .unwrap();
        // Shared stereo gate is intentionally different at silent/active transitions;
        // compare the stable both-active region, after the gate attack has settled.
        for n in 128 * 85..128 * 105 {
            assert!((actual[n] - expected[n * 2 + channel]).abs() < 1e-6);
        }
    }
    assert!(expected[..128 * 70 * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .all(|f| f[0] == 0.0));
    assert!(expected[128 * 120 * 2..]
        .as_chunks::<2>()
        .0
        .iter()
        .all(|f| f[1] == 0.0));
}
