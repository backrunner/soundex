//! Native envelope predictor parity and Envelope-mode stream behavior.

use std::path::PathBuf;

use soundex_core::envelope::EnvelopeModel;
use soundex_core::{EnhancementMode, SoundExConfig, SoundExProcessor};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn config(channels: u16, rate: u32) -> SoundExConfig {
    SoundExConfig::with_model(fixture("envelope-probe.sxenv"))
        .sample_rate(rate)
        .channels(channels)
        .enhancement_mode(EnhancementMode::Envelope)
}

/// Band-limited harmonic signal on channel 0 (as a 64 kbps codec would leave it).
fn lowpassed(frames: usize, channels: usize, rate: u32) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let t = i as f32 / rate as f32;
            let value: f32 = (1..60)
                .map(|k| 200.0 * k as f32)
                .filter(|&hz| hz < 11_000.0)
                .map(|hz| 0.004 * (std::f32::consts::TAU * hz * t + hz).sin())
                .sum();
            (0..channels).map(move |c| if c == 0 { value } else { 0.0 })
        })
        .collect()
}

#[test]
fn native_gru_matches_pytorch_reference() {
    let model = EnvelopeModel::load(&fixture("envelope-probe.sxenv"), 256, 128).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("envelope-probe.json")).unwrap())
            .unwrap();
    let as_vec = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let inputs = reference["inputs"].as_array().unwrap();
    let outputs = reference["outputs"].as_array().unwrap();
    let mut state = vec![0.0; model.state_len()];
    let mut scratch = model.scratch();
    let mut out = vec![0.0; model.output_len()];
    let mut worst = 0.0f32;
    for (x, y) in inputs.iter().zip(outputs) {
        model.step(&as_vec(x), &mut state, &mut scratch, &mut out);
        for (a, b) in out.iter().zip(as_vec(y)) {
            worst = worst.max((a - b).abs());
        }
    }
    assert!(worst < 2e-5, "max |native - torch| = {worst}");
    for (a, b) in state.iter().zip(as_vec(&reference["final_state"])) {
        assert!((a - b).abs() < 2e-5);
    }
    assert_eq!(model.warmup_frames(), 4);
}

#[test]
fn tampered_or_mismatched_artifacts_are_rejected() {
    let mut bytes = std::fs::read(fixture("envelope-probe.sxenv")).unwrap();
    assert!(EnvelopeModel::from_bytes(&bytes, 512, 256).is_err());
    let last = bytes.len() - 1;
    bytes[last] ^= 0x40;
    let error = EnvelopeModel::from_bytes(&bytes, 256, 128).unwrap_err();
    assert!(error.to_string().contains("SHA-256"), "{error}");
    let missing = config(1, 48_000);
    let mut missing = missing.clone();
    missing.model_path = "does-not-exist.sxenv".into();
    assert!(SoundExProcessor::new(missing).is_err());
}

#[test]
fn warmup_is_dry_then_envelope_adds_high_band() {
    let rate = 48_000;
    let input = lowpassed(rate as usize, 1, rate);
    let mut envelope = SoundExProcessor::new(config(1, rate)).unwrap();
    let mut bypass_config = config(1, rate);
    bypass_config.min_bandwidth_ratio = 0.0; // Detector never requests enhancement.
    let mut bypass = SoundExProcessor::new(bypass_config).unwrap();
    let wet = envelope.process_buffer(&input).unwrap();
    let dry = bypass.process_buffer(&input).unwrap();
    assert_eq!(wet.len(), input.len());
    assert!(wet.iter().all(|v| v.is_finite()));
    // Warmup (4 hops) plus analysis latency: identical to the dry path.
    assert_eq!(wet[..4 * 128 - 128], dry[..4 * 128 - 128]);
    let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
    let difference: Vec<f32> = wet.iter().zip(&dry).map(|(a, b)| a - b).collect();
    assert!(
        energy(&difference[rate as usize / 2..]) > 0.0,
        "fill must add energy"
    );
    assert_eq!(
        envelope.inference_run_count(),
        0,
        "no ONNX engine in envelope mode"
    );
}

#[test]
fn silence_isolation_and_reset_are_deterministic() {
    let rate = 44_100;
    let mut processor = SoundExProcessor::new(config(2, rate)).unwrap();
    let silent = vec![0.0; 2 * 8192];
    assert!(processor
        .process_buffer(&silent)
        .unwrap()
        .iter()
        .all(|&v| v == 0.0));
    processor.reset();
    let input = lowpassed(20_000, 2, rate);
    let first = processor.process_buffer(&input).unwrap();
    assert!(
        first.iter().skip(1).step_by(2).all(|&v| v == 0.0),
        "silent right channel"
    );
    processor.reset();
    let second = processor.process_buffer(&input).unwrap();
    assert_eq!(first, second);
}
