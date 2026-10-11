//! Preserving model levels must retain gain-calibrated identity output and reset.
#![cfg(feature = "ort-backend")]

use soundex_core::{HighBandGain, SoundExConfig, SoundExProcessor};

#[test]
fn phase_policies_preserve_identity_reset_silence_and_channel_isolation() {
    use soundex_core::{EnhancementMode, PhasePolicy};
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    for policy in [PhasePolicy::Observed, PhasePolicy::SurvivingPower] {
        assert_eq!(policy.as_str().parse::<PhasePolicy>().unwrap(), policy);
        for rate in [44100, 48000] {
            for channels in [1, 2] {
                let mut config = SoundExConfig::with_model(&model)
                    .sample_rate(rate)
                    .channels(channels)
                    .high_band_gain(HighBandGain::Model);
                config.phase_policy = policy;
                config.min_bandwidth_ratio = 1.0;
                let input: Vec<f32> = (0..8192)
                    .flat_map(|i| {
                        let sample = if (3000..5000).contains(&i) {
                            0.0
                        } else {
                            0.2 * (i as f32 * 0.13).sin()
                        };
                        (0..channels).map(move |c| if c == 0 { sample } else { 0.0 })
                    })
                    .collect();
                let mut processor = SoundExProcessor::new(config.clone()).unwrap();
                let output = processor.process_buffer(&input).unwrap();
                assert!(output.iter().zip(&input).all(|(x, y)| (x - y).abs() < 2e-6));
                if channels == 2 {
                    assert!(output.iter().skip(1).step_by(2).all(|x| *x == 0.0));
                }
                processor.reset();
                assert_eq!(output, processor.process_buffer(&input).unwrap());
                processor.reset();
                assert!(processor
                    .process_buffer(&vec![0.0; input.len()])
                    .unwrap()
                    .iter()
                    .all(|x| *x == 0.0));
                for mode in [EnhancementMode::Spectral, EnhancementMode::Hybrid] {
                    config.enhancement_mode = mode;
                    assert!(SoundExProcessor::new(config.clone()).is_err());
                }
            }
        }
    }
    assert_eq!(SoundExConfig::default().phase_policy, PhasePolicy::Model);
    assert!("typo".parse::<PhasePolicy>().is_err());
}

#[test]
fn model_level_identity_preserves_signal_and_reset_for_mono_and_stereo() {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    for rate in [44100, 48000] {
        for channels in [1u16, 2] {
            let mut config = SoundExConfig::with_model(&model)
                .sample_rate(rate)
                .channels(channels)
                .high_band_gain(HighBandGain::Model);
            config.min_bandwidth_ratio = 1.0;
            let input: Vec<f32> = (0..4096)
                .flat_map(|frame| {
                    let sample = 0.2
                        * (frame as f32 * 2.0 * std::f32::consts::PI * 1000.0 / rate as f32).sin();
                    (0..channels).map(move |channel| if channel == 0 { sample } else { 0.0 })
                })
                .collect();
            let mut processor = SoundExProcessor::new(config).unwrap();
            let result = processor.process_buffer(&input).unwrap();
            let error = result
                .iter()
                .zip(&input)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 2e-6, "{rate}/{channels} identity error {error}");
            processor.reset();
            assert_eq!(result, processor.process_buffer(&input).unwrap());
        }
    }
}

#[test]
fn zero_calibration_is_bit_identical_and_bounded_calibration_resets() {
    use soundex_dsp::gain_calibration::GainCalibration;
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    for rate in [44100, 48000] {
        for channels in [1, 2] {
            let config = SoundExConfig::with_model(&model)
                .sample_rate(rate)
                .channels(channels);
            let input: Vec<f32> = (0..8192)
                .flat_map(|i| {
                    let sample = if (3000..5000).contains(&i) {
                        0.0
                    } else {
                        0.2 * (i as f32 * 0.13).sin()
                    };
                    (0..channels).map(move |c| if c == 0 { sample } else { 0.0 })
                })
                .collect();
            let baseline = SoundExProcessor::new(config.clone())
                .unwrap()
                .process_buffer(&input)
                .unwrap();
            let mut zero = config.clone();
            zero.gain_calibration = Some(GainCalibration::new([0.0; 10]).unwrap());
            assert_eq!(
                baseline,
                SoundExProcessor::new(zero)
                    .unwrap()
                    .process_buffer(&input)
                    .unwrap()
            );
            let mut calibrated = config;
            let mut coefficients = [0.0; 10];
            coefficients[0] = 6.0;
            calibrated.gain_calibration = Some(GainCalibration::new(coefficients).unwrap());
            let mut processor = SoundExProcessor::new(calibrated.clone()).unwrap();
            let output = processor.process_buffer(&input).unwrap();
            assert!(output.iter().all(|x| x.is_finite() && x.abs() <= 0.95));
            if channels == 2 {
                assert!(output.iter().skip(1).step_by(2).all(|x| *x == 0.0));
            }
            processor.reset();
            assert_eq!(output, processor.process_buffer(&input).unwrap());
            calibrated.high_band_gain = HighBandGain::Model;
            assert!(SoundExProcessor::new(calibrated).is_err());
        }
    }
}
