//! Smooth protection must preserve the streaming and channel contracts.
#![cfg(feature = "ort-backend")]

use soundex_core::{HighBandGain, LimiterMode, SoundExConfig, SoundExProcessor};
use soundex_dsp::limiter::Limiter;

#[test]
fn smooth_identity_matches_one_limit_across_chunks_resets_and_channels() {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    assert_eq!(SoundExConfig::default().limiter_mode, LimiterMode::Legacy);
    let limiter = Limiter::with_mode(0.95, 10.0, LimiterMode::SmoothKnee);
    for rate in [44100, 48000] {
        for channels in [1, 2] {
            let mut config = SoundExConfig::with_model(&model)
                .sample_rate(rate)
                .channels(channels)
                .high_band_gain(HighBandGain::Model);
            config.limiter_mode = LimiterMode::SmoothKnee;
            let input: Vec<f32> = (0..12345)
                .flat_map(|n| {
                    let x = if (4000..6000).contains(&n) {
                        0.0
                    } else {
                        1.4 * (n as f32 * std::f32::consts::TAU * 1700.0 / rate as f32).sin()
                    };
                    (0..channels).map(move |c| if c == 0 { x } else { -x })
                })
                .collect();
            let mut processor = SoundExProcessor::new(config).unwrap();
            let expected = processor.process_buffer(&input).unwrap();
            assert!(expected.iter().zip(&input).all(|(y, x)| y.is_finite()
                && y.abs() <= 0.95
                && (*y - limiter.process_sample(*x)).abs() < 2e-6));
            if channels == 2 {
                assert!(expected
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|s| (s[0] + s[1]).abs() < 2e-6));
            }
            processor.reset();
            let mut actual = Vec::new();
            let mut offset = 0;
            for n in [1, 17, 127, 128, 257, 511].into_iter().cycle() {
                let end = (offset + n * channels as usize).min(input.len());
                processor
                    .process_chunk(&input[offset..end], &mut actual)
                    .unwrap();
                offset = end;
                if offset == input.len() {
                    break;
                }
            }
            processor.finalize(&mut actual).unwrap();
            assert_eq!(actual, expected);
            processor.reset();
            let silent = vec![0.0; input.len()];
            assert_eq!(processor.process_buffer(&silent).unwrap(), silent);
        }
    }
}
