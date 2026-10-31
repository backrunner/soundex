//! Allocating, offline-only shadow synthesis. No production output is modified.

use ndarray::Array4;
use soundex_dsp::stft::{SpectralFrame, StftSynthesizer};

use crate::channel::ChannelProcessor;

pub(super) struct StageTrace {
    synths: Vec<[StftSynthesizer; 4]>,
    pub hops: usize,
}

impl StageTrace {
    pub fn new(channels: usize) -> Self {
        Self {
            synths: (0..channels)
                .map(|_| std::array::from_fn(|_| StftSynthesizer::new(256, 128)))
                .collect(),
            hops: 0,
        }
    }

    pub fn observe(
        &mut self,
        channels: &[ChannelProcessor],
        prediction: &Array4<f32>,
    ) -> serde_json::Value {
        let mut records = Vec::with_capacity(channels.len());
        for (c, channel) in channels.iter().enumerate() {
            let active = channel.needs_enhancement();
            let observed = channel.observed_spectrum();
            let (fused, magnitude_weights, phase_weights) = channel.synthesis_trace();
            let raw = SpectralFrame {
                log_magnitude: (0..129)
                    .map(|k| prediction[[c, 0, 0, k]].clamp(-200.0, 100.0))
                    .collect(),
                phase: (0..129).map(|k| prediction[[c, 1, 0, k]]).collect(),
            };
            let activity = if active { &raw } else { observed };
            let magnitude = SpectralFrame {
                log_magnitude: fused.log_magnitude.clone(),
                phase: activity.phase.clone(),
            };
            let frames = [&raw, activity, &magnitude, fused];
            let waves: Vec<_> = self.synths[c]
                .iter_mut()
                .zip(frames)
                .map(|(synth, frame)| synth.synthesize(frame))
                .collect();
            let effective = |weights: &[f32]| {
                if active {
                    weights.to_vec()
                } else {
                    vec![0.0; 129]
                }
            };
            records.push(serde_json::json!({
                "active": active,
                "magnitude_weights": effective(magnitude_weights),
                "phase_weights": effective(phase_weights),
                "fused_db": fused.log_magnitude,
                "fused_phase": fused.phase,
                "raw_ola": waves[0],
                "activity_ola": waves[1],
                "magnitude_ola": waves[2],
                "phase_ola": waves[3],
                "wet_limited": channel.wet(),
            }));
        }
        self.hops += 1;
        serde_json::json!(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HighBandGain, SoundExConfig};

    #[test]
    fn shadow_fused_synthesis_matches_native_unlimited_signal_including_bypass() {
        let config = SoundExConfig {
            fft_size: 256,
            hop_size: 128,
            channels: 1,
            min_bandwidth_ratio: 1.0,
            high_band_gain: HighBandGain::Model,
            ..SoundExConfig::default()
        };
        let mut channel = ChannelProcessor::new(&config, 1);
        let mut trace = StageTrace::new(1);
        let mut prediction = Array4::zeros((1, 2, 1, 129));
        for hop in 0..20 {
            let pcm: Vec<_> = (0..128)
                .map(|i| 0.03 * ((hop * 128 + i) as f32 * 0.13).sin())
                .collect();
            channel.analyze_hop(&pcm, &config).unwrap();
            for k in 0..129 {
                prediction[[0, 0, 0, k]] = channel.observed_spectrum().log_magnitude[k];
                prediction[[0, 1, 0, k]] =
                    channel.observed_spectrum().phase[k] + if k > 32 { 0.03 } else { 0.0 };
            }
            if hop % 4 == 0 {
                channel.suppress_enhancement();
            }
            channel
                .finish_hop(Some(prediction.slice(ndarray::s![0, .., .., ..])), &config)
                .unwrap();
            let record = trace.observe(std::slice::from_ref(&channel), &prediction);
            let shadow: Vec<f32> = serde_json::from_value(record[0]["phase_ola"].clone()).unwrap();
            assert_eq!(shadow, channel.wet());
            if hop % 4 == 0 {
                assert!(record[0]["phase_weights"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|v| v.as_f64() == Some(0.0)));
            }
        }
        assert_eq!(trace.hops, 20);
    }
}
