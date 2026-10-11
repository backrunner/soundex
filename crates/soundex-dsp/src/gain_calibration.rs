//! Small causal input-conditioned correction to the legacy edge target.
//! Coefficients are fitted offline on aligned, reviewed training audio only.

use crate::loudness::rms;

pub const GAIN_FEATURES: usize = 10;

/// Immutable linear predictor. No allocation or history beyond the existing gain smoother.
#[derive(Debug, Clone)]
pub struct GainCalibration {
    coefficients: [f32; GAIN_FEATURES],
}

impl GainCalibration {
    pub fn new(coefficients: [f32; GAIN_FEATURES]) -> Result<Self, &'static str> {
        if coefficients
            .iter()
            .any(|x| !x.is_finite() || x.abs() > 64.0)
        {
            return Err("gain coefficients must be finite and bounded by 64");
        }
        Ok(Self { coefficients })
    }

    pub fn coefficients(&self) -> &[f32; GAIN_FEATURES] {
        &self.coefficients
    }

    /// Correct the -6 dB legacy target by at most +/-6 dB. Final gain bounds and
    /// smoothing remain in LoudnessMatcher, rather than clipping the waveform here.
    pub fn target_ratio_db(&self, features: &[f32; GAIN_FEATURES]) -> f32 {
        let correction: f32 = self
            .coefficients
            .iter()
            .zip(features)
            .map(|(a, b)| a * b)
            .sum();
        -6.0 + correction.clamp(-6.0, 6.0)
    }
}

/// Bias, normalized cutoff, rate, six retained-band/edge RMS ratios, and surviving
/// high-band/edge RMS ratio. Geometry and rounding are supplied by the runtime.
/// Inputs are finite nonnegative FFT magnitudes, cutoff is in 1..len.
pub fn gain_features(
    magnitude: &[f32],
    cutoff: usize,
    reference: usize,
    rate: u32,
) -> [f32; GAIN_FEATURES] {
    assert!(cutoff > 0 && cutoff < magnitude.len());
    assert!(reference > 0 && reference <= cutoff);
    let edge = rms(&magnitude[cutoff - reference..cutoff]).max(1e-10);
    let relative =
        |values: &[f32]| (20.0 * (rms(values).max(1e-10) / edge).log10()).clamp(-60.0, 60.0) / 20.0;
    let mut result = [0.0; GAIN_FEATURES];
    result[0] = 1.0;
    result[1] = cutoff as f32 / (magnitude.len() - 1) as f32;
    result[2] = rate as f32 / 48000.0;
    for band in 0..6 {
        let start = band * cutoff / 6;
        let end = ((band + 1) * cutoff / 6).max(start + 1).min(cutoff);
        result[3 + band] = relative(&magnitude[start..end]);
    }
    result[9] = relative(&magnitude[cutoff..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_level_invariant_bounded_and_zero_is_legacy() {
        let a: Vec<f32> = (0..129).map(|i| 1.0 / (i + 1) as f32).collect();
        let b: Vec<f32> = a.iter().map(|x| x * 0.1).collect();
        let x = gain_features(&a, 80, 6, 48000);
        let y = gain_features(&b, 80, 6, 48000);
        assert!(x.iter().zip(y).all(|(x, y)| (x - y).abs() < 1e-6));
        assert_eq!(
            GainCalibration::new([0.0; 10]).unwrap().target_ratio_db(&x),
            -6.0
        );
        for bias in [-64.0, 64.0] {
            let mut coefficients = [0.0; 10];
            coefficients[0] = bias;
            let ratio = GainCalibration::new(coefficients)
                .unwrap()
                .target_ratio_db(&x);
            assert_eq!(ratio, if bias < 0.0 { -12.0 } else { 0.0 });
        }
        for cutoff in [1, 128] {
            assert!(gain_features(&[0.0; 129], cutoff, 1, 44100)
                .iter()
                .all(|x| x.is_finite()));
        }
        assert!(GainCalibration::new([f32::NAN; 10]).is_err());
        assert!(GainCalibration::new([65.0; 10]).is_err());
    }
}
