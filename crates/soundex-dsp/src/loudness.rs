//! Loudness matching between original low-band and generated high-band.

/// Matches the loudness of generated high-frequency content to the original low-band.
pub struct LoudnessMatcher {
    /// Target ratio in dB: high_band_rms relative to low_band_rms.
    target_ratio_db: f32,
    /// Smoothing coefficient for gain changes.
    smoothing_alpha: f32,
    /// Current smoothed gain value.
    current_gain: f32,
}

impl LoudnessMatcher {
    /// Create a new loudness matcher.
    ///
    /// # Arguments
    /// * `target_ratio_db` - Desired high/low band energy ratio in dB (e.g., -6.0)
    /// * `smoothing_alpha` - Gain smoothing factor (0..1)
    pub fn new(target_ratio_db: f32, smoothing_alpha: f32) -> Self {
        Self {
            target_ratio_db,
            smoothing_alpha,
            current_gain: 1.0,
        }
    }

    /// Compute the gain to apply to the high-band signal.
    ///
    /// Returns a linear gain factor clamped to [0.25, 2.0] (-12dB..+6dB).
    pub fn compute_gain(&mut self, low_band_rms: f32, high_band_rms: f32) -> f32 {
        if low_band_rms < 1e-10 {
            self.current_gain = 0.0;
            return 0.0;
        }
        if high_band_rms < 1e-10 {
            return 1.0;
        }

        let target_rms = low_band_rms * 10.0f32.powf(self.target_ratio_db / 20.0);
        let gain = target_rms / high_band_rms;
        let gain = gain.clamp(0.25, 2.0);

        self.current_gain =
            self.smoothing_alpha * gain + (1.0 - self.smoothing_alpha) * self.current_gain;

        self.current_gain
    }

    /// Get the current smoothed gain without updating.
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

/// Compute RMS of a signal slice.
pub fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|s| s * s).sum();
    (sum_sq / signal.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_sine() {
        // RMS of a full-scale sine is 1/sqrt(2) ≈ 0.707
        let signal: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 1000.0).sin())
            .collect();
        let r = rms(&signal);
        assert!((r - 0.707).abs() < 0.01, "RMS of sine: {r}");
    }

    #[test]
    fn test_rms_empty() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn test_loudness_gain_reduction() {
        let mut matcher = LoudnessMatcher::new(-6.0, 1.0); // No smoothing
                                                           // High band is too loud relative to low band
        let gain = matcher.compute_gain(0.1, 0.5);
        assert!(gain < 1.0, "Should reduce gain, got {gain}");
    }

    #[test]
    fn test_loudness_gain_boost() {
        let mut matcher = LoudnessMatcher::new(-6.0, 1.0);
        // High band is too quiet
        let gain = matcher.compute_gain(0.5, 0.01);
        assert!(gain > 1.0, "Should boost gain, got {gain}");
    }

    #[test]
    fn test_loudness_clamped() {
        let mut matcher = LoudnessMatcher::new(-6.0, 1.0);
        // Extreme case: gain should be clamped
        let gain = matcher.compute_gain(1.0, 0.0001);
        assert!(gain <= 2.0, "Gain should be clamped to 2.0, got {gain}");
    }

    #[test]
    fn test_silent_low_band_mutes_generated_high_band() {
        let mut matcher = LoudnessMatcher::new(-6.0, 1.0);
        assert_eq!(matcher.compute_gain(0.0, 0.1), 0.0);
    }
}
