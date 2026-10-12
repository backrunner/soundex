//! Soft limiter to prevent clipping in the enhanced output.

/// Output protection policy. Experimental alternatives must be selected explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimiterMode {
    #[default]
    Legacy,
    /// Same knee/ceiling with continuous first and second derivatives at the knee.
    SmoothKnee,
}

impl LimiterMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::SmoothKnee => "smooth-knee",
        }
    }
}

impl std::str::FromStr for LimiterMode {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "smooth-knee" => Ok(Self::SmoothKnee),
            _ => Err("limiter mode must be legacy or smooth-knee"),
        }
    }
}

/// A soft-knee limiter that prevents output samples from exceeding a ceiling.
pub struct Limiter {
    /// Maximum allowed absolute amplitude (default: 0.95).
    ceiling: f32,
    /// Compression ratio above ceiling (e.g., 10.0 means 10:1).
    ratio: f32,
    mode: LimiterMode,
}

impl Limiter {
    /// Create a new limiter.
    ///
    /// # Arguments
    /// * `ceiling` - Maximum amplitude (0.0..1.0)
    /// * `ratio` - Compression ratio above ceiling
    pub fn new(ceiling: f32, ratio: f32) -> Self {
        Self::with_mode(ceiling, ratio, LimiterMode::Legacy)
    }

    /// Preserve legacy behavior or opt into a smooth, memoryless shoulder.
    /// No lookahead, buffering, envelope state or allocation is introduced.
    pub fn with_mode(ceiling: f32, ratio: f32, mode: LimiterMode) -> Self {
        Self {
            mode,
            ceiling: ceiling.clamp(0.0, 1.0),
            ratio: ratio.max(1.0),
        }
    }

    /// Process a single sample through the soft limiter.
    #[inline]
    pub fn process_sample(&self, sample: f32) -> f32 {
        if self.ceiling == 0.0 {
            return 0.0;
        }
        let abs_sample = sample.abs();
        let knee_start = self.ceiling * 0.9;
        if abs_sample <= knee_start {
            return sample;
        }

        if self.mode == LimiterMode::SmoothKnee {
            // Normalize before shaping so tiny positive ceilings remain bounded.
            // y/c = 0.9 + 0.1*tanh((|x|/c - 0.9)/0.1): slope 1 and
            // curvature 0 at the knee, asymptotic ceiling, odd symmetry.
            let shaped =
                self.ceiling * (0.9 + 0.1 * ((abs_sample / self.ceiling - 0.9) * 10.0).tanh());
            return sample.signum() * shaped.min(self.ceiling);
        }

        let knee_width = (self.ceiling - knee_start).max(f32::EPSILON);
        let overshoot = abs_sample - knee_start;
        let compressed =
            knee_start + knee_width * (1.0 - (-overshoot / (knee_width * self.ratio)).exp());
        let sign = if sample >= 0.0 { 1.0 } else { -1.0 };
        sign * compressed.clamp(0.0, self.ceiling)
    }

    /// Final guard after mixing already-limited paths; no second compression.
    #[inline]
    pub fn clamp_sample(&self, sample: f32) -> f32 {
        sample.clamp(-self.ceiling, self.ceiling)
    }

    /// Process a buffer in-place.
    pub fn process_buffer(&self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }
}

impl Default for Limiter {
    fn default() -> Self {
        Self {
            ceiling: 0.95,
            ratio: 10.0,
            mode: LimiterMode::Legacy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_ceiling_passthrough() {
        let limiter = Limiter::default();
        assert_eq!(limiter.process_sample(0.5), 0.5);
        assert_eq!(limiter.process_sample(-0.3), -0.3);
    }

    #[test]
    fn test_above_ceiling_compressed() {
        let limiter = Limiter::new(0.9, 10.0);
        let out = limiter.process_sample(1.0);
        assert!(out < 0.9);
        assert!(out > 0.81);
        assert!(limiter.process_sample(100.0) <= 0.9);
    }

    #[test]
    fn zero_and_tiny_ceilings_are_strictly_respected() {
        for ceiling in [0.0, 1e-20, 0.95] {
            let limiter = Limiter::new(ceiling, 10.0);
            for sample in [-f32::MAX, -1.0, 0.0, 1.0, f32::MAX] {
                let output = limiter.process_sample(sample);
                assert!(output.is_finite() && output.abs() <= ceiling);
            }
        }
    }
}

#[cfg(test)]
mod smooth_tests {
    use super::*;

    #[test]
    fn smooth_knee_is_transparent_odd_monotone_and_bounded() {
        for ceiling in [0.0, 1e-20, 0.5, 0.95, 1.0] {
            let limiter = Limiter::with_mode(ceiling, 10.0, LimiterMode::SmoothKnee);
            let mut last = 0.0;
            for i in 0..=20000 {
                let x = i as f32 / 10000.0 * ceiling;
                let y = limiter.process_sample(x);
                assert!(y.is_finite() && y >= last && y <= ceiling);
                assert_eq!(limiter.process_sample(-x), -y);
                if x <= ceiling * 0.9 {
                    assert_eq!(y, x);
                }
                last = y;
            }
            for x in [-f32::MAX, f32::MAX] {
                assert!(limiter.process_sample(x).abs() <= ceiling);
            }
        }
        assert!("typo".parse::<LimiterMode>().is_err());
        assert_eq!(
            "smooth-knee".parse::<LimiterMode>().unwrap(),
            LimiterMode::SmoothKnee
        );
    }

    #[test]
    fn smooth_knee_has_unit_slope_at_compression_onset() {
        let limiter = Limiter::with_mode(0.95, 10.0, LimiterMode::SmoothKnee);
        let knee = 0.95 * 0.9;
        let delta = 1e-4;
        for sign in [-1.0, 1.0] {
            let slope = (limiter.process_sample(knee + sign * delta) - knee) / (sign * delta);
            assert!((slope - 1.0).abs() < 0.002, "slope {slope}");
        }
    }

    #[test]
    fn smooth_knee_reduces_legal_peak_distortion_without_raising_ceiling() {
        let smooth = Limiter::with_mode(0.95, 10.0, LimiterMode::SmoothKnee);
        let legacy = Limiter::default();
        let mut errors = [0.0_f64; 2];
        for n in 0..48000 {
            let x = 0.95 * (std::f32::consts::TAU * 1000.0 * n as f32 / 48000.0).sin();
            for (i, limiter) in [&legacy, &smooth].iter().enumerate() {
                errors[i] += (limiter.process_sample(x) - x).powi(2) as f64;
            }
        }
        assert!(errors[1] < errors[0] * 0.1, "squared errors {errors:?}");
    }
}
