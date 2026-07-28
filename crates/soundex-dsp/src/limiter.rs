//! Soft limiter to prevent clipping in the enhanced output.

/// A soft-knee limiter that prevents output samples from exceeding a ceiling.
pub struct Limiter {
    /// Maximum allowed absolute amplitude (default: 0.95).
    ceiling: f32,
    /// Compression ratio above ceiling (e.g., 10.0 means 10:1).
    ratio: f32,
}

impl Limiter {
    /// Create a new limiter.
    ///
    /// # Arguments
    /// * `ceiling` - Maximum amplitude (0.0..1.0)
    /// * `ratio` - Compression ratio above ceiling
    pub fn new(ceiling: f32, ratio: f32) -> Self {
        Self {
            ceiling: ceiling.clamp(0.0, 1.0),
            ratio: ratio.max(1.0),
        }
    }

    /// Process a single sample through the soft limiter.
    #[inline]
    pub fn process_sample(&self, sample: f32) -> f32 {
        let abs_sample = sample.abs();
        let knee_start = self.ceiling * 0.9;
        if abs_sample <= knee_start {
            return sample;
        }

        let knee_width = (self.ceiling - knee_start).max(f32::EPSILON);
        let overshoot = abs_sample - knee_start;
        let compressed =
            knee_start + knee_width * (1.0 - (-overshoot / (knee_width * self.ratio)).exp());
        let sign = if sample >= 0.0 { 1.0 } else { -1.0 };
        sign * compressed
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
}
