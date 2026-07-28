//! Window functions for STFT analysis and synthesis.

/// Generate a Hann window of the given length.
///
/// The STFT synthesizer divides by accumulated squared-window weights, so
/// reconstruction does not depend on one specific COLA overlap ratio.
pub fn hann(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let n = length as f32;
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_endpoints() {
        let w = hann(1024);
        assert_eq!(w.len(), 1024);
        // Periodic Hann starts at zero and has a non-zero final sample.
        assert!(w[0].abs() < 1e-6);
        assert!(w[1023] > 0.0);
        // Center should be ~1
        assert!((w[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hann_empty() {
        assert!(hann(0).is_empty());
    }

    #[test]
    fn test_hann_single() {
        assert_eq!(hann(1), vec![1.0]);
    }
}
