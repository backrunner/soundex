//! Crossover blending between original low-band and generated high-band.
//!
//! Uses a raised-cosine transition to smoothly blend the two bands,
//! avoiding audible "splicing" artifacts at the cutoff frequency.

use std::fmt;

/// Invalid spectral arrays supplied to a crossover operation.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendError {
    /// Original, generated, and weight arrays must describe the same bins.
    LengthMismatch {
        /// Number of original bins.
        original: usize,
        /// Number of generated bins.
        generated: usize,
        /// Number of weights.
        weights: usize,
    },
    /// Blend weights must be finite values in the inclusive range 0..=1.
    InvalidWeight {
        /// Index of the invalid weight.
        index: usize,
        /// Invalid value.
        value: f32,
    },
}

impl fmt::Display for BlendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                original,
                generated,
                weights,
            } => write!(
                formatter,
                "spectral length mismatch: original={original}, generated={generated}, weights={weights}"
            ),
            Self::InvalidWeight { index, value } => {
                write!(formatter, "invalid crossover weight at bin {index}: {value}")
            }
        }
    }
}

impl std::error::Error for BlendError {}

pub(crate) fn validate_blend_inputs(
    original_len: usize,
    generated_len: usize,
    weights: &[f32],
) -> Result<(), BlendError> {
    if original_len != generated_len || original_len != weights.len() {
        return Err(BlendError::LengthMismatch {
            original: original_len,
            generated: generated_len,
            weights: weights.len(),
        });
    }
    if let Some((index, value)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(BlendError::InvalidWeight { index, value });
    }
    Ok(())
}

/// Computes and applies crossover blend weights.
pub struct CrossoverBlend {
    /// Width of the transition band in Hz.
    transition_width_hz: f32,
}

impl CrossoverBlend {
    /// Create a new crossover blend processor.
    ///
    /// # Arguments
    /// * `transition_width_hz` - Width of the smooth transition region (default: 1000 Hz)
    pub fn new(transition_width_hz: f32) -> Self {
        Self {
            transition_width_hz: transition_width_hz.max(0.0),
        }
    }

    /// Compute per-bin blend weights.
    ///
    /// Returns a vector where 0.0 = fully original, 1.0 = fully generated.
    ///
    /// # Arguments
    /// * `cutoff_freq` - Cutoff frequency in Hz
    /// * `bin_resolution` - Hz per bin
    /// * `num_bins` - Total number of frequency bins
    pub fn compute_weights(
        &self,
        cutoff_freq: f32,
        bin_resolution: f32,
        num_bins: usize,
    ) -> Vec<f32> {
        let mut weights = vec![0.0; num_bins];
        self.compute_weights_into(cutoff_freq, bin_resolution, &mut weights);
        weights
    }

    /// Fill caller-owned per-bin blend weights without allocating.
    pub fn compute_weights_into(&self, cutoff_freq: f32, bin_resolution: f32, weights: &mut [f32]) {
        let half_width = self.transition_width_hz / 2.0;
        let start_freq = (cutoff_freq - half_width).max(0.0);
        let end_freq = (cutoff_freq + half_width).max(start_freq);

        for (index, weight) in weights.iter_mut().enumerate() {
            let frequency = index as f32 * bin_resolution;
            *weight = if frequency <= start_freq {
                0.0
            } else if frequency >= end_freq {
                1.0
            } else {
                let progress = (frequency - start_freq) / (end_freq - start_freq).max(1.0);
                0.5 * (1.0 - (std::f32::consts::PI * progress).cos())
            };
        }
    }

    /// Apply blend weights to magnitude spectra.
    ///
    /// # Arguments
    /// * `original_mag` - Original magnitude spectrum
    /// * `generated_mag` - Generated magnitude spectrum
    /// * `weights` - Blend weights from `compute_weights`
    pub fn apply(
        &self,
        original_mag: &[f32],
        generated_mag: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>, BlendError> {
        let mut output = vec![0.0; original_mag.len()];
        self.apply_into(original_mag, generated_mag, weights, &mut output)?;
        Ok(output)
    }

    /// Blend into caller-owned output without allocating.
    pub fn apply_into(
        &self,
        original_mag: &[f32],
        generated_mag: &[f32],
        weights: &[f32],
        output: &mut [f32],
    ) -> Result<(), BlendError> {
        validate_blend_inputs(original_mag.len(), generated_mag.len(), weights)?;
        if output.len() != original_mag.len() {
            return Err(BlendError::LengthMismatch {
                original: original_mag.len(),
                generated: output.len(),
                weights: weights.len(),
            });
        }
        for (((output, &original), &generated), &weight) in output
            .iter_mut()
            .zip(original_mag)
            .zip(generated_mag)
            .zip(weights)
        {
            *output = match weight {
                0.0 => original,
                1.0 => generated,
                _ => original * (1.0 - weight) + generated * weight,
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_below_cutoff_are_zero() {
        let blend = CrossoverBlend::new(1000.0);
        let weights = blend.compute_weights(10000.0, 44_100.0 / 1024.0, 513);
        // Bins well below cutoff should be 0
        for w in &weights[..200] {
            assert_eq!(*w, 0.0);
        }
    }

    #[test]
    fn test_weights_above_cutoff_are_one() {
        let blend = CrossoverBlend::new(1000.0);
        let weights = blend.compute_weights(10000.0, 44_100.0 / 1024.0, 513);
        // Bins well above cutoff should be 1
        for w in &weights[250..] {
            assert_eq!(*w, 1.0);
        }
    }

    #[test]
    fn test_weights_transition_is_smooth() {
        let blend = CrossoverBlend::new(2000.0);
        let weights = blend.compute_weights(10000.0, 44_100.0 / 1024.0, 513);
        // In transition region, weights should be monotonically increasing
        let start_bin = 200;
        let end_bin = 260;
        for i in start_bin..end_bin {
            assert!(
                weights[i + 1] >= weights[i],
                "Not monotonic at bin {i}: {} > {}",
                weights[i],
                weights[i + 1]
            );
        }
    }

    #[test]
    fn test_apply_blend() {
        let blend = CrossoverBlend::new(1000.0);
        let original = vec![1.0, 1.0, 1.0, 1.0];
        let generated = vec![2.0, 2.0, 2.0, 2.0];
        let weights = vec![0.0, 0.25, 0.75, 1.0];
        let result = blend.apply(&original, &generated, &weights).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.25).abs() < 1e-6);
        assert!((result[2] - 1.75).abs() < 1e-6);
        assert!((result[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn zero_and_one_weights_select_inputs_exactly() {
        let blend = CrossoverBlend::new(1000.0);
        let original = vec![0.125, 0.25, 0.5];
        let generated = vec![8.0, 4.0, 2.0];
        let weights = vec![0.0, 0.5, 1.0];

        let result = blend.apply(&original, &generated, &weights).unwrap();

        assert_eq!(result[0].to_bits(), original[0].to_bits());
        assert_eq!(result[2].to_bits(), generated[2].to_bits());
    }

    #[test]
    fn mismatched_lengths_are_not_silently_truncated() {
        let blend = CrossoverBlend::new(1000.0);
        let result = blend.apply(&[1.0, 2.0], &[3.0], &[0.0, 1.0]);

        assert!(matches!(result, Err(BlendError::LengthMismatch { .. })));
    }
}
