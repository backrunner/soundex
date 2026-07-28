//! Phase smoothing and consistency repair.
//!
//! Blends original and generated unit phasors with the same weights used for
//! magnitude crossover, preserving the low band exactly.

use rustfft::num_complex::Complex;

use crate::crossover::{validate_blend_inputs, BlendError};

/// Smooths phase between original and generated spectral content.
pub struct PhaseSmoother;

impl PhaseSmoother {
    /// Create a phase smoother.
    pub fn new() -> Self {
        Self
    }

    /// Blend original and predicted phase using shared per-bin crossover weights.
    ///
    /// Weight zero preserves the original phase bit-for-bit, weight one copies
    /// the prediction, and intermediate weights interpolate unit phasors.
    pub fn smooth(
        &self,
        original_phase: &[f32],
        predicted_phase: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>, BlendError> {
        let mut output = vec![0.0; original_phase.len()];
        self.smooth_into(original_phase, predicted_phase, weights, &mut output)?;
        Ok(output)
    }

    /// Blend into caller-owned phase storage without allocating.
    pub fn smooth_into(
        &self,
        original_phase: &[f32],
        predicted_phase: &[f32],
        weights: &[f32],
        output: &mut [f32],
    ) -> Result<(), BlendError> {
        validate_blend_inputs(original_phase.len(), predicted_phase.len(), weights)?;
        if output.len() != original_phase.len() {
            return Err(BlendError::LengthMismatch {
                original: original_phase.len(),
                generated: output.len(),
                weights: weights.len(),
            });
        }
        for (((output, &original), &predicted), &weight) in output
            .iter_mut()
            .zip(original_phase)
            .zip(predicted_phase)
            .zip(weights)
        {
            *output = match weight {
                0.0 => original,
                1.0 => predicted,
                _ => {
                    let original_unit = Complex::from_polar(1.0, original);
                    let predicted_unit = Complex::from_polar(1.0, predicted);
                    let blended = original_unit * (1.0 - weight) + predicted_unit * weight;
                    if blended.norm_sqr() > f32::EPSILON {
                        blended.arg()
                    } else {
                        circular_lerp(original, predicted, weight)
                    }
                }
            };
        }
        Ok(())
    }
}

impl Default for PhaseSmoother {
    fn default() -> Self {
        Self::new()
    }
}

fn circular_lerp(original: f32, predicted: f32, weight: f32) -> f32 {
    let delta = (predicted - original)
        .sin()
        .atan2((predicted - original).cos());
    let phase = original + weight * delta;
    phase.sin().atan2(phase.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_weights_select_phases_exactly() {
        let smoother = PhaseSmoother::new();
        let original = vec![0.125, 0.25, 0.5, 0.75, 1.0];
        let predicted = vec![3.0; original.len()];
        let weights = vec![0.0, 0.0, 0.5, 1.0, 1.0];

        let result = smoother.smooth(&original, &predicted, &weights).unwrap();

        for index in 0..2 {
            assert_eq!(result[index].to_bits(), original[index].to_bits());
        }
        assert_eq!(result[4].to_bits(), predicted[4].to_bits());
    }

    #[test]
    fn phase_wrap_transition_stays_near_pi() {
        let smoother = PhaseSmoother::new();
        let original = vec![3.13; 5];
        let predicted = vec![-3.13; 5];
        let weights = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let result = smoother.smooth(&original, &predicted, &weights).unwrap();

        assert!(
            result[2].abs() > 3.0,
            "phase wrapped through zero: {}",
            result[2]
        );
    }

    #[test]
    fn mismatched_lengths_return_an_error() {
        let smoother = PhaseSmoother::new();
        let result = smoother.smooth(&[0.0, 1.0], &[0.0], &[0.0, 1.0]);

        assert!(matches!(result, Err(BlendError::LengthMismatch { .. })));
    }
}
