//! Unit-aware parity policy shared with the Python exporter.

use std::{
    error::Error,
    f64::consts::{PI, TAU},
};

use ndarray::Array4;
use rustfft::{num_complex::Complex, FftPlanner};
use serde_json::{json, Value};

const POLICY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../training/parity_policy.json"
));

pub fn compare(actual: &Array4<f32>, expected: &[f32]) -> Result<Value, Box<dyn Error>> {
    let shape = actual.shape();
    if shape[0] == 0
        || shape[1] != 2
        || shape[2] != 1
        || shape[3] < 2
        || actual.len() != expected.len()
    {
        return Err("parity requires nonempty matching [B, 2, 1, F] tensors".into());
    }
    let policy: Value = serde_json::from_str(POLICY)?;
    let bins = shape[3];
    let fft_size = (bins - 1) * 2;
    let mut magnitude_max = 0.0_f64;
    let mut magnitude_sum = 0.0;
    let mut phase_max = 0.0_f64;
    let mut phase_sum = 0.0;
    let mut relative_max = 0.0_f64;
    let mut bound_max = 0.0_f64;
    let mut inverse_max = 0.0_f64;
    let inverse = FftPlanner::<f64>::new().plan_fft_inverse(fft_size);
    let mut raw_max = 0.0_f64;
    let mut raw_sum = 0.0;
    for batch in 0..shape[0] {
        let offset = batch * 2 * bins;
        let mut error_energy = 0.0;
        let mut reference_energy = 0.0;
        let mut inverse_bound = 0.0;
        let mut difference_spectrum = vec![Complex::new(0.0, 0.0); fft_size];
        for bin in 0..bins {
            let expected_db = f64::from(expected[offset + bin]);
            let expected_phase = f64::from(expected[offset + bins + bin]);
            let actual_db = f64::from(actual[[batch, 0, 0, bin]]);
            let actual_phase = f64::from(actual[[batch, 1, 0, bin]]);
            if [expected_db, expected_phase, actual_db, actual_phase]
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err("runtime or golden tensor contains a non-finite value".into());
            }
            let db_error = (actual_db - expected_db).abs();
            let phase_raw = (actual_phase - expected_phase).abs();
            let phase_error = ((actual_phase - expected_phase + PI).rem_euclid(TAU) - PI).abs();
            magnitude_max = magnitude_max.max(db_error);
            magnitude_sum += db_error;
            phase_max = phase_max.max(phase_error);
            phase_sum += phase_error;
            raw_max = raw_max.max(db_error).max(phase_raw);
            raw_sum += db_error + phase_raw;
            let reference_amp = 10.0_f64.powf(expected_db / 20.0);
            let actual_amp = 10.0_f64.powf(actual_db / 20.0);
            let real_delta = actual_amp * actual_phase.cos() - reference_amp * expected_phase.cos();
            let imaginary_delta =
                actual_amp * actual_phase.sin() - reference_amp * expected_phase.sin();
            let error = real_delta.hypot(imaginary_delta);
            difference_spectrum[bin] = Complex::new(real_delta, imaginary_delta);
            if bin > 0 && bin < bins - 1 {
                difference_spectrum[fft_size - bin] = difference_spectrum[bin].conj();
            }
            let weight = if bin == 0 || bin == bins - 1 {
                1.0
            } else {
                2.0
            };
            error_energy += weight * error * error;
            reference_energy += weight * reference_amp * reference_amp;
            inverse_bound += weight * error;
        }
        let reference_rms = (reference_energy / fft_size as f64).sqrt();
        let error_rms = (error_energy / fft_size as f64).sqrt();
        let relative = error_rms
            / reference_rms.max(
                policy["complex_rms_floor"]
                    .as_f64()
                    .ok_or("invalid policy")?,
            );
        let bound = inverse_bound / fft_size as f64;
        difference_spectrum[0].im = 0.0;
        difference_spectrum[bins - 1].im = 0.0;
        inverse.process(&mut difference_spectrum);
        for value in &difference_spectrum {
            let sample_error = value.re.abs() / fft_size as f64;
            if !sample_error.is_finite() {
                return Err("non-finite inverse FFT parity metric".into());
            }
            inverse_max = inverse_max.max(sample_error);
        }
        if !relative.is_finite() || !bound.is_finite() {
            return Err("non-finite parity metric".into());
        }
        relative_max = relative_max.max(relative);
        bound_max = bound_max.max(bound);
    }
    let count = (shape[0] * bins) as f64;
    let metrics = json!({
        "max_magnitude_db": magnitude_max,
        "mean_magnitude_db": magnitude_sum / count,
        "max_phase_rad": phase_max,
        "mean_phase_rad": phase_sum / count,
        "max_complex_relative_rms": relative_max,
        "max_inverse_fft_error": inverse_max,
        "raw_inverse_fft_error_bound": bound_max,
        "raw_max_absolute_error": raw_max,
        "raw_mean_error": raw_sum / (count * 2.0)
    });
    for (field, value) in metrics.as_object().ok_or("invalid metrics")? {
        let number = value.as_f64().ok_or("non-finite parity metric")?;
        if field.starts_with("raw_") {
            continue;
        }
        let limit = policy[field].as_f64().ok_or("missing parity budget")?;
        if !limit.is_finite() || limit <= 0.0 {
            return Err("invalid parity budget".into());
        }
        if number > limit {
            return Err(format!("parity mismatch: {field}={number:.8e}, limit={limit:.3e}").into());
        }
    }
    Ok(json!({"policy": policy, "metrics": metrics}))
}
