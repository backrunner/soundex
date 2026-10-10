//! Experimental, causal spectral patching for missing high frequencies.
//!
//! This is a blind bandwidth-extension baseline, not a codec SBR decoder: no
//! original high-band envelope or side information is available. Only the upper
//! retained band is transposed. Tonal and low-cutoff material is attenuated to
//! reduce inharmonic buzzing. All storage is allocated at construction.

use std::f32::consts::{PI, TAU};

use crate::stft::SpectralFrame;

pub struct SpectralExtension {
    fft_size: usize,
    hop_size: usize,
    position: usize,
    strength: f32,
    cutoff_bin: usize,
    magnitude: Vec<f32>,
}

impl SpectralExtension {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        assert!(fft_size >= 2 && hop_size > 0 && hop_size <= fft_size);
        Self {
            fft_size,
            hop_size,
            position: 0,
            strength: 0.0,
            cutoff_bin: 0,
            magnitude: vec![0.0; fft_size / 2 + 1],
        }
    }

    /// Advance on every input hop, including bypass. The first causal frame
    /// starts at `-(fft_size - hop_size)`, equivalent to `hop_size` modulo N.
    pub fn advance(&mut self) {
        self.position = (self.position + self.hop_size) % self.fft_size;
    }

    /// Write a high-band candidate and return its broadband confidence [0, 1].
    /// No retained bins, DC, or Nyquist are generated. Phase rotation accounts
    /// for the frequency shift between successive overlapping analysis frames.
    pub fn generate_into(
        &mut self,
        input: &SpectralFrame,
        cutoff_hz: f32,
        sample_rate: u32,
        output: &mut SpectralFrame,
    ) -> f32 {
        let bins = self.magnitude.len();
        assert_eq!(input.log_magnitude.len(), bins);
        assert_eq!(input.phase.len(), bins);
        assert_eq!(output.log_magnitude.len(), bins);
        assert_eq!(output.phase.len(), bins);
        output.log_magnitude.fill(-200.0);
        output.phase.fill(0.0);
        let resolution = sample_rate as f32 / self.fft_size as f32;
        let cutoff = ((cutoff_hz / resolution).ceil() as usize).clamp(1, bins - 1);
        // Exclude the filter transition, as well as bass fundamentals.
        let end = cutoff.saturating_sub((500.0 / resolution).ceil() as usize);
        let start = (cutoff / 2).max(1);
        if end <= start + 2 || cutoff >= bins - 1 || !cutoff_hz.is_finite() {
            self.strength = 0.0;
            return 0.0;
        }
        let mut power = 0.0;
        let mut log_power = 0.0;
        for bin in start..end {
            let magnitude = 10.0_f32.powf(input.log_magnitude[bin].clamp(-200.0, 100.0) / 20.0);
            self.magnitude[bin] = magnitude;
            let p = magnitude * magnitude;
            power += p;
            log_power += p.max(1e-20).ln();
        }
        let width = end - start;
        let mean_power = power / width as f32;
        if mean_power <= 1e-18 {
            self.strength = 0.0;
            return 0.0;
        }
        let flatness = ((log_power / width as f32).exp() / mean_power).clamp(0.0, 1.0);
        let reliability = ((cutoff_hz - 6000.0) / 2000.0).clamp(0.0, 1.0);
        let target = flatness.sqrt() * reliability;
        // A changed patch map fades in again rather than abruptly moving tones.
        if cutoff != self.cutoff_bin {
            self.strength = 0.0;
            self.cutoff_bin = cutoff;
        }
        let time_ms = if target > self.strength { 5.0 } else { 20.0 };
        let alpha = 1.0 - (-(self.hop_size as f32) / (sample_rate as f32 * time_ms / 1000.0)).exp();
        self.strength += alpha * (target - self.strength);
        let peak_cap = mean_power.sqrt() * 4.0;
        for bin in cutoff..bins - 1 {
            let patch = (bin - cutoff) / width;
            let donor = start + (bin - cutoff) % width;
            let shift = bin - donor;
            // Smooth boundaries between patches and taper toward Nyquist.
            let u = (donor - start) as f32 / (width - 1) as f32;
            let taper = (PI * u).sin().powi(2);
            let rolloff = (cutoff as f32 / bin as f32) * 0.5_f32.powi(patch as i32);
            let magnitude =
                self.magnitude[donor].min(peak_cap) * 0.5 * self.strength * taper * rolloff;
            output.log_magnitude[bin] = 20.0 * magnitude.max(1e-10).log10();
            let rotation =
                TAU * ((shift * self.position) % self.fft_size) as f32 / self.fft_size as f32;
            output.phase[bin] = (input.phase[donor] + rotation + PI).rem_euclid(TAU) - PI;
        }
        self.strength
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.strength = 0.0;
        self.cutoff_bin = 0;
        self.magnitude.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_band_only_and_no_silent_energy() {
        let mut extension = SpectralExtension::new(256, 128);
        let mut input = SpectralFrame::zeros(129);
        let mut output = SpectralFrame::zeros(129);
        input.log_magnitude.fill(-200.0);
        extension.advance();
        assert_eq!(
            extension.generate_into(&input, 12000.0, 48000, &mut output),
            0.0
        );
        assert!(output.log_magnitude.iter().all(|&v| v == -200.0));
        input.log_magnitude[32..61].fill(0.0);
        assert!(extension.generate_into(&input, 12000.0, 48000, &mut output) > 0.0);
        assert!(output.log_magnitude[..64].iter().all(|&v| v == -200.0));
        assert_eq!(output.log_magnitude[128], -200.0);
        assert!(output.log_magnitude[64..128].iter().any(|&v| v > -100.0));
        assert!(output.log_magnitude.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn phase_rotation_survives_bypass_and_reset() {
        let mut extension = SpectralExtension::new(256, 128);
        let input = SpectralFrame::zeros(129);
        let mut output = SpectralFrame::zeros(129);
        extension.advance();
        extension.generate_into(&input, 12000.0, 48000, &mut output);
        let first = output.phase[65]; // shift=32 (even): no rotation
        let odd_first = output.phase[97]; // second patch shift=61 (odd)
        extension.advance();
        extension.generate_into(&input, 12000.0, 48000, &mut output);
        assert!((output.phase[65] - first).abs() < 1e-5);
        assert!((output.phase[97] - odd_first).abs() > 3.0);
        extension.reset();
        extension.advance();
        extension.generate_into(&input, 12000.0, 48000, &mut output);
        assert_eq!(output.phase[97], odd_first);
    }

    #[test]
    fn tonal_and_low_cutoff_inputs_do_not_become_broadband_hiss() {
        let mut extension = SpectralExtension::new(256, 128);
        let mut input = SpectralFrame::zeros(129);
        let mut output = SpectralFrame::zeros(129);
        input.log_magnitude.fill(-200.0);
        input.log_magnitude[40] = 0.0;
        extension.advance();
        let strength = extension.generate_into(&input, 12000.0, 48000, &mut output);
        assert!(strength < 0.001);
        input.log_magnitude.fill(0.0);
        extension.reset();
        assert_eq!(
            extension.generate_into(&input, 4000.0, 48000, &mut output),
            0.0
        );
    }
}
