//! Experimental envelope-driven high-band fill (P108/P109 design).
//!
//! A causal model predicts per-frame half-ERB band energies above 4 kHz. Above
//! the detected cutoff, a hop-coherent copy-up of the 4 kHz..cutoff band is
//! added with that energy. Retained input is never attenuated. This mirrors
//! `training/evaluation/envelope_fill.py`; all buffers are allocated up front.

use crate::stft::SpectralFrame;
use crate::window;

/// Lowest frequency of the predicted high band and of the copy-up source.
pub const LOW_HZ: f64 = 4000.0;
/// Band width in ERB-number units (Glasberg & Moore).
pub const ERB_STEP: f64 = 0.5;
/// Highest frequency covered by the shared band index (48 kHz Nyquist).
pub const MAX_HZ: f64 = 24000.0;
/// Power floor inside log10, in calibrated PCM mean-square units.
pub const LOG_FLOOR: f32 = 1e-12;
/// Amplitude cap on the patch gain; a silent source band cannot be raised arbitrarily.
pub const MAX_GAIN: f32 = 1e3;

fn erb_number(hz: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * hz).log10()
}

/// How the predicted band energy is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillTarget {
    /// Prediction is the reference band energy; add max(predicted - present, 0).
    Level,
    /// Prediction is the missing band energy itself; add it directly.
    Deficit,
}

/// Bin-to-band mapping and feature scaling for one FFT size and sample rate.
#[derive(Debug, Clone)]
pub struct EnvelopeLayout {
    band_of_bin: Vec<i32>,
    band_bins: Vec<u32>,
    low_bin: usize,
    low_bins: usize,
    calibration: f32,
    rate_flag: f32,
    bin_resolution: f32,
}

impl EnvelopeLayout {
    /// Total band slots shared by 44.1 and 48 kHz (bands past Nyquist stay empty).
    pub fn band_count() -> usize {
        (erb_number(MAX_HZ) / ERB_STEP).floor() as usize
            - (erb_number(LOW_HZ) / ERB_STEP).floor() as usize
            + 1
    }

    pub fn new(fft_size: usize, sample_rate: u32) -> Self {
        let bins = fft_size / 2 + 1;
        let rate = sample_rate as f64;
        let first = (erb_number(LOW_HZ) / ERB_STEP).floor() as i64;
        let mut band_of_bin = vec![-1; bins];
        let mut band_bins = vec![0u32; Self::band_count()];
        let mut low_bin = bins;
        let mut low_bins = 0;
        for (k, band) in band_of_bin.iter_mut().enumerate() {
            let hz = k as f64 * rate / fft_size as f64;
            if hz < LOW_HZ {
                low_bins += 1;
                continue;
            }
            low_bin = low_bin.min(k);
            let id = (erb_number(hz) / ERB_STEP).floor() as i64 - first;
            *band = id as i32;
            band_bins[id as usize] += 1;
        }
        let energy: f64 = window::hann(fft_size)
            .iter()
            .map(|&w| (w as f64) * (w as f64))
            .sum();
        Self {
            band_of_bin,
            band_bins,
            low_bin: low_bin.max(1),
            low_bins,
            calibration: (2.0 / (fft_size as f64 * energy)) as f32,
            rate_flag: if sample_rate == 48_000 { 1.0 } else { 0.0 },
            bin_resolution: (rate / 2.0 / (bins - 1) as f64) as f32,
        }
    }

    /// Number of model input features: per-bin log power, loudness, rate flag.
    pub fn feature_count(&self) -> usize {
        self.band_of_bin.len() + 2
    }

    /// Write current-frame features; returns the log10 loudness reference.
    pub fn features_into(&self, log_magnitude: &[f32], out: &mut [f32]) -> f32 {
        let bins = self.band_of_bin.len();
        debug_assert_eq!(log_magnitude.len(), bins);
        debug_assert_eq!(out.len(), bins + 2);
        let mut low_power = 0.0f64;
        for (k, (&db, value)) in log_magnitude.iter().zip(out.iter_mut()).enumerate() {
            let power = 10f32.powf(db / 10.0) * self.calibration;
            if k < self.low_bins {
                low_power += power as f64;
            }
            *value = (power + LOG_FLOOR).log10();
        }
        let reference = ((low_power / self.low_bins.max(1) as f64) as f32 + LOG_FLOOR).log10();
        for value in &mut out[..bins] {
            *value -= reference;
        }
        out[bins] = reference / 10.0;
        out[bins + 1] = self.rate_flag;
        reference
    }
}

/// Applies predicted band energies above the detected cutoff.
pub struct EnvelopeFill {
    layout: EnvelopeLayout,
    target: FillTarget,
    gate_db: Option<(f32, f32)>,
    real: Vec<f32>,
    imag: Vec<f32>,
    gain: Vec<f32>,
    patch_power: Vec<f32>,
    present_power: Vec<f32>,
    band_selected: Vec<u32>,
}

impl EnvelopeFill {
    pub fn new(fft_size: usize, sample_rate: u32, target: FillTarget) -> Self {
        let layout = EnvelopeLayout::new(fft_size, sample_rate);
        let bins = fft_size / 2 + 1;
        let bands = EnvelopeLayout::band_count();
        Self {
            layout,
            target,
            gate_db: None,
            real: vec![0.0; bins],
            imag: vec![0.0; bins],
            gain: vec![0.0; bands],
            patch_power: vec![0.0; bands],
            present_power: vec![0.0; bands],
            band_selected: vec![0; bands],
        }
    }

    pub fn layout(&self) -> &EnvelopeLayout {
        &self.layout
    }

    /// Soft gate on 10*log10(fill / present band energy): no fill at or below
    /// `low_db`, full fill at or above `high_db`, linear in dB between. A band
    /// that is already present (clean input) receives nothing; a band that is
    /// missing (deficit far above what remains) is filled in full.
    pub fn with_gate_db(mut self, low_db: f32, high_db: f32) -> Self {
        assert!(
            low_db.is_finite() && high_db > low_db,
            "invalid envelope gate"
        );
        self.gate_db = Some((low_db, high_db));
        self
    }

    /// Fill `out` from `original`. `band_log10` holds absolute log10 band energies
    /// (calibrated units). `hop_index` is the frame index since reset; only its
    /// parity times the bin shift matters for hop-coherent translation.
    pub fn apply(
        &mut self,
        original: &SpectralFrame,
        band_log10: &[f32],
        detected_bandwidth_hz: f32,
        hop_index: u64,
        out: &mut SpectralFrame,
    ) {
        let bins = original.log_magnitude.len();
        out.log_magnitude.copy_from_slice(&original.log_magnitude);
        out.phase.copy_from_slice(&original.phase);
        let cut = ((detected_bandwidth_hz / self.layout.bin_resolution).round() as usize)
            .clamp(1, bins - 1);
        let start = cut + 1;
        let low = self.layout.low_bin;
        if start >= bins || start <= low {
            return;
        }
        let width = start - low;
        for k in 0..bins {
            let magnitude = 10f32.powf(original.log_magnitude[k] / 20.0);
            let (sin, cos) = original.phase[k].sin_cos();
            self.real[k] = magnitude * cos;
            self.imag[k] = magnitude * sin;
        }
        self.patch_power.fill(0.0);
        self.present_power.fill(0.0);
        self.band_selected.fill(0);
        let cal = self.layout.calibration;
        for k in start..bins {
            let source = low + (k - start) % width;
            let band = self.layout.band_of_bin[k] as usize;
            let source_power =
                self.real[source] * self.real[source] + self.imag[source] * self.imag[source];
            self.patch_power[band] += source_power * cal;
            self.present_power[band] +=
                (self.real[k] * self.real[k] + self.imag[k] * self.imag[k]) * cal;
            self.band_selected[band] += 1;
        }
        self.update_gains(band_log10);
        let odd_hop = hop_index % 2 == 1;
        for k in start..bins {
            let source = low + (k - start) % width;
            let band = self.layout.band_of_bin[k] as usize;
            let gain = self.gain[band];
            if gain == 0.0 {
                continue;
            }
            // exp(i*pi*shift*t) is +-1: a shift of s bins advances phase by pi*s per hop.
            let sign = if odd_hop && (k - source) % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            let re = self.real[k] + sign * gain * self.real[source];
            let im = self.imag[k] + sign * gain * self.imag[source];
            out.log_magnitude[k] = 20.0 * (re * re + im * im).sqrt().max(1e-10).log10();
            out.phase[k] = im.atan2(re);
        }
    }

    fn update_gains(&mut self, band_log10: &[f32]) {
        let bands = self
            .gain
            .iter_mut()
            .zip(&self.band_selected)
            .zip(&self.layout.band_bins)
            .zip(self.patch_power.iter().zip(&self.present_power))
            .zip(band_log10);
        for ((((gain, &selected), &total), (&patch, &present)), &log10) in bands {
            *gain = 0.0;
            if selected == 0 {
                continue;
            }
            let want = 10f32.powf(log10) * selected as f32 / total.max(1) as f32;
            let mut fill = match self.target {
                FillTarget::Deficit => want,
                FillTarget::Level => (want - present).max(0.0),
            };
            if let Some((low, high)) = self.gate_db {
                let ratio_db = 10.0 * (fill.max(1e-30) / present.max(1e-30)).log10();
                fill *= ((ratio_db - low) / (high - low)).clamp(0.0, 1.0);
            }
            *gain = (fill / patch.max(1e-30)).sqrt().min(MAX_GAIN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_training_band_geometry() {
        assert_eq!(EnvelopeLayout::band_count(), 33);
        let l44 = EnvelopeLayout::new(256, 44_100);
        let l48 = EnvelopeLayout::new(256, 48_000);
        let used = |l: &EnvelopeLayout| l.band_bins.iter().filter(|&&n| n > 0).count();
        assert_eq!(used(&l44), 32);
        assert_eq!(used(&l48), 33);
        assert_eq!(l44.band_bins.iter().sum::<u32>(), 105);
        assert_eq!(l48.band_bins.iter().sum::<u32>(), 107);
        // 2 / (256 * sum(hann^2)) with sum(hann^2) = 3N/8 = 96.
        assert!((l44.calibration - 2.0 / (256.0 * 96.0)).abs() < 1e-12);
    }

    #[test]
    fn features_are_gain_relative() {
        let layout = EnvelopeLayout::new(256, 48_000);
        let a: Vec<f32> = (0..129).map(|k| 10.0 - k as f32 * 0.1).collect();
        let b: Vec<f32> = a.iter().map(|v| v + 12.0).collect();
        let (mut fa, mut fb) = (vec![0.0; 131], vec![0.0; 131]);
        let ra = layout.features_into(&a, &mut fa);
        let rb = layout.features_into(&b, &mut fb);
        assert!((rb - ra - 1.2).abs() < 1e-4);
        for k in 0..129 {
            assert!((fa[k] - fb[k]).abs() < 1e-4);
        }
        assert_eq!(fa[130], 1.0);
    }

    #[test]
    fn floor_deficit_leaves_input_nearly_unchanged_and_level_never_attenuates() {
        let mut frame = SpectralFrame::zeros(129);
        for k in 0..129 {
            frame.log_magnitude[k] = if k < 90 { -10.0 } else { -200.0 };
            frame.phase[k] = 0.3 * k as f32;
        }
        let mut out = SpectralFrame::zeros(129);
        let mut fill = EnvelopeFill::new(256, 44_100, FillTarget::Level);
        // Predicted level far below present: nothing is added anywhere.
        fill.apply(&frame, &[-30.0; 33], 15_000.0, 7, &mut out);
        assert_eq!(out.log_magnitude, frame.log_magnitude);
        // A real deficit raises the empty band to the requested energy.
        let mut fill = EnvelopeFill::new(256, 44_100, FillTarget::Deficit);
        fill.apply(&frame, &[-4.0; 33], 15_000.0, 7, &mut out);
        assert!(out.log_magnitude[100] > -100.0);
        assert!(out.log_magnitude[..88] == frame.log_magnitude[..88]);
    }

    #[test]
    fn gate_skips_present_bands_and_fills_missing_ones() {
        let mut frame = SpectralFrame::zeros(129);
        for k in 0..129 {
            // Full-band input with a hole in 100..110.
            frame.log_magnitude[k] = if (100..110).contains(&k) {
                -200.0
            } else {
                -10.0
            };
            frame.phase[k] = 0.1 * k as f32;
        }
        let mut out = SpectralFrame::zeros(129);
        let mut fill = EnvelopeFill::new(256, 44_100, FillTarget::Deficit).with_gate_db(0.0, 6.0);
        // On full-band input a deficit below present energy is gated out entirely.
        let mut full = frame.clone();
        full.log_magnitude.fill(-10.0);
        fill.apply(&full, &[-12.0; 33], 15_000.0, 3, &mut out);
        assert_eq!(out.log_magnitude, full.log_magnitude);
        // A large deficit fills the band that is actually empty.
        fill.apply(&frame, &[-4.0; 33], 15_000.0, 3, &mut out);
        assert!(out.log_magnitude[105] > -100.0);
    }
}
