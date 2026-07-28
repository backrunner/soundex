//! STFT analysis and synthesis with overlap-add.
//!
//! Provides forward (time → frequency) and inverse (frequency → time)
//! Short-Time Fourier Transform with configurable FFT size, hop size, and window.

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

use crate::window;

/// Spectral frame containing log-magnitude and phase for each frequency bin.
#[derive(Debug, Clone)]
pub struct SpectralFrame {
    /// Log-magnitude spectrum in dB, length = fft_size / 2 + 1.
    ///
    /// The value is `20 * log10(|X[k]|)` and is therefore compatible with the
    /// dB thresholds used by [`BandwidthDetector`](crate::bandwidth::BandwidthDetector).
    pub log_magnitude: Vec<f32>,
    /// Phase spectrum in radians, length = fft_size / 2 + 1.
    pub phase: Vec<f32>,
}

impl SpectralFrame {
    /// Allocate a reusable spectral frame with zeroed bins.
    pub fn zeros(num_bins: usize) -> Self {
        Self {
            log_magnitude: vec![0.0; num_bins],
            phase: vec![0.0; num_bins],
        }
    }
}

/// STFT analyzer: converts time-domain frames to spectral representations.
pub struct StftAnalyzer {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl StftAnalyzer {
    /// Create a new STFT analyzer.
    ///
    /// # Arguments
    /// * `fft_size` - FFT point count (e.g., 1024)
    /// * `hop_size` - Hop size in samples (e.g., 512 for 50% overlap)
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let input = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft_size,
            hop_size,
            window: window::hann(fft_size),
            fft,
            input,
            spectrum,
            scratch,
        }
    }

    /// FFT size used by this analyzer.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size used by this analyzer.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Number of frequency bins (fft_size / 2 + 1).
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Analyze a time-domain frame into log-magnitude and phase.
    ///
    /// # Arguments
    /// * `frame` - Input samples, length must equal `fft_size`
    ///
    /// # Panics
    /// Panics if `frame.len() != fft_size` (debug builds).
    pub fn analyze(&mut self, frame: &[f32]) -> SpectralFrame {
        let mut spectral = SpectralFrame::zeros(self.num_bins());
        self.analyze_into(frame, &mut spectral);
        spectral
    }

    /// Analyze into caller-owned spectral buffers without allocating.
    pub fn analyze_into(&mut self, frame: &[f32], spectral: &mut SpectralFrame) {
        debug_assert_eq!(frame.len(), self.fft_size);
        assert_eq!(spectral.log_magnitude.len(), self.num_bins());
        assert_eq!(spectral.phase.len(), self.num_bins());

        for ((value, sample), window) in self.input.iter_mut().zip(frame).zip(&self.window) {
            *value = sample * window;
        }
        self.fft
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .expect("FFT processing failed");

        for ((log_magnitude, phase), value) in spectral
            .log_magnitude
            .iter_mut()
            .zip(&mut spectral.phase)
            .zip(&self.spectrum)
        {
            *log_magnitude = 20.0 * value.norm().max(1e-10).log10();
            *phase = value.arg();
        }
    }
}

/// STFT synthesizer: converts spectral frames back to time-domain via overlap-add.
pub struct StftSynthesizer {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    fft: Arc<dyn ComplexToReal<f32>>,
    spectrum: Vec<Complex<f32>>,
    output: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    overlap_buffer: Vec<f32>,
    /// Per-sample squared-window accumulation for overlap-add normalization.
    overlap_weights: Vec<f32>,
}

impl StftSynthesizer {
    /// Create a new STFT synthesizer.
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let window = window::hann(fft_size);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(fft_size);
        let spectrum = fft.make_input_vec();
        let output = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft_size,
            hop_size,
            window,
            fft,
            spectrum,
            output,
            scratch,
            overlap_buffer: vec![0.0; fft_size],
            overlap_weights: vec![0.0; fft_size],
        }
    }

    /// Synthesize a spectral frame and return `hop_size` output samples via overlap-add.
    pub fn synthesize(&mut self, frame: &SpectralFrame) -> Vec<f32> {
        let mut result = vec![0.0; self.hop_size];
        self.synthesize_into(frame, &mut result);
        result
    }

    /// Synthesize into a caller-owned hop buffer without allocating.
    pub fn synthesize_into(&mut self, frame: &SpectralFrame, result: &mut [f32]) {
        assert_eq!(frame.log_magnitude.len(), self.spectrum.len());
        assert_eq!(frame.phase.len(), self.spectrum.len());
        assert_eq!(result.len(), self.hop_size);
        for ((value, log_magnitude), phase) in self
            .spectrum
            .iter_mut()
            .zip(&frame.log_magnitude)
            .zip(&frame.phase)
        {
            let magnitude = 10.0f32.powf(*log_magnitude / 20.0);
            *value = Complex::from_polar(magnitude, *phase);
        }

        // realfft requires DC and Nyquist bins to have zero imaginary part
        if !self.spectrum.is_empty() {
            self.spectrum[0].im = 0.0;
            if let Some(last) = self.spectrum.last_mut() {
                last.im = 0.0;
            }
        }

        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.output, &mut self.scratch)
            .expect("IFFT processing failed");

        // IFFT gives N*x, normalize by fft_size.
        // Apply synthesis window. IFFT output is scaled by fft_size.
        for (s, w) in self.output.iter_mut().zip(self.window.iter()) {
            *s *= w / self.fft_size as f32;
        }

        // Overlap-add
        for (i, s) in self.output.iter().enumerate() {
            self.overlap_buffer[i] += s;
            self.overlap_weights[i] += self.window[i] * self.window[i];
        }

        // Extract hop_size samples
        for ((result_sample, sample), weight) in result
            .iter_mut()
            .zip(&self.overlap_buffer[..self.hop_size])
            .zip(&self.overlap_weights[..self.hop_size])
        {
            *result_sample = if *weight > 1e-12 {
                sample / weight
            } else {
                0.0
            };
        }

        // Shift buffer
        self.overlap_buffer.copy_within(self.hop_size.., 0);
        let tail_start = self.fft_size - self.hop_size;
        for v in &mut self.overlap_buffer[tail_start..] {
            *v = 0.0;
        }
        self.overlap_weights.copy_within(self.hop_size.., 0);
        for weight in &mut self.overlap_weights[tail_start..] {
            *weight = 0.0;
        }
    }

    /// Reset the overlap buffer state.
    pub fn reset(&mut self) {
        self.overlap_buffer.fill(0.0);
        self.overlap_weights.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a test signal: sum of sinusoids.
    fn test_signal(len: usize, sample_rate: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 4000.0 * t).sin()
            })
            .collect()
    }

    /// Compute SNR between original and reconstructed signals.
    fn compute_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
        let signal_power: f32 = original.iter().map(|s| s * s).sum();
        let noise_power: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r) * (o - r))
            .sum();
        if noise_power < 1e-20 {
            return 200.0; // Effectively infinite SNR
        }
        10.0 * (signal_power / noise_power).log10()
    }

    fn round_trip(signal: &[f32], fft_size: usize, hop_size: usize) -> Vec<f32> {
        assert_eq!(signal.len() % hop_size, 0);
        let mut analyzer = StftAnalyzer::new(fft_size, hop_size);
        let mut synthesizer = StftSynthesizer::new(fft_size, hop_size);
        let mut frame = vec![0.0; fft_size];
        let mut output = Vec::with_capacity(signal.len());

        for position in (0..signal.len()).step_by(hop_size) {
            frame.fill(0.0);
            let available = fft_size.min(signal.len() - position);
            frame[..available].copy_from_slice(&signal[position..position + available]);
            let spectral = analyzer.analyze(&frame);
            output.extend_from_slice(&synthesizer.synthesize(&spectral));
        }
        output
    }

    #[test]
    fn supported_stft_contracts_roundtrip_above_120_db() {
        let sample_rate = 44100.0;
        let signal_len = 16384; // ~0.37s
        let signal = test_signal(signal_len, sample_rate);

        for (fft_size, hop_size) in [(1024, 512), (2048, 512)] {
            let output = round_trip(&signal, fft_size, hop_size);
            let snr = compute_snr(&signal[fft_size..], &output[fft_size..]);
            assert!(
                snr > 120.0,
                "STFT {fft_size}/{hop_size} roundtrip SNR too low: {snr:.1} dB"
            );
        }
    }

    #[test]
    fn selected_contract_preserves_impulses_and_final_tail() {
        let fft_size = 1024;
        let hop_size = 512;
        let mut signal = vec![0.0; 8 * hop_size];
        signal[fft_size + 17] = 0.75;
        let tail_index = signal.len() - 1;
        signal[tail_index] = -0.5;

        let output = round_trip(&signal, fft_size, hop_size);
        assert_eq!(output.len(), signal.len());
        let maximum_error = signal
            .iter()
            .zip(&output)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_error < 1e-5,
            "1024/512 impulse or final-tail error was {maximum_error:.3e}"
        );
    }

    #[test]
    fn test_spectral_frame_dimensions() {
        let mut analyzer = StftAnalyzer::new(1024, 512);
        let frame = vec![0.0f32; 1024];
        let spectral = analyzer.analyze(&frame);

        assert_eq!(spectral.log_magnitude.len(), 513); // 1024/2 + 1
        assert_eq!(spectral.phase.len(), 513);
    }

    #[test]
    fn test_analyzer_accessors() {
        let analyzer = StftAnalyzer::new(1024, 512);
        assert_eq!(analyzer.fft_size(), 1024);
        assert_eq!(analyzer.hop_size(), 512);
        assert_eq!(analyzer.num_bins(), 513);
    }
}
