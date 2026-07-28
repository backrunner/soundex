//! Bandwidth detection (gatekeeping).
//!
//! Detects whether the input audio has truncated high-frequency content
//! typical of lossy compression (MP3, AAC). If bandwidth is full, the
//! enhancement model is bypassed.

/// Configuration for the bandwidth detector.
#[derive(Debug, Clone)]
pub struct BandwidthDetectorConfig {
    /// Energy drop threshold in dB below mean to consider as noise floor.
    pub threshold_db: f32,
    /// Minimum bandwidth ratio (detected_bw / nyquist) to trigger enhancement.
    pub min_bandwidth_ratio: f32,
    /// Exponential smoothing coefficient (0..1).
    pub smoothing_alpha: f32,
    /// Number of frames to hold enhancement state after trigger.
    pub hangover_frames: usize,
}

impl Default for BandwidthDetectorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -60.0,
            min_bandwidth_ratio: 0.85,
            smoothing_alpha: 0.1,
            hangover_frames: 10,
        }
    }
}

/// Detects effective bandwidth of input audio frames.
pub struct BandwidthDetector {
    config: BandwidthDetectorConfig,
    smoothed_bandwidth: f32,
    hangover_counter: usize,
}

impl BandwidthDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: BandwidthDetectorConfig) -> Self {
        Self {
            config: BandwidthDetectorConfig {
                threshold_db: if config.threshold_db.is_finite() {
                    config.threshold_db.min(0.0)
                } else {
                    -60.0
                },
                min_bandwidth_ratio: config.min_bandwidth_ratio.clamp(0.0, 1.0),
                smoothing_alpha: config.smoothing_alpha.clamp(0.0, 1.0),
                hangover_frames: config.hangover_frames,
            },
            smoothed_bandwidth: 0.0,
            hangover_counter: 0,
        }
    }

    /// Detect effective bandwidth from a log-magnitude spectrum.
    ///
    /// Returns the detected bandwidth in Hz.
    pub fn detect(&mut self, log_magnitude: &[f32], sample_rate: u32) -> f32 {
        if log_magnitude.is_empty() || sample_rate == 0 {
            return 0.0;
        }
        let nyquist = sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / (log_magnitude.len() - 1).max(1) as f32;

        // Use peak energy as reference (more robust than mean for truncated spectra)
        let peak_energy = log_magnitude
            .iter()
            .filter(|value| value.is_finite())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if !peak_energy.is_finite() {
            return 0.0;
        }

        // Scan from high frequency to low, find first bin above noise floor
        let noise_floor = peak_energy + self.config.threshold_db;
        let mut cutoff_bin = 0;
        for i in (1..log_magnitude.len()).rev() {
            if log_magnitude[i] > noise_floor {
                cutoff_bin = i;
                break;
            }
        }

        let detected_bw = cutoff_bin as f32 * bin_resolution;

        // Exponential smoothing
        if self.smoothed_bandwidth == 0.0 {
            self.smoothed_bandwidth = detected_bw;
        } else {
            self.smoothed_bandwidth = self.config.smoothing_alpha * detected_bw
                + (1.0 - self.config.smoothing_alpha) * self.smoothed_bandwidth;
        }

        self.smoothed_bandwidth
    }

    /// Determine whether enhancement is needed based on detected bandwidth.
    pub fn needs_enhancement(&mut self, detected_bw: f32, nyquist: f32) -> bool {
        let ratio = detected_bw / nyquist.max(1.0);
        let needs = ratio < self.config.min_bandwidth_ratio;

        if needs {
            self.hangover_counter = self.config.hangover_frames;
        } else if self.hangover_counter > 0 {
            self.hangover_counter -= 1;
            return true;
        }

        needs
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.smoothed_bandwidth = 0.0;
        self.hangover_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a synthetic spectrum with energy up to `cutoff_bin`, silence above.
    fn truncated_spectrum(num_bins: usize, cutoff_bin: usize, floor_db: f32) -> Vec<f32> {
        (0..num_bins)
            .map(|i| {
                if i <= cutoff_bin {
                    -20.0 + (i as f32 * 0.01).sin() * 5.0 // Signal around -20 dB
                } else {
                    floor_db // Noise floor
                }
            })
            .collect()
    }

    #[test]
    fn test_detect_full_bandwidth() {
        let mut detector = BandwidthDetector::new(BandwidthDetectorConfig::default());
        // Full spectrum: energy in all bins
        let spectrum = vec![-20.0f32; 513];
        let bw = detector.detect(&spectrum, 44100);
        // Should detect close to Nyquist
        assert!(bw > 20000.0, "Expected full bandwidth, got {bw} Hz");
    }

    #[test]
    fn test_detect_truncated_bandwidth() {
        let mut detector = BandwidthDetector::new(BandwidthDetectorConfig {
            threshold_db: -40.0,
            ..Default::default()
        });
        // Spectrum truncated at bin 250 (~10.8kHz at 44.1kHz)
        let spectrum = truncated_spectrum(513, 250, -100.0);
        let bw = detector.detect(&spectrum, 44100);
        // Should detect around 10-12 kHz
        assert!(bw < 15000.0, "Expected truncated bandwidth, got {bw} Hz");
        assert!(bw > 5000.0, "Bandwidth too low: {bw} Hz");
    }

    #[test]
    fn test_needs_enhancement_truncated() {
        let mut detector = BandwidthDetector::new(BandwidthDetectorConfig::default());
        let nyquist = 22050.0;
        // Simulate truncated bandwidth at 12kHz
        assert!(detector.needs_enhancement(12000.0, nyquist));
    }

    #[test]
    fn test_needs_enhancement_full() {
        let mut detector = BandwidthDetector::new(BandwidthDetectorConfig::default());
        let nyquist = 22050.0;
        // Full bandwidth (95% of Nyquist)
        assert!(!detector.needs_enhancement(21000.0, nyquist));
    }

    #[test]
    fn test_hangover_logic() {
        let mut detector = BandwidthDetector::new(BandwidthDetectorConfig {
            hangover_frames: 3,
            ..Default::default()
        });
        let nyquist = 22050.0;

        // Trigger enhancement
        assert!(detector.needs_enhancement(10000.0, nyquist));
        // Hangover: still active even with full bandwidth
        assert!(detector.needs_enhancement(21000.0, nyquist));
        assert!(detector.needs_enhancement(21000.0, nyquist));
        assert!(detector.needs_enhancement(21000.0, nyquist));
        // Hangover expired
        assert!(!detector.needs_enhancement(21000.0, nyquist));
    }
}
