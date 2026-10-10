//! Configuration for the SoundEx processor.

use std::path::PathBuf;

/// High-frequency candidate source. Every mode retains the common detection,
/// crossover, causal overlap-add, wet/dry ramp and output safety stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnhancementMode {
    /// Existing neural candidate and DSP post-processing.
    #[default]
    Neural,
    /// Experimental blind DSP extension; no model is loaded or invoked.
    Spectral,
    /// Experimental neural + broadband DSP candidate blend.
    Hybrid,
}

impl EnhancementMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neural => "neural",
            Self::Spectral => "spectral",
            Self::Hybrid => "hybrid",
        }
    }
}

impl std::str::FromStr for EnhancementMode {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "neural" => Ok(Self::Neural),
            "spectral" => Ok(Self::Spectral),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err("enhancement mode must be neural, spectral or hybrid"),
        }
    }
}

/// Neural high-band level policy. Model levels are experimental and opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HighBandGain {
    /// Match the candidate to the retained edge band with the legacy -6 dB ratio.
    #[default]
    MatchEdge,
    /// Preserve the model's calibrated high-band level; output safety still applies.
    Model,
}

impl HighBandGain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MatchEdge => "match-edge",
            Self::Model => "model",
        }
    }
}

impl std::str::FromStr for HighBandGain {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "match-edge" => Ok(Self::MatchEdge),
            "model" => Ok(Self::Model),
            _ => Err("high-band gain must be match-edge or model"),
        }
    }
}

/// Configuration for creating a [`SoundExProcessor`](crate::processor::SoundExProcessor).
#[derive(Debug, Clone)]
pub struct SoundExConfig {
    /// Candidate path; experimental alternatives are opt-in.
    pub enhancement_mode: EnhancementMode,
    /// Experimental model level preservation; the default retains legacy matching.
    pub high_band_gain: HighBandGain,
    /// Path to the ONNX model file (unused in spectral mode).
    pub model_path: PathBuf,

    /// Audio sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,

    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,

    /// FFT size for STFT analysis (default: 256).
    pub fft_size: usize,

    /// Hop size for STFT (default: 128).
    pub hop_size: usize,

    /// Bandwidth detection threshold in dB (default: -60.0).
    pub bypass_threshold_db: f32,

    /// Minimum bandwidth ratio to trigger enhancement (default: 0.85).
    pub min_bandwidth_ratio: f32,

    /// Crossover transition width in Hz (default: 1000.0).
    pub crossover_width_hz: f32,

    /// Limiter ceiling amplitude (default: 0.95).
    pub limiter_ceiling: f32,

    /// ONNX Runtime intra-op worker threads (default: 1).
    pub ort_intra_threads: usize,

    /// ONNX Runtime inter-op worker threads (default: 1).
    pub ort_inter_threads: usize,

    /// Whether ONNX Runtime may schedule graph nodes in parallel (default: false).
    pub ort_parallel_execution: bool,

    /// Request macOS audio time-constraint scheduling for the inference worker.
    /// QoS is requested independently. Other platforms ignore this flag.
    pub worker_time_constraint: bool,
}

impl Default for SoundExConfig {
    fn default() -> Self {
        Self {
            enhancement_mode: EnhancementMode::Neural,
            high_band_gain: HighBandGain::MatchEdge,
            model_path: PathBuf::from("soundex-v1.onnx"),
            sample_rate: 44100,
            channels: 1,
            fft_size: 256,
            hop_size: 128,
            bypass_threshold_db: -60.0,
            min_bandwidth_ratio: 0.85,
            crossover_width_hz: 1000.0,
            limiter_ceiling: 0.95,
            ort_intra_threads: 1,
            ort_inter_threads: 1,
            ort_parallel_execution: false,
            worker_time_constraint: true,
        }
    }
}

impl SoundExConfig {
    pub fn enhancement_mode(mut self, mode: EnhancementMode) -> Self {
        self.enhancement_mode = mode;
        self
    }
    pub fn high_band_gain(mut self, policy: HighBandGain) -> Self {
        self.high_band_gain = policy;
        self
    }

    /// Create a config with a specific model path, using defaults for everything else.
    pub fn with_model(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: path.into(),
            ..Default::default()
        }
    }

    /// Set the sample rate.
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Set the number of channels.
    pub fn channels(mut self, ch: u16) -> Self {
        self.channels = ch;
        self
    }

    /// Set the FFT size for the artifact-bound STFT contract.
    pub fn fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = fft_size;
        self
    }

    /// Set the hop size for the artifact-bound STFT contract.
    pub fn hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = hop_size;
        self
    }

    /// Set the artifact-bound crossover transition width in Hz.
    pub fn crossover_width_hz(mut self, width_hz: f32) -> Self {
        self.crossover_width_hz = width_hz;
        self
    }

    /// Set the relative spectral floor used by bandwidth detection, in dB.
    pub fn bypass_threshold_db(mut self, threshold_db: f32) -> Self {
        self.bypass_threshold_db = threshold_db;
        self
    }

    /// Set explicit ONNX Runtime intra/inter-op thread counts.
    pub fn ort_threads(mut self, intra_threads: usize, inter_threads: usize) -> Self {
        self.ort_intra_threads = intra_threads;
        self.ort_inter_threads = inter_threads;
        self
    }

    /// Enable or disable ONNX Runtime parallel graph execution.
    pub fn ort_parallel_execution(mut self, enabled: bool) -> Self {
        self.ort_parallel_execution = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_gain_policy_is_strict_and_default_remains_legacy() {
        assert_eq!(
            SoundExConfig::default().high_band_gain,
            HighBandGain::MatchEdge
        );
        assert_eq!(
            "model".parse::<HighBandGain>().unwrap(),
            HighBandGain::Model
        );
        assert!("typo".parse::<HighBandGain>().is_err());
    }
}
