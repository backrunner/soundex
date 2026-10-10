//! # soundex-core
//!
//! Core library for SoundEx real-time audio enhancement.
//!
//! This crate provides the main [`SoundExProcessor`] that combines:
//! - STFT-based spectral analysis
//! - Bandwidth detection (gatekeeping)
//! - AI model inference for high-frequency restoration
//! - DSP post-processing (loudness, phase, crossover, limiting)
//!
//! ## Usage
//!
//! ```no_run
//! use soundex_core::{SoundExConfig, SoundExProcessor};
//!
//! let config = SoundExConfig::with_model("models/soundex-v1.onnx")
//!     .sample_rate(44100)
//!     .channels(2);
//!
//! let mut processor = SoundExProcessor::new(config).unwrap();
//! let input: Vec<f32> = vec![0.0; 44100]; // 1 second of silence
//! let output = processor.process_buffer(&input).unwrap();
//! ```

mod channel;
#[cfg(feature = "ort-backend")]
mod inference_contract;

pub mod analysis;
pub mod config;
pub mod error;
pub mod inference;
pub mod processor;
pub mod realtime;
pub mod realtime_policy;
pub mod stream;

pub use analysis::{analyze_buffer, AnalysisInfo};
pub use config::{EnhancementMode, SoundExConfig};
pub use error::{Result, SoundExError};
pub use processor::{ProcessInfo, SoundExProcessor, StreamProgress};
pub use realtime::{RealtimeProcessor, RealtimeStats, RealtimeWorkerStats};
pub use stream::StreamBuffer;
