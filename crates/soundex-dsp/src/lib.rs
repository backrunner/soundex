//! # soundex-dsp
//!
//! DSP algorithms for SoundEx audio enhancement.
//!
//! This crate provides pure signal processing primitives with no I/O dependencies:
//! - STFT / iSTFT analysis and synthesis
//! - Bandwidth detection (gatekeeping)
//! - Loudness matching
//! - Phase smoothing
//! - Crossover blending
//! - Limiter protection

pub mod bandwidth;
pub mod crossover;
pub mod extension;
pub mod limiter;
pub mod loudness;
pub mod phase;
pub mod stft;
pub mod window;
