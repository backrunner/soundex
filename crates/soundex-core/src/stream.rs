//! Streaming audio buffer management.
//!
//! Provides a ring buffer for accumulating input samples until a full
//! FFT frame is available, and managing overlap-add output.

use crate::error::{Result, SoundExError};

/// A simple streaming buffer that accumulates samples and yields full frames.
pub struct StreamBuffer {
    buffer: Vec<f32>,
    frame_size: usize,
    hop_size: usize,
    write_pos: usize,
    start: usize,
}

impl StreamBuffer {
    /// Create a new stream buffer.
    ///
    /// # Arguments
    /// * `frame_size` - Full frame size (e.g., 1024 for FFT)
    /// * `hop_size` - Number of new samples per step (e.g., 512)
    ///
    /// # Panics
    /// Panics when either size is zero or `hop_size > frame_size`.
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        assert!(frame_size > 0, "frame_size must be greater than zero");
        assert!(
            hop_size > 0 && hop_size <= frame_size,
            "hop_size must be in 1..=frame_size"
        );
        Self {
            // Mirroring every physical sample at +frame_size keeps the logical
            // circular frame contiguous without shifting it on every hop.
            buffer: vec![0.0; frame_size * 2],
            frame_size,
            hop_size,
            write_pos: 0,
            start: 0,
        }
    }

    /// Create a buffer prefilled with `frame_size - hop_size` zeros.
    ///
    /// This represents the signal history before a stream starts and allows a
    /// causal overlap-add pipeline to accept exactly one hop immediately.
    pub fn with_latency_padding(frame_size: usize, hop_size: usize) -> Self {
        let mut buffer = Self::new(frame_size, hop_size);
        buffer.write_pos = frame_size - hop_size;
        buffer
    }

    /// Push samples into the buffer.
    ///
    /// Returns the number of samples consumed. An oversized write is rejected
    /// before any state is changed; samples are never silently dropped.
    pub fn push(&mut self, samples: &[f32]) -> Result<usize> {
        let capacity = self.frame_size - self.write_pos;
        if samples.len() > capacity {
            return Err(SoundExError::StreamOverflow {
                capacity,
                attempted: samples.len(),
            });
        }
        for (offset, &sample) in samples.iter().enumerate() {
            let physical = (self.start + self.write_pos + offset) % self.frame_size;
            self.buffer[physical] = sample;
            self.buffer[physical + self.frame_size] = sample;
        }
        self.write_pos += samples.len();
        Ok(samples.len())
    }

    /// Check if a full frame is available.
    pub fn is_frame_ready(&self) -> bool {
        self.write_pos >= self.frame_size
    }

    /// Number of complete frames available.
    pub fn available_frames(&self) -> usize {
        if self.write_pos >= self.frame_size {
            1
        } else {
            0
        }
    }

    /// Get the current frame (only valid if `is_frame_ready()`).
    pub fn current_frame(&self) -> &[f32] {
        &self.buffer[self.start..self.start + self.frame_size]
    }

    /// Advance the circular frame by one hop and clear its new tail.
    pub fn advance(&mut self) {
        self.start = (self.start + self.hop_size) % self.frame_size;
        self.write_pos = self.frame_size - self.hop_size;
        for logical in self.write_pos..self.frame_size {
            let physical = (self.start + logical) % self.frame_size;
            self.buffer[physical] = 0.0;
            self.buffer[physical + self.frame_size] = 0.0;
        }
    }

    /// Reset the buffer to initial state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.start = 0;
    }

    /// Reset and restore the zero history used by a causal STFT stream.
    pub fn reset_with_latency_padding(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = self.frame_size - self.hop_size;
        self.start = 0;
    }

    /// Current fill level in samples.
    pub fn fill_level(&self) -> usize {
        self.write_pos
    }

    /// Frame size configured for this buffer.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Hop size configured for this buffer.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
}

/// Hop iterator with causal zero history, a padded final input hop, and the
/// exact zero tail required to expose the STFT delay.
pub(crate) struct CausalHopIter<'a> {
    input: &'a [f32],
    hop_samples: usize,
    index: usize,
    total_hops: usize,
}

impl<'a> CausalHopIter<'a> {
    pub(crate) fn new(input: &'a [f32], channels: usize, fft_size: usize, hop_size: usize) -> Self {
        debug_assert!(channels > 0);
        debug_assert_eq!(input.len() % channels, 0);
        debug_assert_eq!(fft_size % hop_size, 0);
        let samples_per_channel = input.len() / channels;
        let signal_hops = samples_per_channel.div_ceil(hop_size);
        let tail_hops = if signal_hops == 0 {
            0
        } else {
            latency_hops(fft_size, hop_size)
        };
        Self {
            input,
            hop_samples: hop_size * channels,
            index: 0,
            total_hops: signal_hops + tail_hops,
        }
    }
}

impl Iterator for CausalHopIter<'_> {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total_hops {
            return None;
        }
        let start = self.index * self.hop_samples;
        let mut hop = vec![0.0; self.hop_samples];
        if start < self.input.len() {
            let end = (start + self.hop_samples).min(self.input.len());
            hop[..end - start].copy_from_slice(&self.input[start..end]);
        }
        self.index += 1;
        Some(hop)
    }
}

pub(crate) fn latency_hops(fft_size: usize, hop_size: usize) -> usize {
    (fft_size - hop_size) / hop_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_frame_ready() {
        let mut buf = StreamBuffer::new(8, 4);
        assert!(!buf.is_frame_ready());

        buf.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(!buf.is_frame_ready());

        buf.push(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(buf.is_frame_ready());
        assert_eq!(
            buf.current_frame(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_advance_shifts_buffer() {
        let mut buf = StreamBuffer::new(8, 4);
        buf.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        buf.advance();

        assert_eq!(buf.fill_level(), 4);
        // After advance: [5,6,7,8,0,0,0,0], write_pos = 4
        buf.push(&[9.0, 10.0, 11.0, 12.0]).unwrap();
        assert!(buf.is_frame_ready());
        assert_eq!(
            buf.current_frame(),
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_reset() {
        let mut buf = StreamBuffer::new(8, 4);
        buf.push(&[1.0; 8]).unwrap();
        buf.reset();
        assert_eq!(buf.fill_level(), 0);
        assert!(!buf.is_frame_ready());
    }

    #[test]
    fn test_latency_padding_accepts_one_hop() {
        let mut buf = StreamBuffer::with_latency_padding(8, 2);
        assert_eq!(buf.fill_level(), 6);
        buf.push(&[1.0, 2.0]).unwrap();
        assert!(buf.is_frame_ready());
        assert_eq!(
            buf.current_frame(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn oversized_push_is_rejected_without_mutation() {
        let mut buffer = StreamBuffer::new(8, 4);
        buffer.push(&[1.0; 6]).unwrap();

        let error = buffer.push(&[2.0; 3]).unwrap_err();

        assert!(matches!(error, SoundExError::StreamOverflow { .. }));
        assert_eq!(buffer.fill_level(), 6);
        assert_eq!(&buffer.current_frame()[..6], &[1.0; 6]);
    }

    #[test]
    fn causal_hops_include_startup_alignment_and_tail() {
        let input = [1.0, 2.0, 3.0];
        let hops: Vec<_> = CausalHopIter::new(&input, 1, 8, 2).collect();

        assert_eq!(hops.len(), 5);
        assert_eq!(hops[0], vec![1.0, 2.0]);
        assert_eq!(hops[1], vec![3.0, 0.0]);
        assert!(hops[2..].iter().all(|hop| hop == &vec![0.0, 0.0]));
    }
}
