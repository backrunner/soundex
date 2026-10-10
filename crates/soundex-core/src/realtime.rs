//! Nonblocking real-time output with fixed delay and an independent model worker.
//!
//! Create and destroy this object outside the audio callback. Call `process`
//! with matching interleaved input/output blocks; every valid call writes the
//! entire output. Model lateness/failure causes an aligned dry fallback, never
//! a wait, output-length change, or permanent callback error.

use std::sync::{atomic::Ordering, Arc};
use std::thread::JoinHandle;

use crate::{Result, SoundExConfig, SoundExError, SoundExProcessor};

mod latency;
mod safety;
mod state;
mod transport;
mod worker;
mod worker_wait;

use state::AudioState;
use transport::{AudioQueues, Control, HOP};

/// Cumulative callback health counters. Startup padding is excluded from misses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RealtimeStats {
    /// Output sample frames (one sample per channel), including startup padding.
    pub output_frames: u64,
    /// Valid, on-time worker hops accepted by the callback.
    pub accepted_hops: u64,
    /// Hops that missed their fixed presentation deadline; output still proceeds.
    pub deadline_misses: u64,
    /// Longest missed-deadline sequence, excluding startup padding.
    pub max_consecutive_deadline_misses: u64,
    /// Input hops dropped because the bounded worker queue was full.
    pub queue_overflows: u64,
    /// Results discarded because their timestamp was already obsolete.
    pub stale_results: u64,
    /// Invalid source samples replaced by a decaying held sample.
    pub invalid_samples: u64,
    /// Non-finite or excessive worker output rejected before mixing.
    pub rejected_results: u64,
    /// Worker exited after a processing error or panic; dry output remains live.
    pub worker_failed: bool,
}

/// Worker scheduling diagnostics, read outside the audio callback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RealtimeWorkerStats {
    /// Hops processed, including warmup and predictions that finished late.
    pub processed_hops: u64,
    /// Obsolete input hops skipped without spending inference time.
    pub discarded_input_hops: u64,
    /// Late predictions or predictions dropped from a full result queue.
    pub discarded_output_hops: u64,
    /// Maximum submission-to-worker-start wall time, in nanoseconds.
    pub max_queue_wait_ns: u64,
    /// Maximum worker processing wall time, in nanoseconds.
    pub max_processing_ns: u64,
    /// Approximate queue-wait p99 upper bound, with 25us histogram resolution.
    pub queue_wait_p99_upper_bound_ns: u64,
    /// Approximate processing p99 upper bound, with 25us histogram resolution.
    pub processing_p99_upper_bound_ns: u64,
    /// Processing calls exceeding one hop period (not presentation misses).
    pub processing_period_overruns: u64,
    /// macOS accepted the worker's USER_INITIATED QoS request.
    pub macos_qos_applied: bool,
    /// macOS accepted the worker's audio time-constraint request at startup.
    /// The OS may subsequently demote the thread; this is not a live query.
    pub macos_realtime_request_accepted: bool,
}

/// Audio callback adapter for 256-point FFT / 128-sample hop models.
///
/// Its fixed delay is 256 samples per channel: 128 STFT + 128 worker handoff.
/// Valid callback calls allocate no memory, lock no mutex, and never invoke ORT.
/// Arbitrary block sizes are supported without an extra conversion buffer.
pub struct RealtimeProcessor {
    state: AudioState,
    queues: AudioQueues,
    control: Arc<Control>,
    worker: Option<JoinHandle<()>>,
}

impl RealtimeProcessor {
    /// Load and validate the model and start the worker outside the callback.
    ///
    /// Returns an error for incompatible FFT/hop/sample rates, model failures,
    /// invalid configurations, or inability to create the worker thread.
    pub fn new(config: SoundExConfig) -> Result<Self> {
        if config.fft_size != 256
            || config.hop_size != HOP
            || !matches!(config.sample_rate, 44100 | 48000)
        {
            return Err(SoundExError::InvalidInput(
                "real-time adapter requires FFT256/hop128 at 44100 or 48000 Hz".into(),
            ));
        }
        let processor = SoundExProcessor::new(config.clone())?;
        let (queues, worker_queues) = transport::queues();
        let control = Arc::clone(&queues.control);
        let worker = worker::spawn(
            processor,
            worker_queues,
            Arc::clone(&control),
            HOP * config.channels as usize,
            config.sample_rate,
            config.worker_time_constraint,
        )?;
        Ok(Self {
            state: AudioState::new(config.channels as usize, config.limiter_ceiling),
            queues,
            control,
            worker: Some(worker),
        })
    }

    /// Process matching interleaved blocks without waiting for enhancement.
    ///
    /// Non-finite source samples are sanitized; model failures and missed
    /// deadlines remain successful calls. Only caller shape errors return an
    /// error (and clear output). No buffering delay depends on worker speed.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<RealtimeStats> {
        if input.len() != output.len() || !input.len().is_multiple_of(self.state.channels) {
            output.fill(0.0);
            return Err(SoundExError::InvalidInput(
                "real-time input/output lengths must match and contain complete channel frames"
                    .into(),
            ));
        }
        self.state.process(input, output, &mut self.queues);
        Ok(self.stats())
    }

    /// Fixed added delay in samples per channel, including worker handoff.
    pub fn latency_samples_per_channel(&self) -> usize {
        2 * HOP
    }

    /// Read callback counters and worker failure status without locking.
    pub fn stats(&self) -> RealtimeStats {
        RealtimeStats {
            worker_failed: self.control.failed.load(Ordering::Acquire),
            ..self.state.stats
        }
    }

    /// Read worker counters without locking. Not needed by the audio callback.
    /// Concurrent updates may produce a snapshot from slightly different instants.
    pub fn worker_stats(&self) -> RealtimeWorkerStats {
        RealtimeWorkerStats {
            processed_hops: self.control.processed.load(Ordering::Relaxed),
            discarded_input_hops: self.control.discarded_inputs.load(Ordering::Relaxed),
            discarded_output_hops: self.control.discarded_outputs.load(Ordering::Relaxed),
            max_queue_wait_ns: self.control.max_queue_wait_ns.load(Ordering::Relaxed),
            max_processing_ns: self.control.max_processing_ns.load(Ordering::Relaxed),
            queue_wait_p99_upper_bound_ns: self.control.queue_wait.p99_upper_bound_ns(),
            processing_p99_upper_bound_ns: self.control.processing.p99_upper_bound_ns(),
            processing_period_overruns: self.control.processing_overruns.load(Ordering::Relaxed),
            macos_qos_applied: self.control.qos_applied.load(Ordering::Relaxed),
            macos_realtime_request_accepted: self.control.realtime_applied.load(Ordering::Relaxed),
        }
    }

    /// Stop and join the worker on a control thread, never in an audio callback.
    /// An active inference finishes before shutdown returns.
    pub fn shutdown(mut self) {
        self.control.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for RealtimeProcessor {
    fn drop(&mut self) {
        self.control.stop.store(true, Ordering::Release);
        // Detach: normal Drop never waits for a slow or hung inference. Explicit
        // shutdown is available to control-thread owners that need to join.
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod stress_tests;
