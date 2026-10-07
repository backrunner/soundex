//! Independent inference worker; gaps reset history and invalidate warmup output.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{atomic::Ordering, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::transport::{Control, Packet, WorkerQueues, CAPACITY};
use super::worker_wait::{apply_qos, Scheduling, Wait};
use crate::{Result, SoundExError, SoundExProcessor};

pub(super) fn spawn(
    mut processor: SoundExProcessor,
    mut queues: WorkerQueues,
    control: Arc<Control>,
    samples: usize,
    sample_rate: u32,
) -> Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("soundex-inference".into())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                control.qos_applied.store(apply_qos(), Ordering::Relaxed);
                let scheduling = Scheduling::new(sample_rate);
                control
                    .realtime_applied
                    .store(scheduling.accepted(), Ordering::Relaxed);
                let period =
                    Duration::from_secs_f64(super::transport::HOP as f64 / sample_rate as f64);
                // Real-time computation quota belongs to inference, not idle spinning.
                let mut wait = Wait::new(period, !scheduling.accepted());
                let mut expected = Some(0);
                while !control.stop.load(Ordering::Acquire) {
                    let Ok(mut packet) = queues.input.pop() else {
                        wait.idle();
                        continue;
                    };
                    // Bounded backlog: process the latest hop, not accumulated delay.
                    for _ in 1..CAPACITY {
                        if let Ok(newer) = queues.input.pop() {
                            control.discarded_inputs.fetch_add(1, Ordering::Relaxed);
                            packet = newer;
                        }
                    }
                    wait.submitted(packet.submitted_at);
                    if expired(packet.sequence, &control) {
                        control.discarded_inputs.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let started = Instant::now();
                    if let Some(at) = packet.submitted_at {
                        control.max_queue_wait_ns.fetch_max(
                            started.saturating_duration_since(at).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                    }
                    let contiguous = expected == Some(packet.sequence);
                    if !contiguous {
                        processor.reset();
                    }
                    let mut output = Packet {
                        sequence: packet.sequence,
                        valid: contiguous && packet.sequence != 0,
                        ..Packet::default()
                    };
                    processor.process_frame(
                        &packet.samples[..samples],
                        &mut output.samples[..samples],
                    )?;
                    let elapsed = started.elapsed();
                    control.processed.fetch_add(1, Ordering::Relaxed);
                    control
                        .max_processing_ns
                        .fetch_max(elapsed.as_nanos() as u64, Ordering::Relaxed);
                    if elapsed >= period {
                        control.processing_overruns.fetch_add(1, Ordering::Relaxed);
                    }
                    expected = packet.sequence.checked_add(1);
                    // Full result queue means the audio thread has already moved on.
                    // Never wait for it, and never grow the queue.
                    if expired(packet.sequence, &control) || queues.output.push(output).is_err() {
                        control.discarded_outputs.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Ok::<(), SoundExError>(())
            }));
            if !matches!(result, Ok(Ok(()))) {
                control.failed.store(true, Ordering::Release);
            }
        })
        .map_err(|error| SoundExError::ModelLoad(format!("cannot start inference worker: {error}")))
}

fn expired(sequence: u64, control: &Control) -> bool {
    sequence
        .checked_add(1)
        .is_none_or(|deadline| deadline <= control.presentation_sequence.load(Ordering::Acquire))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_is_presentation_hop_not_age_or_queue_length() {
        let control = Control::default();
        assert!(!expired(0, &control));
        control.presentation_sequence.store(2, Ordering::Release);
        assert!(expired(0, &control));
        assert!(expired(1, &control));
        assert!(!expired(2, &control));
        assert!(!expired(3, &control));
        assert!(expired(u64::MAX, &control));
    }
}
