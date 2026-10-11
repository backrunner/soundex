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
    time_constraint: bool,
    warmup: bool,
) -> Result<JoinHandle<()>> {
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let startup_control = Arc::clone(&control);
    let worker = thread::Builder::new()
        .name("soundex-inference".into())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                control.qos_applied.store(apply_qos(), Ordering::Relaxed);
                if warmup {
                    let started = Instant::now();
                    let runs = processor.warm_up_inference()?;
                    control.warmup_runs.store(runs, Ordering::Relaxed);
                    control
                        .warmup_ns
                        .store(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                let scheduling = Scheduling::new(sample_rate, time_constraint);
                control
                    .realtime_applied
                    .store(scheduling.accepted(), Ordering::Relaxed);
                let period =
                    Duration::from_secs_f64(super::transport::HOP as f64 / sample_rate as f64);
                // Native RT preserves its quota; normal QoS uses bounded idle spin.
                let mut wait = Wait::new(period, !scheduling.accepted());
                let mut expected = Some(0);
                // Control-thread construction rendezvous, never an audio callback wait.
                let _ = ready_tx.send(());
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
                        let elapsed = started.saturating_duration_since(at).as_nanos() as u64;
                        control.queue_wait.record(elapsed);
                        control
                            .max_queue_wait_ns
                            .fetch_max(elapsed, Ordering::Relaxed);
                    }
                    let contiguous = expected == Some(packet.sequence);
                    if !contiguous {
                        processor.reset();
                    }
                    let state_warming = processor.state_history_warming();
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
                    control.processing.record(elapsed.as_nanos() as u64);
                    control.processed.fetch_add(1, Ordering::Relaxed);
                    if state_warming {
                        control.state_warmup.fetch_add(1, Ordering::Relaxed);
                    }
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
            // Failed initialization also releases the constructor and leaves dry fallback live.
            let _ = ready_tx.send(());
        })
        .map_err(|error| {
            SoundExError::ModelLoad(format!("cannot start inference worker: {error}"))
        })?;
    if warmup {
        if let Err(error) = ready_rx.recv_timeout(Duration::from_secs(10)) {
            startup_control.stop.store(true, Ordering::Release);
            return Err(SoundExError::ModelLoad(format!(
                "inference worker initialization did not complete: {error}"
            )));
        }
    }
    Ok(worker)
}

fn expired(sequence: u64, control: &Control) -> bool {
    sequence
        .checked_add(1)
        .is_none_or(|deadline| deadline <= control.presentation_sequence.load(Ordering::Acquire))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ort-backend")]
    #[test]
    fn sequence_gap_clears_model_memory_and_counts_dry_state_warmup() {
        let mut config = crate::SoundExConfig::with_model(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/stateful-probe.onnx"),
        );
        config.high_band_gain = crate::HighBandGain::Model;
        config.min_bandwidth_ratio = 1.0;
        let mut fresh = SoundExProcessor::new(config.clone()).unwrap();
        let processor = SoundExProcessor::new(config).unwrap();
        let (mut audio, worker_queues) = super::super::transport::queues();
        let control = Arc::clone(&audio.control);
        let handle = spawn(
            processor,
            worker_queues,
            Arc::clone(&control),
            128,
            44100,
            false,
            true,
        )
        .unwrap();
        for index in 0..140 {
            let sequence = if index < 70 { index } else { index + 10 };
            let mut packet = Packet {
                sequence,
                ..Packet::default()
            };
            for (n, value) in packet.samples[..128].iter_mut().enumerate() {
                *value = 0.1 * ((index * 128 + n as u64) as f32 * 0.13).sin();
            }
            assert!(audio.input.push(packet).is_ok());
            let deadline = Instant::now() + Duration::from_secs(5);
            let result = loop {
                if let Ok(result) = audio.output.pop() {
                    break result;
                }
                assert!(Instant::now() < deadline, "worker did not respond");
                thread::yield_now();
            };
            if index >= 70 {
                let mut expected = [0.0; 128];
                fresh
                    .process_frame(&packet.samples[..128], &mut expected)
                    .unwrap();
                assert_eq!(result.samples[..128], expected);
                assert_eq!(result.valid, index != 70);
            }
        }
        control.stop.store(true, Ordering::Release);
        handle.join().unwrap();
        assert_eq!(control.state_warmup.load(Ordering::Relaxed), 128);
        assert!(!control.failed.load(Ordering::Relaxed));
    }

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
