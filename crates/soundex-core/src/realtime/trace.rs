//! Opt-in fixed-capacity event slots. No allocation, locks or serialization on callback.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub(super) const CALLBACK_BEGIN: usize = 0;
pub(super) const CALLBACK_DECISION: usize = 1;
pub(super) const CALLBACK_READY: usize = 2;
pub(super) const WET_ENABLED: usize = 3;
pub(super) const SUBMITTED: usize = 4;
pub(super) const ENQUEUED: usize = 5;
pub(super) const STARTED: usize = 6;
pub(super) const FINISHED: usize = 7;
pub(super) const PUBLICATION: usize = 8;
pub(super) const WORKER_OUTCOME: usize = 9;
pub(super) const HISTORY_WARMING: usize = 10;
pub(super) const HISTORY_RESET: usize = 11;
pub(super) const CONSUMED: usize = 12;
pub(super) const CONSUMED_OUTCOME: usize = 13;
pub(super) const TAKEN: usize = 14;
pub(super) const PRESENTATION_AT_FINISH: usize = 15;
const FIELDS: usize = 16;

pub(super) struct Trace {
    epoch: Instant,
    slots: Box<[[AtomicU64; FIELDS]]>,
    ignored_writes: AtomicU64,
}

impl Trace {
    pub fn new(capacity: usize, epoch: Instant) -> crate::Result<Self> {
        if !(1..=65536).contains(&capacity) {
            return Err(crate::SoundExError::InvalidInput(
                "trace capacity must be 1..65536 hops; allocate before playback".into(),
            ));
        }
        Ok(Self {
            epoch,
            slots: (0..capacity)
                .map(|_| std::array::from_fn(|_| AtomicU64::new(0)))
                .collect(),
            ignored_writes: AtomicU64::new(0),
        })
    }

    pub fn offset(&self, time: Instant) -> u64 {
        time.saturating_duration_since(self.epoch).as_nanos() as u64 + 1
    }

    pub fn set(&self, sequence: u64, field: usize, value: u64) {
        if let Some(slot) = self.slots.get(sequence as usize) {
            slot[field].store(value, Ordering::Relaxed);
        } else {
            self.ignored_writes.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn stamp(&self, sequence: u64, field: usize) {
        self.set(sequence, field, self.offset(Instant::now()));
    }

    /// Only called after the worker is joined; no concurrently changing fields.
    pub fn snapshot(&self) -> serde_json::Value {
        const NAMES: [&str; FIELDS] = [
            "callback_begin_ns",
            "callback_decision_ns",
            "callback_ready",
            "wet_enabled",
            "submitted_ns",
            "enqueued",
            "started_ns",
            "finished_ns",
            "terminal_ns",
            "worker_outcome",
            "history_warming",
            "history_reset",
            "consumed_ns",
            "consumed_outcome",
            "taken_ns",
            "presentation_at_finish",
        ];
        let events: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .map(|(sequence, slot)| {
                let mut row = serde_json::Map::new();
                row.insert("sequence".into(), sequence.into());
                for (name, value) in NAMES.iter().zip(slot) {
                    row.insert((*name).into(), value.load(Ordering::Relaxed).into());
                }
                serde_json::Value::Object(row)
            })
            .collect();
        serde_json::json!({
            "format": "soundex-worker-events-v1", "capacity": self.slots.len(),
            "ignored_writes": self.ignored_writes.load(Ordering::Relaxed),
            "clock": "Monotonic nanoseconds from constructor epoch plus1; zero means absent. Worker publication timestamp is AFTER the push/expiry decision, not an atomic enqueue linearization time.",
            "boolean_codes": "0 absent,1 true,2 false",
            "worker_outcome_codes": "1 queued,2 superseded input,3 expired input,4 expired output,5 output queue full",
            "consumed_outcome_codes": "1 accepted,2 stale,3 invalid continuity,4 nonfinite/excessive",
            "events": events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::{
        state::AudioState,
        transport::{with_control, Control, Packet},
        worker::{self, Processor},
    };
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn queue_rejection_and_dry_presentation_have_distinct_events() {
        let (mut queues, _worker) = with_control(Control {
            trace: Some(Trace::new(4, Instant::now()).unwrap()),
            ..Control::default()
        });
        let mut state = AudioState::new(1, 0.95, crate::LimiterMode::SmoothKnee);
        let mut out = [0.0; 128];
        for _ in 0..4 {
            state.process(&[0.1; 128], &mut out, &mut queues);
        }
        assert_eq!(out, [0.1; 128]);
        let snapshot = queues.control.trace.as_ref().unwrap().snapshot();
        assert_eq!(snapshot["events"][0]["enqueued"], 1);
        assert_eq!(snapshot["events"][2]["enqueued"], 2);
        assert_eq!(snapshot["events"][2]["callback_ready"], 2);
        assert_eq!(state.stats.queue_overflows, 2);
        assert_eq!(state.stats.deadline_misses, 2);
    }

    struct Delayed {
        began: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        first: bool,
    }
    impl Processor for Delayed {
        fn warm_up_inference(&mut self) -> crate::Result<u64> {
            Ok(0)
        }
        fn reset(&mut self) {}
        fn state_history_warming(&self) -> bool {
            false
        }
        fn process_frame(&mut self, _: &[f32], output: &mut [f32]) -> crate::Result<()> {
            if self.first {
                self.first = false;
                self.began.send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            output.fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn actual_worker_trace_distinguishes_late_completion_and_history_reset() {
        let (mut queues, worker_queues) = with_control(Control {
            trace: Some(Trace::new(3, Instant::now()).unwrap()),
            ..Control::default()
        });
        let control = Arc::clone(&queues.control);
        let (began_tx, began_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let handle = worker::spawn(
            Delayed {
                began: began_tx,
                release: release_rx,
                first: true,
            },
            worker_queues,
            Arc::clone(&control),
            128,
            48000,
            false,
            true,
        )
        .unwrap();
        queues
            .input
            .push(Packet {
                submitted_at: Some(Instant::now()),
                ..Packet::default()
            })
            .unwrap();
        began_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        control.presentation_sequence.store(1, Ordering::Release);
        release_tx.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while control.discarded_outputs.load(Ordering::Relaxed) == 0 {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        queues
            .input
            .push(Packet {
                sequence: 2,
                submitted_at: Some(Instant::now()),
                ..Packet::default()
            })
            .unwrap();
        loop {
            if let Ok(packet) = queues.output.pop() {
                assert!(!packet.valid);
                break;
            }
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        control.stop.store(true, Ordering::Release);
        handle.join().unwrap();
        let snapshot = control.trace.as_ref().unwrap().snapshot();
        assert_eq!(snapshot["events"][0]["worker_outcome"], 4);
        assert_eq!(snapshot["events"][2]["worker_outcome"], 1);
        assert_eq!(snapshot["events"][2]["history_reset"], 1);
        assert_eq!(snapshot["events"][0]["presentation_at_finish"], 1);
    }
}
