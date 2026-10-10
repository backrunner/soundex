//! Fixed-size packets and bounded, nonblocking audio/worker queues.

use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc,
};
use std::time::Instant;

use rtrb::{Consumer, Producer, RingBuffer};

pub(super) const HOP: usize = crate::realtime_policy::CALLBACK_FRAMES;
pub(super) const MAX_SAMPLES: usize = HOP * 2;
pub(super) const CAPACITY: usize = 2;

#[derive(Clone, Copy)]
pub(super) struct Packet {
    pub sequence: u64,
    pub samples: [f32; MAX_SAMPLES],
    pub valid: bool,
    pub submitted_at: Option<Instant>,
}

impl Default for Packet {
    fn default() -> Self {
        Self {
            sequence: 0,
            samples: [0.0; MAX_SAMPLES],
            valid: false,
            submitted_at: None,
        }
    }
}

#[derive(Default)]
pub(super) struct Control {
    pub stop: AtomicBool,
    pub failed: AtomicBool,
    pub presentation_sequence: AtomicU64,
    pub processed: AtomicU64,
    pub discarded_inputs: AtomicU64,
    pub discarded_outputs: AtomicU64,
    pub max_queue_wait_ns: AtomicU64,
    pub max_processing_ns: AtomicU64,
    pub queue_wait: super::latency::Histogram,
    pub processing: super::latency::Histogram,
    pub processing_overruns: AtomicU64,
    pub qos_applied: AtomicBool,
    pub realtime_applied: AtomicBool,
}

pub(super) struct AudioQueues {
    pub control: Arc<Control>,
    pub input: Producer<Packet>,
    pub output: Consumer<Packet>,
}

pub(super) struct WorkerQueues {
    pub input: Consumer<Packet>,
    pub output: Producer<Packet>,
}

pub(super) fn queues() -> (AudioQueues, WorkerQueues) {
    let (input_tx, input_rx) = RingBuffer::new(CAPACITY);
    let (output_tx, output_rx) = RingBuffer::new(CAPACITY);
    (
        AudioQueues {
            control: Arc::new(Control::default()),
            input: input_tx,
            output: output_rx,
        },
        WorkerQueues {
            input: input_rx,
            output: output_tx,
        },
    )
}
