//! Deterministic presentation-deadline and safety regression tests.

use super::safety::SafetyMixer;
use super::state::AudioState;
use super::transport::{queues, AudioQueues, Packet, WorkerQueues, HOP, MAX_SAMPLES};

struct Harness {
    state: AudioState,
    audio: AudioQueues,
    worker: WorkerQueues,
    previous: [f32; MAX_SAMPLES],
}

impl Harness {
    fn new() -> Self {
        let (audio, worker) = queues();
        Self {
            state: AudioState::new(2, 0.95),
            audio,
            worker,
            previous: [0.0; MAX_SAMPLES],
        }
    }

    fn hop(&mut self, input: &[f32], available: bool) -> Vec<f32> {
        let mut output = vec![f32::NAN; input.len()];
        self.state.process(input, &mut output, &mut self.audio);
        let packet = self.worker.input.pop().unwrap();
        if available {
            self.worker
                .output
                .push(Packet {
                    sequence: packet.sequence,
                    samples: self.previous,
                    valid: packet.sequence != 0,
                    ..Packet::default()
                })
                .unwrap();
        }
        self.previous = packet.samples;
        output
    }
}

#[test]
fn fixed_delay_and_output_length_do_not_depend_on_worker_availability() {
    let input: Vec<f32> = (0..40 * HOP)
        .flat_map(|n| {
            let left = 0.25 * (n as f32 * 0.03).sin();
            [left, -left * 0.5]
        })
        .collect();
    let mut harness = Harness::new();
    let mut output = Vec::new();
    for (n, hop) in input.as_chunks::<MAX_SAMPLES>().0.iter().enumerate() {
        // Contiguous loss, isolated miss, recovery, and stereo alignment.
        output.extend(harness.hop(hop, !(10..20).contains(&n) && n != 30));
    }
    assert_eq!(output.len(), input.len());
    assert_eq!(&output[..2 * MAX_SAMPLES], &[0.0; 2 * MAX_SAMPLES]);
    assert_eq!(
        &output[2 * MAX_SAMPLES..],
        &input[..input.len() - 2 * MAX_SAMPLES]
    );
    assert_eq!(harness.state.stats.deadline_misses, 11);
    assert_eq!(harness.state.stats.max_consecutive_deadline_misses, 10);
}

#[test]
fn stalled_worker_queue_is_bounded_and_dry_output_remains_exact() {
    let mut harness = Harness::new();
    let input = [0.125; MAX_SAMPLES];
    for n in 0..1000 {
        let mut output = [f32::NAN; MAX_SAMPLES];
        harness
            .state
            .process(&input, &mut output, &mut harness.audio);
        assert_eq!(output, [if n < 2 { 0.0 } else { 0.125 }; MAX_SAMPLES]);
    }
    assert_eq!(harness.state.stats.output_frames, 1000 * HOP as u64);
    assert_eq!(harness.state.stats.queue_overflows, 998);
    assert_eq!(harness.state.stats.deadline_misses, 998);
    assert_eq!(harness.state.stats.max_consecutive_deadline_misses, 998);
}

#[test]
fn stale_and_invalid_model_packets_are_rejected() {
    let mut harness = Harness::new();
    for n in 0..5 {
        let input = [0.1; MAX_SAMPLES];
        let mut output = [0.0; MAX_SAMPLES];
        harness
            .state
            .process(&input, &mut output, &mut harness.audio);
        let packet = harness.worker.input.pop().unwrap();
        let bad = match n {
            2 => f32::NAN,
            3 => f32::INFINITY,
            _ => 100.0,
        };
        if n >= 1 {
            harness
                .worker
                .output
                .push(Packet {
                    sequence: packet.sequence,
                    samples: [bad; MAX_SAMPLES],
                    valid: true,
                    ..Packet::default()
                })
                .unwrap();
        }
        assert!(output
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 0.95));
    }
    assert_eq!(harness.state.stats.rejected_results, 3);
    harness
        .worker
        .output
        .push(Packet {
            sequence: 0,
            valid: true,
            ..Packet::default()
        })
        .unwrap();
    let mut output = [0.0; MAX_SAMPLES];
    harness
        .state
        .process(&[0.1; MAX_SAMPLES], &mut output, &mut harness.audio);
    assert_eq!(harness.state.stats.stale_results, 1);
}

#[test]
fn arbitrary_callback_blocks_have_identical_timeline_and_no_worker_wait() {
    let input: Vec<f32> = (0..10000)
        .flat_map(|n| [n as f32 / 20000.0, -0.2])
        .collect();
    let mut whole = Harness::new();
    let mut expected = vec![0.0; input.len()];
    whole.state.process(&input, &mut expected, &mut whole.audio);
    let mut split = Harness::new();
    let mut actual = vec![0.0; input.len()];
    let mut offset = 0;
    for frames in [1, 17, 31, 127, 128, 257, 1024].into_iter().cycle() {
        let end = (offset + frames * 2).min(input.len());
        split.state.process(
            &input[offset..end],
            &mut actual[offset..end],
            &mut split.audio,
        );
        offset = end;
        if offset == input.len() {
            break;
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(split.state.stats, whole.state.stats);
}

#[test]
fn fallback_and_recovery_preserve_correction_at_the_boundary() {
    let mut mixer = SafetyMixer::new(0.95);
    let mut output = [0.0; 2];
    mixer.begin_hop(true);
    for _ in 0..=HOP {
        mixer.frame(&[0.1, -0.1], &[0.3, -0.3], &mut output);
    }
    assert!((output[0] - 0.3).abs() < 1e-6);
    let previous = output;
    mixer.begin_hop(false);
    mixer.frame(&[0.1, -0.1], &[f32::NAN; 2], &mut output);
    assert_eq!(output, previous);
    let mut max_step = 0.0_f32;
    for _ in 0..HOP {
        let previous = output;
        mixer.frame(&[0.1, -0.1], &[f32::NAN; 2], &mut output);
        max_step = max_step.max((output[0] - previous[0]).abs());
        assert!((output[0] + output[1]).abs() < 1e-7);
    }
    assert_eq!(output, [0.1, -0.1]);
    assert!(max_step < 0.0025, "fade step {max_step}");
    mixer.begin_hop(true);
    mixer.frame(&[0.1, -0.1], &[-0.3, 0.3], &mut output);
    assert_eq!(output, [0.1, -0.1]);
    for _ in 0..HOP {
        mixer.frame(&[0.1, -0.1], &[-0.3, 0.3], &mut output);
    }
    assert!((output[0] + 0.3).abs() < 1e-6);
}

#[test]
fn malformed_samples_cannot_contaminate_dry_fallback_or_limiter() {
    let mut harness = Harness::new();
    let mut input = [f32::MAX; MAX_SAMPLES];
    input[10] = f32::NAN;
    input[11] = f32::INFINITY;
    input[12] = f32::NEG_INFINITY;
    for _ in 0..20 {
        let output = harness.hop(&input, false);
        assert!(output
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 0.95));
    }
    assert_eq!(harness.state.stats.invalid_samples, 60);
}
