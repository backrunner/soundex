//! Accelerated long-duration presentation test; does not replace paced/device stress.

use super::state::AudioState;
use super::transport::{queues, Packet, HOP, MAX_SAMPLES};

#[test]
#[ignore = "30 minutes of stereo audio; run with --release --ignored"]
fn thirty_minutes_preserve_timeline_through_faults_and_recovery() {
    let started = std::time::Instant::now();
    let (mut audio, mut worker) = queues();
    let mut state = AudioState::new(2, 0.95);
    let mut previous = [0.0; MAX_SAMPLES];
    let mut input = [0.0; MAX_SAMPLES];
    let mut output = [0.0; MAX_SAMPLES];
    let hops = 1800 * 48000 / HOP;
    let mut peak = 0.0_f32;
    let mut last = [0.0_f32; 2];
    let mut maximum_step = 0.0_f32;
    for hop in 0..hops {
        for frame in 0..HOP {
            let position = (hop * HOP + frame) as f32;
            input[frame * 2] = 0.2 * (position * 0.005).sin();
            input[frame * 2 + 1] = -input[frame * 2];
        }
        state.process(&input, &mut output, &mut audio);
        let packet = worker.input.pop().unwrap();
        // Regular outages and malformed predictions, with actual enhancement
        // correction on healthy hops rather than an always-dry fixture.
        if hop % 997 >= 20 {
            let mut samples = previous;
            for sample in &mut samples {
                *sample *= 1.3;
            }
            if hop % 1031 == 10 {
                samples[42] = f32::NAN;
            }
            worker
                .output
                .push(Packet {
                    sequence: packet.sequence,
                    samples,
                    valid: packet.sequence != 0,
                    ..Packet::default()
                })
                .unwrap();
        }
        previous = packet.samples;
        for values in output.as_chunks::<2>().0 {
            assert!(values
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 0.95));
            assert!((values[0] + values[1]).abs() < 1e-6);
            for channel in 0..2 {
                peak = peak.max(values[channel].abs());
                maximum_step = maximum_step.max((values[channel] - last[channel]).abs());
                last[channel] = values[channel];
            }
        }
    }
    assert_eq!(state.stats.output_frames, 1800 * 48000);
    assert!(state.stats.deadline_misses > 10000);
    assert!(state.stats.rejected_results > 100);
    assert_eq!(state.stats.queue_overflows, 0);
    // Known slow sine: whole-waveform slope must stay bounded even at changes
    // in presentation mode. High-frequency music needs different thresholds.
    assert!(maximum_step < 0.02, "maximum step {maximum_step}");
    println!(
        "virtual_audio_seconds=1800 wall_seconds={} peak={} max_step={} stats={:?}",
        started.elapsed().as_secs_f64(),
        peak,
        maximum_step,
        state.stats
    );
}
