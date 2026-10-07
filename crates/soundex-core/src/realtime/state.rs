//! Fixed-delay callback state, timeline matching, and arbitrary block assembly.

use super::safety::SafetyMixer;
use super::transport::{AudioQueues, Packet, CAPACITY, HOP, MAX_SAMPLES};
use super::RealtimeStats;
use std::{sync::atomic::Ordering, time::Instant};

pub(super) struct AudioState {
    pub channels: usize,
    pub stats: RealtimeStats,
    sequence: u64,
    position: usize,
    delay_position: usize,
    delay: [f32; MAX_SAMPLES * 2],
    input: Packet,
    wet: Packet,
    last_input: [f32; 2],
    healthy_hops: usize,
    consecutive_misses: u64,
    mixer: SafetyMixer,
}

impl AudioState {
    pub fn new(channels: usize, ceiling: f32) -> Self {
        Self {
            channels,
            stats: RealtimeStats::default(),
            sequence: 0,
            position: 0,
            delay_position: 0,
            delay: [0.0; MAX_SAMPLES * 2],
            input: Packet::default(),
            wet: Packet::default(),
            last_input: [0.0; 2],
            healthy_hops: 0,
            consecutive_misses: 0,
            mixer: SafetyMixer::new(ceiling),
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32], queues: &mut AudioQueues) {
        for (source, destination) in input
            .chunks_exact(self.channels)
            .zip(output.chunks_exact_mut(self.channels))
        {
            if self.position == 0 {
                self.begin_hop(queues);
            }
            let mut dry = [0.0; 2];
            for channel in 0..self.channels {
                let value = if source[channel].is_finite() {
                    source[channel].clamp(-1.0, 1.0)
                } else {
                    self.stats.invalid_samples += 1;
                    // Hold rather than introduce a one-sample zero spike.
                    self.last_input[channel] * 0.99
                };
                self.last_input[channel] = value;
                self.input.samples[self.position * self.channels + channel] = value;
                dry[channel] = self.delay[self.delay_position + channel];
                self.delay[self.delay_position + channel] = value;
            }
            let offset = self.position * self.channels;
            self.mixer.frame(
                &dry[..self.channels],
                &self.wet.samples[offset..offset + self.channels],
                destination,
            );
            self.delay_position = (self.delay_position + self.channels) % (2 * HOP * self.channels);
            self.position += 1;
            self.stats.output_frames += 1;
            if self.position == HOP {
                self.input.sequence = self.sequence;
                self.input.submitted_at = Some(Instant::now());
                if queues.input.push(self.input).is_err() {
                    self.stats.queue_overflows += 1;
                }
                self.position = 0;
                self.sequence += 1;
            }
        }
    }

    fn begin_hop(&mut self, queues: &mut AudioQueues) {
        queues
            .control
            .presentation_sequence
            .store(self.sequence, Ordering::Release);
        let mut ready = false;
        // Snapshot bounded capacity: a producer cannot extend this loop.
        for _ in 0..CAPACITY {
            let Ok(packet) = queues.output.pop() else {
                break;
            };
            if self.sequence.checked_sub(1) != Some(packet.sequence) {
                self.stats.stale_results += 1;
                continue;
            }
            if !packet.valid {
                continue;
            }
            let values = &packet.samples[..HOP * self.channels];
            if values
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.000_001)
            {
                self.stats.rejected_results += 1;
                continue;
            }
            self.wet = packet;
            ready = true;
        }
        if ready {
            self.consecutive_misses = 0;
            self.stats.accepted_hops += 1;
            self.healthy_hops = (self.healthy_hops + 1).min(2);
        } else {
            self.healthy_hops = 0;
            if self.sequence >= 2 {
                self.stats.deadline_misses += 1;
                self.consecutive_misses += 1;
                self.stats.max_consecutive_deadline_misses = self
                    .stats
                    .max_consecutive_deadline_misses
                    .max(self.consecutive_misses);
            }
        }
        let dry_hop = &self.delay[self.delay_position..self.delay_position + HOP * self.channels];
        let silent = dry_hop.iter().all(|value| value.abs() <= 1e-10);
        self.mixer
            .begin_hop(ready && self.healthy_hops >= 2 && !silent);
    }
}
