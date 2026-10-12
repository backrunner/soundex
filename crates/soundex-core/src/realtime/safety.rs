//! Channel-linked, sample-domain recovery fades and final output protection.

use soundex_dsp::limiter::{Limiter, LimiterMode};

use super::transport::HOP;

pub(super) struct SafetyMixer {
    limiter: Limiter,
    mode: LimiterMode,
    active: bool,
    position: usize,
    anchor: [f32; 2],
    last_correction: [f32; 2],
}

impl SafetyMixer {
    pub fn new(ceiling: f32, mode: LimiterMode) -> Self {
        Self {
            limiter: Limiter::with_mode(ceiling, 10.0, mode),
            mode,
            active: false,
            position: HOP,
            anchor: [0.0; 2],
            last_correction: [0.0; 2],
        }
    }

    pub fn begin_hop(&mut self, active: bool) {
        if active != self.active {
            self.anchor = self.last_correction;
            self.position = 0;
            self.active = active;
        }
    }

    pub fn frame(&mut self, dry: &[f32], wet: &[f32], output: &mut [f32]) {
        // First sample preserves the previous correction; the fade takes one
        // hop. Missing wet samples are never read or multiplied by zero.
        let progress = self.position.min(HOP) as f32 / HOP as f32;
        let weight = 0.5 - 0.5 * (std::f32::consts::PI * progress).cos();
        for channel in 0..dry.len() {
            // Smooth mode protects dry once and accepts the worker's already
            // limited wet signal. Recompressing the mix would color wet twice.
            let dry_sample = if self.mode == LimiterMode::SmoothKnee {
                self.limiter.process_sample(dry[channel])
            } else {
                dry[channel]
            };
            let target = if self.active {
                wet[channel] - dry_sample
            } else {
                0.0
            };
            let correction = self.anchor[channel] * (1.0 - weight) + target * weight;
            self.last_correction[channel] = correction;
            let mixed = dry_sample + correction;
            output[channel] = if self.mode == LimiterMode::SmoothKnee {
                // A held recovery correction can exceed the ceiling when dry
                // changes abruptly; keep the final safety guard on every frame.
                self.limiter.clamp_sample(mixed)
            } else {
                self.limiter.process_sample(mixed)
            };
        }
        self.position = (self.position + 1).min(HOP);
    }
}
