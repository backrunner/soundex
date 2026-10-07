//! Channel-linked, sample-domain recovery fades and final output protection.

use soundex_dsp::limiter::Limiter;

use super::transport::HOP;

pub(super) struct SafetyMixer {
    limiter: Limiter,
    active: bool,
    position: usize,
    anchor: [f32; 2],
    last_correction: [f32; 2],
}

impl SafetyMixer {
    pub fn new(ceiling: f32) -> Self {
        Self {
            limiter: Limiter::new(ceiling, 10.0),
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
            let target = if self.active {
                wet[channel] - dry[channel]
            } else {
                0.0
            };
            let correction = self.anchor[channel] * (1.0 - weight) + target * weight;
            self.last_correction[channel] = correction;
            output[channel] = self.limiter.process_sample(dry[channel] + correction);
        }
        self.position = (self.position + 1).min(HOP);
    }
}
