"""P65 frozen band magnitude with a causal phase-only readout.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch
from torch import nn


class BandPhaseCorrection(nn.Module):
    """Keep every base tensor frozen; append one physical-channel phase history hop.

    Exact base dB output is preserved, not the reanalyzed OLA magnitude. Rotation
    can change overlap-add interference and must be reviewed after native DSP.
    """

    def __init__(self, base, *, seed=65, hidden=32):
        super().__init__()
        self.study = "p65-frozen-band-phase-calibration"
        self.base = base.eval().requires_grad_(False)
        for name in (
            "arm",
            "width",
            "residual_slope",
            "generation_budget",
            "budget_scope",
            "decoder_normalization",
            "phase_input",
            "residual_coordinates",
            "detector_condition",
            "support",
        ):
            setattr(self, name, getattr(base, name))
        self.base_state_size = base.state_size
        self.state_size = base.state_size + 258
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.phase_head = nn.Sequential(
                nn.Linear(5 * 129 + 3, hidden), nn.Tanh(), nn.Linear(hidden, 129)
            )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

    def initial_state(self, batch, *, device=None):
        return self.phase_head[0].weight.new_zeros(batch, self.state_size, device=device)

    def forward_details(
        self, original, context, condition, state, *, detach_budget_auxiliary=False
    ):
        with torch.no_grad():
            prediction, updated, budget = self.base.forward_details(
                original, context, condition, state[:, : self.base_state_size]
            )
        magnitude = torch.pow(10.0, original[:, 0] / 20)
        scale = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
        # Exact digital silence carries no phase; no learned weak-bin cutoff.
        carrier = torch.log1p((magnitude - 1e-10).clamp_min(0) / scale)
        real, imag = carrier * original[:, 1].cos(), carrier * original[:, 1].sin()
        previous_real = torch.cat(
            (
                state[:, self.base_state_size : self.base_state_size + 129, None].transpose(1, 2),
                real[:, :-1],
            ),
            1,
        )
        previous_imag = torch.cat(
            (state[:, self.base_state_size + 129 :, None].transpose(1, 2), imag[:, :-1]), 1
        )
        product_scale = (
            1 + carrier * (previous_real.square() + previous_imag.square() + 1e-20).sqrt()
        )
        cosine = (real * previous_real + imag * previous_imag) / product_scale
        sine = (imag * previous_real - real * previous_imag) / product_scale
        relative = original[:, 1] - prediction[:, 1]
        features = torch.cat(
            (carrier, carrier * relative.cos(), carrier * relative.sin(), cosine, sine, condition),
            -1,
        )
        active = (carrier.sum(-1, keepdim=True) > 0).to(carrier.dtype)
        rotation = math.pi * torch.tanh(self.phase_head(features) / 64)
        rotation = rotation * self.base.support_mask(condition) * self.base.interior * active
        result = torch.stack((prediction[:, 0], prediction[:, 1] + rotation), 1)
        next_state = torch.cat((updated, real[:, -1], imag[:, -1]), -1)
        return result, next_state, budget

    def forward(self, original, context, condition, state):
        result, next_state, _ = self.forward_details(original, context, condition, state)
        return result, next_state
