"""P74 small joint band representation with causal phase-relation observations.

Same-size magnitude-only observations provide the matched information control.
No DSP teacher, future context or added output delay. SPDX-License-Identifier: Apache-2.0
"""

import torch
from torch import nn

from models.band_phase import BandPhaseCorrection
from models.band_tcn import BandTCN

OBSERVATIONS = ("magnitude", "phase")


class PhaseTemporalBand(BandTCN):
    def __init__(self, observation):
        if observation not in OBSERVATIONS:
            raise ValueError("unknown temporal observation")
        super().__init__(token_width=24)
        self.observation = observation
        self.temporal_state_size = self.state_size
        self.state_size += 258
        self.relation_projections = nn.ModuleList(
            nn.Linear(4 * (end - start), 24, bias=False) for start, end in self.backbone.slices[0]
        )
        for projection in self.relation_projections:
            nn.init.zeros_(projection.weight)

    def initial_state(self, batch, *, device=None):
        return self.backbone.readout.weight.new_zeros(batch, self.state_size, device=device)

    def relations(self, original, magnitude, scale, history):
        carrier = torch.log1p((magnitude - 1e-10).clamp_min(0) / scale)
        if self.observation == "phase":
            real, imag = carrier * original[:, 1].cos(), carrier * original[:, 1].sin()
        else:
            # Identical state/parameter geometry, exactly no phase information.
            real, imag = carrier, torch.zeros_like(carrier)
        previous_real = torch.cat((history[:, :129, None].transpose(1, 2), real[:, :-1]), 1)
        previous_imag = torch.cat((history[:, 129:, None].transpose(1, 2), imag[:, :-1]), 1)
        left_real = torch.cat((torch.zeros_like(real[..., :1]), real[..., :-1]), -1)
        left_imag = torch.cat((torch.zeros_like(imag[..., :1]), imag[..., :-1]), -1)

        def product(other_real, other_imag):
            denominator = 1 + carrier * (other_real.square() + other_imag.square() + 1e-20).sqrt()
            cosine = (real * other_real + imag * other_imag) / denominator
            sine = (imag * other_real - real * other_imag) / denominator
            return (cosine, sine) if self.observation == "phase" else (cosine, cosine)

        observations = (*product(previous_real, previous_imag), *product(left_real, left_imag))
        return observations, torch.cat((real[:, -1], imag[:, -1]), -1)

    def encode_observations(self, original, magnitude, context, condition, state):
        scale = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
        observations, next_history = self.relations(
            original, magnitude, scale, state[:, self.temporal_state_size :]
        )
        tokens = [
            projection(torch.cat([v[..., start:end] for v in observations], -1))
            for projection, (start, end) in zip(
                self.relation_projections, self.backbone.slices[0], strict=True
            )
        ]
        latent, scale, updated = self.backbone.encode(
            magnitude,
            context,
            condition,
            state[:, : self.temporal_state_size],
            token_residual=torch.stack(tokens, 2),
        )
        return latent, scale, torch.cat((updated, next_history), -1)


def phase_temporal_model(observation, seed=74, *, nonzero=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BandPhaseCorrection(
            PhaseTemporalBand(observation),
            seed=seed + 1,
            phase_support="native",
            phase_observation="adjacent",
            train_base=True,
        ).eval()
        model.temporal_observation = observation
        model.study = "p74-small-joint-phase-temporal"
        if nonzero:
            torch.manual_seed(seed + 2)
            with torch.no_grad():
                for layer in (
                    model.base.backbone.decoder[-1],
                    model.base.adapter,
                    model.base.budget_head,
                    model.phase_head[-1],
                    *model.base.relation_projections,
                ):
                    layer.weight.normal_(0, 0.03)
    return model
