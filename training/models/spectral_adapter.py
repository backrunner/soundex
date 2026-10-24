"""P48 single-pass frozen features with an independently scaled residual adapter.

SPDX-License-Identifier: Apache-2.0
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.restoration_head import RestorationHead

SUPPORTS = ("full", "high")


class SpectralAdapter(RestorationHead):
    """Same causal state/feature clock; the adapter never changes frozen features.

    Only the P38 native complex residual configuration is supported. The base
    arithmetic below deliberately preserves its operation order at zero adapter.
    This diagnostic model is not a production artifact contract extension.
    """

    def __init__(self, base_config: dict, support: str):
        super().__init__(**base_config)
        if (
            support not in SUPPORTS
            or self.arm != "complex"
            or self.budget_scope != "residual"
            or not self.generation_budget
            or self.phase_input
            or self.decoder_normalization
            or self.detector_condition != "native"
            or self.residual_coordinates != "input_relative"
        ):
            raise ValueError("P38 native residual complex base and registered support required")
        self.base_config = dict(base_config)
        self.support = support
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.adapter = nn.Linear(2 * self.width, 258)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def support_mask(self, condition: torch.Tensor) -> torch.Tensor:
        hz = torch.arange(129, dtype=condition.dtype, device=condition.device) * (48000 / 256)
        hz = condition[..., :1] * hz
        if self.support == "full":
            return torch.ones_like(hz)
        position = ((hz - 4000) / 4000).clamp(0, 1)
        return position.square() * (3 - 2 * position)

    def forward_details(
        self, original, context, condition, state, *, detach_budget_auxiliary=False
    ):
        magnitude = torch.pow(10.0, original[:, 0] / 20)
        latent, scale, state = self.backbone.encode(magnitude, context, condition, state)
        logits = self.backbone.decoder(latent)
        delta = 2 * torch.tanh(self.residual_slope * logits)
        normalized = magnitude / scale
        power = F.avg_pool1d(F.pad(normalized.square(), (4, 4), mode="replicate"), 9, 1)
        budget = F.softplus(self.budget_head(latent))
        generation = (self.budget_weights(condition) * budget[..., None, :]).sum(-1)
        units = scale * (power + 1).sqrt() * generation
        delta_real, delta_imag = delta[..., :129], delta[..., 129:] * self.interior
        real = magnitude + units * delta_real
        imag = units * delta_imag * self.interior

        # No learned budget factor in the new residual. Fixed input-only units
        # remain nonzero when high frequencies are absent. Both arms use exactly
        # the same coordinates, scale and parameter count; only support differs.
        adjustment = 2 * torch.tanh(self.residual_slope * self.adapter(latent.detach()))
        adapter_units = scale * (power + 1).sqrt() * self.support_mask(condition)
        real = real + adapter_units * adjustment[..., :129]
        imag = imag + adapter_units * adjustment[..., 129:] * self.interior

        epsilon = torch.full_like(scale, 1e-15)
        amplitude = (real.square() + imag.square() + epsilon.square()).sqrt()
        baseline = (magnitude.square() + epsilon.square()).sqrt()
        db = original[:, 0] + 20 * (
            amplitude.clamp_min(1e-10).log10() - baseline.clamp_min(1e-10).log10()
        )
        safe_real = torch.where((real == 0) & (imag == 0), epsilon, real)
        phase = original[:, 1] + torch.atan2(imag, safe_real)
        return torch.stack((db, phase), dim=1), state, budget


def from_base(checkpoint: dict, support: str) -> SpectralAdapter:
    model = SpectralAdapter(checkpoint["model_config"], support)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if set(missing) != {"adapter.weight", "adapter.bias"} or unexpected:
        raise ValueError("base checkpoint geometry mismatch")
    return model


def load_adapter(checkpoint: dict) -> SpectralAdapter:
    if checkpoint["format"] != "soundex-spectral-adapter-v1":
        raise ValueError("explicit adapter checkpoint required")
    model = SpectralAdapter(checkpoint["base_config"], checkpoint["support"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
