"""P56 same-function trainability controls; no added inference parameters.

SPDX-License-Identifier: Apache-2.0
"""

import torch
from torch.nn import functional as F

from models.spectral_adapter import SpectralAdapter

SCOPES = ("adapter", "decoder", "full")


class TrainableAdapter(SpectralAdapter):
    """Keep P54 arithmetic/state; connect the adapter to live features when unfrozen.

    Warmup remains detached in predict_budget. Use eval mode throughout; no
    statistics updates. Full reconstruction training includes the budget head,
    but no historical budget-teacher auxiliary is introduced.
    """

    def __init__(self, base_config, support, scope):
        super().__init__(base_config, support)
        if scope not in SCOPES:
            raise ValueError("unknown trainability scope")
        self.scope = scope
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                scope == "full"
                or name.startswith("adapter.")
                or (scope == "decoder" and name.startswith("backbone.decoder."))
            )

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
        adjustment = 2 * torch.tanh(self.residual_slope * self.adapter(latent))
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


def load_trainable(checkpoint, scope):
    if checkpoint["format"] != "soundex-spectral-adapter-v1":
        raise ValueError("P54 adapter checkpoint required")
    model = TrainableAdapter(checkpoint["base_config"], checkpoint["support"], scope)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()
