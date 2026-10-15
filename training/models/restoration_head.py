"""P20 causal residual heads with smooth input-only frequency units."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.envelope_probe import probe_inputs
from models.fit_diagnostic import BURN
from models.rendered_objective import amplitude_features
from models.rendered_probe import make_rendered_model

ARMS = ("anchored", "amplitude", "complex")


class RestorationHead(nn.Module):
    def __init__(self, arm, width=144, residual_slope=1.0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown restoration head")
        if not math.isfinite(residual_slope) or not 0 < residual_slope <= 1:
            raise ValueError("residual slope must be finite and in (0, 1]")
        if arm == "anchored" and residual_slope != 1:
            raise ValueError("anchored head does not use residual slope")
        self.arm = arm
        self.width = width
        self.residual_slope = residual_slope
        self.backbone = make_rendered_model("rendered_balanced", width)
        self.state_size = self.backbone.state_size
        if arm != "anchored":
            del self.backbone.energy_head
            if arm == "complex":
                # Shared tensors and real output initialization are unchanged.
                with torch.random.fork_rng(devices=[]):
                    self.backbone.decoder[-1] = nn.Linear(width, 258)
            nn.init.zeros_(self.backbone.decoder[-1].weight)
            nn.init.zeros_(self.backbone.decoder[-1].bias)
        self.register_buffer("interior", torch.tensor([0.0] + [1.0] * 127 + [0.0]))

    def initial_state(self, batch, *, device=None):
        return self.backbone.initial_state(batch, device=device)

    def forward(self, original, context, condition, state):
        magnitude = torch.pow(10.0, original[:, 0] / 20)
        if self.arm == "anchored":
            amplitude, state = self.backbone(magnitude, context, condition, state)
            return amplitude_features(amplitude, original, burn=0), state
        latent, scale, state = self.backbone.encode(magnitude, context, condition, state)
        delta = 2 * torch.tanh(self.residual_slope * self.backbone.decoder(latent))
        # Smooth local units remove independent hard-band normalization. No future frames.
        normalized = magnitude / scale
        power = F.avg_pool1d(F.pad(normalized.square(), (4, 4), mode="replicate"), 9, 1)
        # Global RMS remains available when a whole high band is absent.
        units = scale * (power + 1.0).sqrt()
        real = magnitude + units * delta[..., :129]
        imag = (
            units * delta[..., 129:] * self.interior
            if self.arm == "complex"
            else torch.zeros_like(real)
        )
        epsilon = torch.full_like(scale, 1e-15)
        amplitude = (real.square() + imag.square() + epsilon.square()).sqrt()
        baseline = (magnitude.square() + epsilon.square()).sqrt()
        db = original[:, 0] + 20 * (
            amplitude.clamp_min(1e-10).log10() - baseline.clamp_min(1e-10).log10()
        )
        # Exact zero-coordinate identity, including weak/silent bins. Avoid atan2(0,0).
        if self.arm == "complex":
            safe_real = torch.where((real == 0) & (imag == 0), epsilon, real)
            phase = original[:, 1] + torch.atan2(imag, safe_real)
        else:
            phase = original[:, 1]
        return torch.stack((db, phase), dim=1), state


def predict_head(model, batch, context, burn=BURN):
    _, _, condition = probe_inputs(batch, context)
    state = model.initial_state(len(context), device=context.device)
    with torch.no_grad():
        _, state = model(batch["x"][:, :, :burn], context[:, :burn], condition[:, :burn], state)
    tail, _ = model(batch["x"][:, :, burn:], context[:, burn:], condition[:, burn:], state.detach())
    return torch.cat((batch["x"][:, :, :burn], tail), dim=2)


def shared_state(model):
    """Only shared input encoder/TCN/decoder-hidden tensors, for matched initialization."""
    return {
        k: v
        for k, v in copy.deepcopy(model.state_dict()).items()
        if k.startswith(("backbone.encoder.", "backbone.temporal.", "backbone.decoder.0."))
    }


def load_restoration_head(checkpoint):
    """Explicit P21 configuration; old P20 checkpoints retain their exact defaults."""
    config = checkpoint.get("model_config", {"arm": checkpoint["arm"]})
    model = RestorationHead(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
