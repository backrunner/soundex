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
    def __init__(
        self,
        arm,
        width=144,
        residual_slope=1.0,
        generation_budget=False,
        budget_scope="global",
        decoder_normalization=False,
        phase_input=False,
        residual_coordinates="input_relative",
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown restoration head")
        if not math.isfinite(residual_slope) or not 0 < residual_slope <= 1:
            raise ValueError("residual slope must be finite and in (0, 1]")
        if arm == "anchored" and residual_slope != 1:
            raise ValueError("anchored head does not use residual slope")
        if generation_budget and arm != "complex":
            raise ValueError("generation budget requires the complex head")
        if (decoder_normalization or phase_input) and arm != "complex":
            raise ValueError("mechanism interventions require the complex head")
        if residual_coordinates not in {"input_relative", "global_cartesian"} or (
            residual_coordinates != "input_relative" and arm != "complex"
        ):
            raise ValueError("global Cartesian residual coordinates require the complex head")
        if budget_scope not in {"global", "residual"} or (
            budget_scope == "residual" and not generation_budget
        ):
            raise ValueError("residual budget scope requires a generation budget")
        self.arm = arm
        self.width = width
        self.residual_slope = residual_slope
        self.generation_budget = generation_budget
        self.budget_scope = budget_scope
        self.decoder_normalization = decoder_normalization
        self.phase_input = phase_input
        self.residual_coordinates = residual_coordinates
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
        if generation_budget:
            # Preserve shared initialization and the original unit scale (=1).
            with torch.random.fork_rng(devices=[]):
                self.budget_head = nn.Linear(2 * width, 5)
            nn.init.zeros_(self.budget_head.weight)
            nn.init.constant_(self.budget_head.bias, math.log(math.expm1(1.0)))
        if decoder_normalization:
            self.decoder_norm = nn.LayerNorm(width)
        if phase_input:
            # Additive zero initialization preserves the common magnitude-only function
            # and RNG stream. Only observed current-frame phase enters this branch.
            with torch.random.fork_rng(devices=[]):
                self.phase_projection = nn.Linear(258, width, bias=False)
            nn.init.zeros_(self.phase_projection.weight)

    def budget_weights(self, condition):
        # Continuous piecewise-linear interpolation avoids hard band seams.
        hz = torch.arange(129, device=condition.device, dtype=condition.dtype) * (48000 / 256)
        hz = condition[..., :1] * hz
        centers = (2000, 6000, 10000, 14000, 20000)
        weights = []
        for i, center in enumerate(centers):
            left = (hz - centers[i - 1]) / (center - centers[i - 1]) if i else torch.ones_like(hz)
            right = (
                (centers[i + 1] - hz) / (centers[i + 1] - center) if i < 4 else torch.ones_like(hz)
            )
            weights.append(torch.minimum(left, right).clamp(0, 1))
        return torch.stack(weights, dim=-1)

    def initial_state(self, batch, *, device=None):
        return self.backbone.initial_state(batch, device=device)

    def forward(self, original, context, condition, state):
        output, state, _ = self.forward_details(original, context, condition, state)
        return output, state

    def forward_details(
        self, original, context, condition, state, *, detach_budget_auxiliary=False
    ):
        magnitude = torch.pow(10.0, original[:, 0] / 20)
        if self.arm == "anchored":
            amplitude, state = self.backbone(magnitude, context, condition, state)
            return amplitude_features(amplitude, original, burn=0), state, None
        encoder_residual = None
        if self.phase_input:
            phase_scale = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
            # Subtract the feature floor so exact digital silence carries no phase.
            carrier = torch.log1p((magnitude - 1e-10).clamp_min(0) / phase_scale)
            phase = original[:, 1]
            encoder_residual = self.phase_projection(
                torch.cat((carrier * phase.cos(), carrier * phase.sin()), dim=-1)
            )
        latent, scale, state = self.backbone.encode(
            magnitude, context, condition, state, encoder_residual=encoder_residual
        )
        if self.decoder_normalization:
            decoder = self.backbone.decoder
            logits = decoder[2](decoder[1](self.decoder_norm(decoder[0](latent))))
        else:
            logits = self.backbone.decoder(latent)
        delta = 2 * torch.tanh(self.residual_slope * logits)
        # Smooth local units remove independent hard-band normalization. No future frames.
        normalized = magnitude / scale
        power = F.avg_pool1d(F.pad(normalized.square(), (4, 4), mode="replicate"), 9, 1)
        # Global RMS remains available when a whole high band is absent.
        budget = F.softplus(self.budget_head(latent)) if self.generation_budget else None
        generation = (
            (self.budget_weights(condition) * budget[..., None, :]).sum(-1)
            if budget is not None
            else torch.ones_like(power)
        )
        # Experimental residual scope controls BOTH local and missing-band units.
        # Legacy checkpoints keep the original additive global-only budget.
        # This is a continuous scale, never a track-specific hard bypass gate.
        units = (
            scale * (power + 1).sqrt() * generation
            if self.budget_scope == "residual"
            else scale * (power + generation.square()).sqrt()
        )
        delta_real = delta[..., :129]
        delta_imag = delta[..., 129:] * self.interior if self.arm == "complex" else None
        if self.residual_coordinates == "global_cartesian":
            # The decoder predicts physical real/imaginary residuals. Rotate only
            # for the common polar output conversion, not to anchor the residual
            # direction to an unreliable observed phase. Zero stays exact identity.
            phase = original[:, 1]
            delta_real, delta_imag = (
                delta_real * phase.cos() + delta_imag * phase.sin(),
                -delta_real * phase.sin() + delta_imag * phase.cos(),
            )
        real = magnitude + units * delta_real
        imag = (
            units * delta_imag * self.interior if self.arm == "complex" else torch.zeros_like(real)
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
        # Only the auxiliary teacher's route into shared features is detached.
        # The rendered reconstruction still backpropagates through the live budget.
        auxiliary_budget = (
            F.softplus(self.budget_head(latent.detach()))
            if detach_budget_auxiliary and budget is not None
            else budget
        )
        return torch.stack((db, phase), dim=1), state, auxiliary_budget


def predict_budget(model, batch, context, burn=BURN, *, detach_auxiliary=False):
    _, _, condition = probe_inputs(batch, context)
    state = model.initial_state(len(context), device=context.device)
    with torch.no_grad():
        _, state = model(batch["x"][:, :, :burn], context[:, :burn], condition[:, :burn], state)
    tail, _, budget = model.forward_details(
        batch["x"][:, :, burn:],
        context[:, burn:],
        condition[:, burn:],
        state.detach(),
        detach_budget_auxiliary=detach_auxiliary,
    )
    return torch.cat((batch["x"][:, :, :burn], tail), dim=2), budget


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
