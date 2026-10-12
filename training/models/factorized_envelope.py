"""P6 amplitude-only probe with independently controlled band RMS and shape.

The additional envelope head is input-conditioned, not clean-conditioned. The
clean reference is used only by the loss. No production audio contract is implied.
"""

import torch
from torch import nn

from models.envelope_objective import BANDS, envelope_objective
from models.envelope_probe import EnvelopeProbe

BURN = 80
ARMS = ("direct_legacy", "direct_factored_loss", "factorized")
RELATIVE_FLOOR = 1e-4
ABSOLUTE_FLOOR = 1e-7


class FactorizedEnvelope(EnvelopeProbe):
    def __init__(self, width=144):
        super().__init__(width)
        self.energy_head = nn.Linear(2 * width, len(BANDS))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)
        self.register_buffer("bin_index", torch.arange(129, dtype=torch.float32))

    def forward(self, magnitude, context, condition, state):
        latent, scale, next_state = self.encode(magnitude, context, condition, state)
        delta = 2 * torch.tanh(self.decoder(latent))
        # Positive seed also permits generation when an entire band is exactly 0.
        seed = ((magnitude + scale * (delta + 1e-8)).square() + (scale * 1e-8).square()).sqrt()
        energy_delta = 4 * self.energy_head(latent)
        frequency = self.bin_index[None, None, :] * condition[:, :1, :1] * (48000 / 256)
        prediction = torch.zeros_like(magnitude)
        for index, (low, high) in enumerate(BANDS):
            mask = ((frequency >= low) & (frequency < high)).to(magnitude.dtype)
            count = mask.sum(-1, keepdim=True).clamp_min(1)
            input_rms = (
                (magnitude.square() * mask).sum(-1, keepdim=True) / count + (scale * 1e-8).square()
            ).sqrt()
            seed_rms = (
                ((seed.square() * mask).sum(-1, keepdim=True) / count).clamp_min(1e-36).sqrt()
            )
            log_ratio = (input_rms / scale).log() + energy_delta[..., index : index + 1]
            # The lower bound must be below log(1e-8), our input seed floor,
            # otherwise completely missing bands start with a dead energy head.
            energy = scale * log_ratio.clamp(-30, 2).exp()
            prediction = prediction + mask * seed / seed_rms * energy
        return prediction, next_state


def make_factor_model(arm, width=144):
    if arm not in ARMS:
        raise ValueError("unknown factor study arm")
    model = FactorizedEnvelope(width) if arm == "factorized" else EnvelopeProbe(width)
    # All arms start at the same near-identity function, with matching backbone
    # tensors for a given seed. Avoid arm-specific random output contamination.
    nn.init.zeros_(model.decoder[-1].weight)
    nn.init.zeros_(model.decoder[-1].bias)
    return model


def factor_objective(prediction, target, rate, burn=BURN):
    """Shared selection loss; per-frame energy, normalized shape and dynamics.

    Reference floor: 1e-4 of supervised clip spectral RMS, at least 1e-7 DFT
    amplitude. This stabilizes divisions without changing reference targets.
    """
    original = envelope_objective(prediction, target, rate, burn=burn)
    p, t = prediction[:, burn:], target[:, burn:]
    floor = (t.square().mean((1, 2), keepdim=True).sqrt() * RELATIVE_FLOOR).clamp_min(
        ABSOLUTE_FLOOR
    )
    frequency = torch.arange(129, device=rate.device)[None, :] * rate / 256
    energy, shape, temporal = [p.new_zeros(()) for _ in range(3)]
    for low, high in BANDS:
        mask = ((frequency >= low) & (frequency < high))[:, None, :]
        count = mask.sum(-1, keepdim=True).clamp_min(1)
        pp = (p.square() * mask).sum(-1, keepdim=True) / count
        tp = (t.square() * mask).sum(-1, keepdim=True) / count
        # Smooth RMS avoids undefined sqrt derivatives at silence.
        pn, tn = (pp + floor.square()).sqrt(), (tp + floor.square()).sqrt()
        lp, lt = (pn / floor).log(), (tn / floor).log()
        weight = (0.25 if low == 0 else 1.0) / 4.25
        energy = energy + weight * (lp - lt).abs().mean()
        confidence = tp / (tp + floor.square())
        shape = (
            shape
            + weight
            * (((p / pn - t / tn).abs() * mask).sum(-1, keepdim=True) / count * confidence).mean()
        )
        temporal = temporal + weight * (lp.diff(dim=1) - lt.diff(dim=1)).abs().mean()
    total = energy + 0.5 * shape + 0.25 * temporal + 0.25 * original["amplitude"]
    return {
        "total": total,
        "band_envelope": energy,
        "band_shape": shape,
        "envelope_temporal": temporal,
        "amplitude": original["amplitude"],
        "temporal": original["temporal"],
        "legacy_total": original["total"],
    }
