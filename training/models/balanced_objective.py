"""P4: smooth deployment, reference-aware peaks and bidirectional band energy.

All references are training-only. Never limit the clean target to hide distortion.
The original P2/P3 objectives keep their legacy defaults and weights.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from models.causal_objective import BURN, render
from models.causal_objective import objective as base_objective
from models.weak_band_objective import (
    ABSOLUTE_POWER_FLOOR,
    BAND_EDGES_HZ,
    EXCESS_WEIGHT,
    RELATIVE_POWER_FLOOR,
    excess_band_energy,
)

VERSION = "causal-balanced-band-v1"
LIMITER_MODE = "smooth-knee"
ARMS = ("smooth_control", "reference_peak", "balanced")


def balanced_band_energy(predicted, target, sample_rate, fft_size):
    """Absolute log power ratio: a floor-stabilized, two-sided restoration cost.

    Equal multiplicative power excess/deficit have equal costs away from the
    noise floor. At exact reference the cost and additional gradient are zero.
    Per-frame reference-only floors and periodic Hann/one-sided Parseval scaling
    match the P3 bands, isolating the change from one-sided to two-sided error.
    """
    sides = torch.ones(fft_size // 2 + 1, device=predicted.device, dtype=predicted.real.dtype) * 2
    sides[0] = sides[-1] = 1
    scale = sides / (fft_size**2 * 0.375)
    p, t = predicted.abs().square() * scale, target.abs().square() * scale
    floor = (t.sum(-1) * RELATIVE_POWER_FLOOR).clamp_min(ABSOLUTE_POWER_FLOOR).detach()
    bins = torch.arange(fft_size // 2 + 1, device=p.device, dtype=p.dtype)
    hz = bins[None, :] * sample_rate.reshape(-1, 1) / fft_size
    terms = []
    for low, high in pairwise(BAND_EDGES_HZ):
        mask = ((hz >= low) & (hz < high))[:, None, :]
        pp, tt = (p * mask).sum(-1), (t * mask).sum(-1)
        terms.append((torch.log1p(pp / floor) - torch.log1p(tt / floor)).abs())
    return torch.stack(terms).mean()


def objective(predicted, batch, *, arm="balanced"):
    if arm not in ARMS:
        raise ValueError("unknown P4 arm")
    terms = base_objective(
        predicted, batch, limiter_mode=LIMITER_MODE, reference_peak=arm != "smooth_control"
    )
    output, _ = render(
        predicted,
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode=LIMITER_MODE,
    )
    waves = [w[:, (BURN + 1) * 128 :] for w in (output, batch["clean_wave"])]
    band_fn = balanced_band_energy if arm == "balanced" else excess_band_energy
    penalties = []
    for size in (256, 512, 2048):
        window = torch.hann_window(size, device=predicted.device, dtype=predicted.dtype)
        p, t = [torch.fft.rfft(w.unfold(-1, size, size // 4) * window) for w in waves]
        penalties.append(band_fn(p, t, batch["rate"], size))
    terms["base_total"] = terms["total"]
    terms["output_band_energy"] = torch.stack(penalties).mean()
    terms["total"] = terms["base_total"] + EXCESS_WEIGHT * terms["output_band_energy"]
    return terms
