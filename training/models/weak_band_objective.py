"""Reference-conditioned excess-energy supervision; no reference at inference.

Fixed relative (-60 dB power) and absolute (-100 dBFS) floors keep silent
bands finite without the v1 whole-spectrum 1% amplitude floor hiding excess.
Bands are fixed in Hz, independent of title/codec/genre and dataset outliers.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from models.causal_objective import BURN, render
from models.causal_objective import objective as base_objective

VERSION = "causal-weak-band-v1"
EXCESS_WEIGHT = 0.25
BAND_EDGES_HZ = (8000, 10000, 12000, 14000, 16000, 18000, 20000, 24001)
RELATIVE_POWER_FLOOR = 1e-6
ABSOLUTE_POWER_FLOOR = 1e-10


def excess_band_energy(predicted, target, sample_rate, fft_size):
    """Mean log1p excess per 2kHz band/frame, using one-sided Parseval power.

    The detached denominator depends ONLY on the reference. Predicting the
    reference has zero penalty; silence/dry is not rewarded over reconstruction.
    Base reconstruction terms remain responsible for missing/incorrect detail.
    """
    sides = torch.ones(fft_size // 2 + 1, device=predicted.device, dtype=predicted.real.dtype) * 2
    sides[0] = sides[-1] = 1
    # Periodic Hann has mean square 3/8 for all supported FFT lengths.
    scale = sides / (fft_size**2 * 0.375)
    p, t = predicted.abs().square() * scale, target.abs().square() * scale
    floor = (
        (t.sum(-1, keepdim=True) * RELATIVE_POWER_FLOOR).clamp_min(ABSOLUTE_POWER_FLOOR).detach()
    )
    bins = torch.arange(fft_size // 2 + 1, device=p.device, dtype=p.dtype)
    hz = bins[None, :] * sample_rate.reshape(-1, 1) / fft_size
    terms = []
    for low, high in pairwise(BAND_EDGES_HZ):
        mask = ((hz >= low) & (hz < high))[:, None, :]
        pp, tt = (p * mask).sum(-1), (t * mask).sum(-1)
        terms.append(torch.log1p(torch.relu(pp - tt) / (tt + floor[..., 0]).detach()))
    return torch.stack(terms).mean()


def objective(predicted, batch):
    terms = base_objective(predicted, batch)
    output, _ = render(
        predicted, batch["x"], batch["mask"], batch["active"], batch["gate"], batch["dry"]
    )
    waves = [w[:, (BURN + 1) * 128 :] for w in (output, batch["clean_wave"])]
    penalties = []
    for size in (256, 512, 2048):
        window = torch.hann_window(size, device=predicted.device, dtype=predicted.dtype)
        p, t = [torch.fft.rfft(w.unfold(-1, size, size // 4) * window) for w in waves]
        penalties.append(excess_band_energy(p, t, batch["rate"], size))
    terms["base_total"] = terms["total"]
    terms["output_excess_band_energy"] = torch.stack(penalties).mean()
    terms["total"] = terms["base_total"] + EXCESS_WEIGHT * terms["output_excess_band_energy"]
    return terms
