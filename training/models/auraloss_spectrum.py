"""Original-reference, band-balanced auraloss operators on supervised raw OLA.

SPDX-License-Identifier: Apache-2.0

This is NOT stock MultiResolutionSTFTLoss: use upstream SC/log-magnitude operators,
but retain SoundEx's physical bands, per-example reduction and unpadded windows.
Reference-dependent smooth floors avoid division by silence and a hard-clamp dead
zone. The pinned upstream implementation and every transform are registered.
"""

from importlib.metadata import version

import torch
from auraloss.freq import SpectralConvergenceLoss, STFTMagnitudeLoss

from models.envelope_objective import BANDS
from models.multires_spectrum import (
    ABSOLUTE_POWER_FLOOR,
    FFT_SIZES,
    RELATIVE_POWER_FLOOR,
)

VERSION = "soundex-auraloss-bands-v1"


def spectral_loss(wave, reference, rates, *, fft_sizes=FFT_SIZES):
    """Per-example SC + log-L1, equal resolutions; low band weight .25, HF 1.

    No batch-global spectral convergence, mel transform, phase objective, gain
    invariance, DSP teacher, resampling or future audio outside supervised PCM.
    Smooth floors use the same reference power constants as the common scorer;
    they are not an audibility threshold. Window scale is calibrated PCM RMS.
    """
    if version("auraloss") != "0.4.0":
        raise ValueError("auraloss 0.4.0 is required")
    if wave.ndim != 2 or wave.shape != reference.shape or rates.numel() != len(wave):
        raise ValueError("aligned batched PCM and physical rates required")
    if not fft_sizes or any(n not in FFT_SIZES for n in fft_sizes):
        raise ValueError("unsupported spectral resolutions")
    if wave.shape[-1] < max(fft_sizes) + max(fft_sizes) // 4:
        raise ValueError("at least two complete windows per resolution required")
    rates = rates.reshape(-1)
    if not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical sample rates must be 44100 or 48000")
    if not torch.isfinite(wave).all() or not torch.isfinite(reference).all():
        raise ValueError("finite PCM required")
    reference = reference.detach()
    convergence = SpectralConvergenceLoss()
    magnitude = STFTMagnitudeLoss(log=True, distance="L1", reduction="mean")
    components = {k: wave.new_zeros(len(wave)) for k in ("convergence", "log_magnitude")}
    for size in fft_sizes:
        window = torch.hann_window(size, device=wave.device, dtype=wave.dtype)
        sides = wave.new_full((size // 2 + 1,), 2)
        sides[0] = sides[-1] = 1
        scale = (sides / (size * window.square().sum())).sqrt()
        predicted = torch.fft.rfft(wave.unfold(-1, size, size // 4) * window) * scale
        with torch.no_grad():
            target = torch.fft.rfft(reference.unfold(-1, size, size // 4) * window) * scale
            floor = target.abs().square().sum(-1).mean(-1) * RELATIVE_POWER_FLOOR
            floor = floor + ABSOLUTE_POWER_FLOOR
        hz = torch.arange(size // 2 + 1, device=wave.device)[None] * rates[:, None] / size
        for low, high in BANDS:
            mask = (hz >= low) & (hz < high)
            weight = (0.25 if low == 0 else 1.0) / (4.25 * len(fft_sizes))
            values = {k: [] for k in components}
            for i in range(len(wave)):
                # Each example gets its own denominator. Loud music must not
                # normalize away quiet music or the speech member of the batch.
                regularizer = floor[i] / mask[i].sum()
                pm = (predicted[i, :, mask[i]].abs().square() + regularizer).sqrt()
                tm = (target[i, :, mask[i]].abs().square() + regularizer).sqrt()
                values["convergence"].append(convergence(pm, tm))
                values["log_magnitude"].append(magnitude(pm, tm))
            for key in components:
                components[key] = components[key] + weight * torch.stack(values[key])
    return {"per_example": sum(components.values()), "components": components}
