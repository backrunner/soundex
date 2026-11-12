"""P75 diagnostic original-reference complex HF error after synthesis.

Not installed in the training objective. No phase unwrapping, reference clipping,
level matching, weak-band dropping or inference-time operation.
SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.multires_spectrum import (
    ABSOLUTE_POWER_FLOOR,
    FFT_SIZES,
    RELATIVE_POWER_FLOOR,
)

HF_BANDS = ((8000, 12000), (12000, 16000), (16000, 24001))


def realized_hf_terms(wave, reference, rates):
    """Equal-resolution/band smooth relative norms and calibrated PCM powers.

    Uses the existing MR windows, floors and supervised support. Complex error
    couples amplitude and phase; its reduction alone can reward attenuation.
    Retain magnitude error and the exact phase-interaction power separately.
    """
    if wave.ndim != 2 or wave.shape != reference.shape or len(rates) != len(wave):
        raise ValueError("aligned batched PCM and physical rates required")
    if wave.shape[-1] < max(FFT_SIZES) + max(FFT_SIZES) // 4:
        raise ValueError("at least two complete windows per resolution required")
    if not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical sample rates must be 44100 or 48000")
    reference = reference.detach()
    result = {k: wave.new_zeros(len(wave)) for k in ("complex", "magnitude")}
    rows = []
    for size in FFT_SIZES:
        window = torch.hann_window(size, device=wave.device, dtype=wave.dtype)
        sides = wave.new_full((size // 2 + 1,), 2)
        sides[0] = sides[-1] = 1
        scale = (sides / (size * window.square().sum())).sqrt()
        p = torch.fft.rfft(wave.unfold(-1, size, size // 4) * window) * scale
        t = torch.fft.rfft(reference.unfold(-1, size, size // 4) * window) * scale
        pm, tm = p.abs(), t.abs()
        floor = tm.square().sum(-1).mean(-1) * RELATIVE_POWER_FLOOR + ABSOLUTE_POWER_FLOOR
        hz = torch.arange(size // 2 + 1, device=wave.device)[None, :] * rates.reshape(-1, 1) / size
        for low, high in HF_BANDS:
            mask = ((hz >= low) & (hz < high))[:, None]

            def power(values, mask=mask):
                return (values * mask).sum(-1).mean(-1)

            ref_power = power(tm.square())
            mag_power = power((pm - tm).square())
            error = p - t
            complex_power = power(error.real.square() + error.imag.square())
            for key, value in (("complex", complex_power), ("magnitude", mag_power)):
                result[key] = result[key] + ((value / (ref_power + floor) + 1e-6).sqrt() - 1e-3) / (
                    len(FFT_SIZES) * len(HF_BANDS)
                )
            rows.append(
                {
                    "fft_size": size,
                    "hz": [low, high],
                    "reference_power": ref_power,
                    "prediction_power": power(pm.square()),
                    "magnitude_error_power": mag_power,
                    "complex_error_power": complex_power,
                    "phase_interaction_power": complex_power - mag_power,
                }
            )
    return {**result, "bands": rows}
