"""P32 spectrum-first supervision of realizable raw OLA, without DSP masks.

PCM is only the intermediate representation for multiple STFTs. No samplewise
waveform distance is optimized. Targets are original, aligned, unclipped PCM.
"""

import torch

from models.complex_synthesis import complex_features, synthesize
from models.envelope_objective import BANDS

VERSION = "soundex-multires-spectrum-v1"
FFT_SIZES = (256, 1024, 2048)
WEIGHTS = {"amplitude": 1.0, "log_shape": 0.25, "envelope": 0.1, "dynamics": 0.25}
RELATIVE_POWER_FLOOR = 1e-4
ABSOLUTE_POWER_FLOOR = 1e-12


def spectral_loss(wave, reference, rates, *, fft_sizes=FFT_SIZES):
    """Per-example, equal-resolution/Hz-band risk on already aligned support.

    One-sided Hann/Parseval scaling gives per-bin RMS units. Reference-dependent
    floors are detached, finite on silence, and identical for every candidate.
    No centering/padding: each analysis window stays inside supervised support.
    """
    if wave.ndim != 2 or wave.shape != reference.shape or len(rates) != len(wave):
        raise ValueError("aligned batched PCM and physical rates required")
    if not fft_sizes or any(n not in FFT_SIZES for n in fft_sizes):
        raise ValueError("unsupported spectral resolutions")
    if wave.shape[-1] < max(fft_sizes) + max(fft_sizes) // 4:
        raise ValueError("at least two complete windows per resolution required")
    if not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical sample rates must be 44100 or 48000")
    reference = reference.detach()
    components = {k: wave.new_zeros(len(wave)) for k in WEIGHTS}
    resolutions = []
    for size in fft_sizes:
        window = torch.hann_window(size, device=wave.device, dtype=wave.dtype)
        sides = wave.new_full((size // 2 + 1,), 2)
        sides[0] = sides[-1] = 1
        scale = (sides / (size * window.square().sum())).sqrt()
        pm = (torch.fft.rfft(wave.unfold(-1, size, size // 4) * window) * scale).abs()
        with torch.no_grad():
            tm = (torch.fft.rfft(reference.unfold(-1, size, size // 4) * window) * scale).abs()
            floor = tm.square().sum(-1).mean(-1) * RELATIVE_POWER_FLOOR + ABSOLUTE_POWER_FLOOR
        hz = torch.arange(size // 2 + 1, device=wave.device)[None, :] * rates.reshape(-1, 1) / size
        bands = []
        current = {k: wave.new_zeros(len(wave)) for k in WEIGHTS}
        for low, high in BANDS:
            mask = ((hz >= low) & (hz < high))[:, None, :]
            count = mask.sum(-1).clamp_min(1)
            target_power = (tm.square() * mask).sum(-1)
            predicted_power = (pm.square() * mask).sum(-1)
            denominator = target_power.mean(-1) + floor
            error = pm - tm

            def norm(power, denominator=denominator):
                return (power / denominator + 1e-6).sqrt() - 1e-3

            log_floor = (floor[:, None] / count).sqrt()[:, :, None]
            log_error = torch.log1p(pm / log_floor) - torch.log1p(tm / log_floor)
            values = {
                "amplitude": norm((error.square() * mask).sum(-1).mean(-1)),
                "log_shape": (((log_error.square() * mask).sum(-1) / count).mean(-1) + 1e-6).sqrt()
                - 1e-3,
                "envelope": norm(
                    (
                        (predicted_power + floor[:, None]).sqrt()
                        - (target_power + floor[:, None]).sqrt()
                    )
                    .square()
                    .mean(-1)
                ),
                "dynamics": norm((error.diff(dim=1).square() * mask).sum(-1).mean(-1)),
            }
            weight = (0.25 if low == 0 else 1.0) / 4.25
            for name, value in values.items():
                current[name] = current[name] + weight * value
            bands.append({"hz": [low, high], **values})
        for name in WEIGHTS:
            components[name] = components[name] + current[name] / len(fft_sizes)
        resolutions.append({"fft_size": size, "components": current, "bands": bands})
    return {
        "per_example": sum(WEIGHTS[k] * v for k, v in components.items()),
        "components": components,
        "resolutions": resolutions,
    }


def supervised_waves(predicted, batch, burn=89):
    """Same region as FixedFitObjective, excluding startup and final OLA support.

    The preceding predicted frame contributes to the first sample by overlap;
    it is guard context, NOT the 80-frame no-grad model warmup. No reference
    projection, masking or clipping is applied to the clean PCM target.
    """
    if predicted.shape != batch["x"].shape or predicted.shape != batch["y"].shape:
        raise ValueError("prediction and paired feature geometry must match")
    if burn < 80:
        raise ValueError("supervision must exclude model warmup")
    wave = synthesize(complex_features(predicted))
    target = batch["clean_wave"].detach()
    if wave.shape != target.shape:
        raise ValueError("raw OLA and clean PCM timeline mismatch")
    region = slice(burn * 128, wave.shape[-1] - 128)
    return wave[:, region], target[:, region]


def multires_spectrum_loss(predicted, batch, burn=89):
    identity = batch["identity"].bool()
    if not identity.any() or identity.all():
        raise ValueError("paired and identity roles required")
    result = spectral_loss(*supervised_waves(predicted, batch, burn), batch["rate"])
    per_example = result["per_example"]
    result["total"] = 0.5 * (per_example[~identity].mean() + per_example[identity].mean())
    return result
