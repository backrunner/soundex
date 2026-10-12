"""Amplitude-only diagnostic, fixed frequency bands, signed reference dynamics.

No waveform synthesis or deployment-quality claim. Floors are normalization,
not clean-target modification. All arms use this identical objective.
"""

import torch

BANDS = ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001))


def envelope_objective(prediction, target, rate, burn=64):
    prediction, target = prediction[:, burn:], target[:, burn:]
    floor = (target.square().mean((1, 2), keepdim=True).sqrt() * 0.01).clamp_min(1e-5)
    p, t = torch.log1p(prediction / floor), torch.log1p(target / floor)
    frequency = torch.arange(129, device=rate.device)[None, :] * rate / 256
    amplitude, temporal = prediction.new_zeros(()), prediction.new_zeros(())
    for low, high in BANDS:
        mask = ((frequency >= low) & (frequency < high))[:, None, :]
        weight = 0.25 if low == 0 else 1.0
        count = mask.sum((1, 2)).clamp_min(1)
        amplitude = (
            amplitude
            + weight * (((p - t).abs() * mask).sum((1, 2)) / (count * p.shape[1])).mean() / 4.25
        )
        temporal = (
            temporal
            + weight
            * (
                ((p.diff(dim=1) - t.diff(dim=1)).abs() * mask).sum((1, 2))
                / (count * (p.shape[1] - 1))
            ).mean()
            / 4.25
        )
    return {"total": amplitude + 0.25 * temporal, "amplitude": amplitude, "temporal": temporal}


def amplitude_metrics(prediction, target, rate, burn=64):
    """Single-clip raw amplitude diagnostics, never masquerading as audio metrics."""
    p, t = prediction[0, burn:].double(), target[0, burn:].double()
    frequency = torch.arange(129, device=p.device) * float(rate[0, 0]) / 256
    result = {}
    for low, high in BANDS:
        a, b = (
            p[:, (frequency >= low) & (frequency < high)],
            t[:, (frequency >= low) & (frequency < high)],
        )
        prefix = f"band_{low}_{high}"
        result[prefix + "_sc"] = float((a - b).norm() / b.norm().clamp_min(1e-10))
        result[prefix + "_flux"] = float(
            (a.diff(dim=0).clamp_min(0) - b.diff(dim=0).clamp_min(0)).norm()
            / b.diff(dim=0).clamp_min(0).norm().clamp_min(1e-10)
        )
        bias = (
            10 * (a.square().mean().clamp_min(1e-20) / b.square().mean().clamp_min(1e-20)).log10()
        )
        result[prefix + "_energy_bias_db"] = float(bias)
        result[prefix + "_energy_error_db"] = float(bias.abs())
        # Spectral power in DFT units, explicitly not waveform dBFS.
        result[prefix + "_reference_power_db"] = float(
            10 * b.square().mean().clamp_min(1e-20).log10()
        )
    result["high_sc"] = sum(result[f"band_{a}_{b}_sc"] for a, b in BANDS[1:]) / 4
    result["high_flux"] = sum(result[f"band_{a}_{b}_flux"] for a, b in BANDS[1:]) / 4
    result["high_energy_error_db"] = (
        sum(result[f"band_{a}_{b}_energy_error_db"] for a, b in BANDS[1:]) / 4
    )
    return result
