"""P8: reference-normalized amplitude and signed dynamics, unchanged P7 model.

No reference statistic enters inference. The old objective remains diagnostic;
it is not added to the optimized sum, avoiding its log-energy ranking conflict.
"""

import torch

from models.envelope_objective import BANDS
from models.factorized_envelope import ABSOLUTE_FLOOR, BURN, RELATIVE_FLOOR, factor_objective
from models.residual_energy import make_energy_model

ARMS = ("energy_control", "linear_reconstruction")
SMOOTHING = 1e-3


def make_linear_model(arm, width=144):
    if arm not in ARMS:
        raise ValueError("unknown linear objective arm")
    return make_energy_model("rms_residual", width)


def linear_objective(prediction, target, rate, burn=BURN):
    old = factor_objective(prediction, target, rate, burn=burn)
    p, t = prediction[:, burn:], target[:, burn:]
    floor = (t.square().mean((1, 2), keepdim=True).sqrt() * RELATIVE_FLOOR).clamp_min(
        ABSOLUTE_FLOOR
    )
    frequency = torch.arange(129, device=rate.device)[None, :] * rate / 256
    amplitude, dynamics, weak_sum, weak_count = [p.new_zeros(()) for _ in range(4)]
    for low, high in BANDS:
        mask = ((frequency >= low) & (frequency < high))[:, None, :]
        count = mask.sum((1, 2)).clamp_min(1)
        reference_power = (t.square() * mask).sum((1, 2)) / (count * t.shape[1])
        denominator = reference_power + floor[:, 0, 0].square()
        error_power = ((p - t).square() * mask).sum((1, 2)) / (count * t.shape[1])
        temporal_power = ((p.diff(dim=1) - t.diff(dim=1)).square() * mask).sum((1, 2)) / (
            count * (t.shape[1] - 1)
        )
        # Smooth norm: exact zero at reference, finite zero-error gradients.
        reconstruction = (error_power / denominator + SMOOTHING**2).sqrt() - SMOOTHING
        temporal = (temporal_power / denominator + SMOOTHING**2).sqrt() - SMOOTHING
        weight = (0.25 if low == 0 else 1.0) / 4.25
        amplitude = amplitude + weight * reconstruction.mean()
        dynamics = dynamics + weight * temporal.mean()
        if low != 0:
            weak = reference_power <= floor[:, 0, 0].square()
            weak_sum = weak_sum + (reconstruction * weak).sum()
            weak_count = weak_count + weak.sum()
    return {
        **old,
        "factor_total": old["total"],
        "total": amplitude + 0.25 * dynamics,
        "linear_reconstruction": amplitude,
        "linear_dynamics": dynamics,
        "weak_reconstruction": weak_sum / weak_count.clamp_min(1),
        "weak_band_count": weak_count,
    }
