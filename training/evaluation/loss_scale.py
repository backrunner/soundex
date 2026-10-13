"""Read-only decomposition of P13 spectral risk; never changes the training loss."""

import torch

from models.envelope_objective import BANDS
from models.factorized_envelope import ABSOLUTE_FLOOR, RELATIVE_FLOOR
from models.linear_objective import SMOOTHING


def spectral_parts(prediction, target, rate, burn=89):
    """Single-clip loss terms with the exact P13 normalizers and frame support."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] <= burn + 1:
        raise ValueError("expected matching [frames, bins] with supervised dynamics")
    p, t = prediction[burn:], target[burn:]
    global_rms = t.square().mean().sqrt()
    floor = (global_rms * RELATIVE_FLOOR).clamp_min(ABSOLUTE_FLOOR)
    frequency = torch.arange(p.shape[-1], device=p.device) * rate / 256
    result = []
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high)
        count = mask.sum().clamp_min(1)
        reference_power = (t.square() * mask).sum() / (count * t.shape[0])
        output_power = (p.square() * mask).sum() / (count * p.shape[0])
        error_power = ((p - t).square() * mask).sum() / (count * p.shape[0])
        temporal_power = ((p.diff(dim=0) - t.diff(dim=0)).square() * mask).sum() / (
            count * (p.shape[0] - 1)
        )
        denominator = reference_power + floor.square()
        weight = (0.25 if low == 0 else 1.0) / 4.25
        amplitude = weight * ((error_power / denominator + SMOOTHING**2).sqrt() - SMOOTHING)
        dynamics = (
            0.25 * weight * ((temporal_power / denominator + SMOOTHING**2).sqrt() - SMOOTHING)
        )
        result.append(
            {
                "band": f"{low}_{high}",
                "bins": int(mask.sum()),
                "reference_rms": reference_power.sqrt(),
                "output_rms": output_power.sqrt(),
                "error_rms": error_power.sqrt(),
                "global_reference_rms": global_rms,
                "reference_to_global": reference_power.sqrt() / global_rms.clamp_min(1e-20),
                "error_to_global": error_power.sqrt() / global_rms.clamp_min(1e-20),
                "normalization_floor": floor,
                "denominator": denominator,
                "floor_dominated": bool(reference_power <= floor.square()),
                "amplitude_loss": amplitude,
                "dynamics_loss": dynamics,
                "total": amplitude + dynamics,
            }
        )
    return result


def summarize_parts(parts, prediction=None):
    records = []
    for part in parts:
        record = {
            k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in part.items()
        }
        if prediction is not None:
            gradient = torch.autograd.grad(part["total"], prediction, retain_graph=True)[0]
            record["amplitude_coordinate_gradient_l2"] = float(gradient.norm())
            record["clip_relative_coordinate_gradient_l2"] = float(
                (gradient * part["global_reference_rms"].detach()).norm()
            )
        records.append(record)
    return records


def gradient_comparison(paired, identity):
    """Unweighted single-example parameter gradients, not actual Adam updates."""
    pairs = [
        (a, b) for a, b in zip(paired, identity, strict=True) if a is not None and b is not None
    ]
    aa = sum(float(a.double().square().sum()) for a, _ in pairs)
    bb = sum(float(b.double().square().sum()) for _, b in pairs)
    ab = sum(float((a.double() * b.double()).sum()) for a, b in pairs)
    return {
        "paired_l2": aa**0.5,
        "identity_l2": bb**0.5,
        "equal_pair_l2": max(0.0, 0.25 * (aa + bb + 2 * ab)) ** 0.5,
        "cosine": ab / (aa * bb) ** 0.5 if aa > 0 and bb > 0 else None,
    }
