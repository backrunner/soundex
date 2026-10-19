"""P29 direct clean-spectrum supervision, before any detector mask or DSP.

Training-only reference scales never enter inference. This diagnostic objective
does not replace the common deployed quality metrics or qualify a model.
"""

import torch

from models.complex_synthesis import complex_features
from models.envelope_objective import BANDS

WEIGHTS = {"amplitude": 1.0, "complex": 0.1, "envelope": 0.1, "dynamics": 0.25}


def direct_spectrum_loss(predicted, batch, burn=89):
    """Equal paired/identity risk, HF emphasis, signed reference dynamics.

    All frequency bins are supervised, including inactive/masked-out regions. The
    low band has weight .25, other bands 1. Reference-band power is stabilized by
    1e-4 of full-reference power and an absolute 1e-14 power floor. Smooth norms
    are zero with zero gradient at the reference, also for silent/weak material.
    """
    if predicted.shape != batch["y"].shape or predicted.shape != batch["x"].shape:
        raise ValueError("aligned prediction/input/reference required")
    if burn < 0 or predicted.shape[2] - burn < 2:
        raise ValueError("at least two supervised frames required")
    identity = batch["identity"].bool()
    if not identity.any() or identity.all():
        raise ValueError("both paired and identity examples required")
    p = complex_features(predicted)[:, burn:]
    with torch.no_grad():
        target = torch.where(identity[:, None, None, None], batch["x"], batch["y"])
        t = complex_features(target)[:, burn:]
        tm = t.abs()
        floor = tm.square().mean((1, 2)) * 1e-4 + 1e-14
    pm = p.abs()
    hz = torch.arange(p.shape[-1], device=p.device)[None, :] * batch["rate"] / 256
    components = {k: pm.new_zeros(len(pm)) for k in WEIGHTS}
    bands = []
    for low, high in BANDS:
        mask = ((hz >= low) & (hz < high))[:, None, :]
        count = mask.sum(-1).clamp_min(1)
        target_power = (tm.square() * mask).sum(-1) / count
        predicted_power = (pm.square() * mask).sum(-1) / count
        denominator = target_power.mean(1) + floor

        def norm(power, denominator=denominator):
            return (power / denominator + 1e-6).sqrt() - 1e-3

        error = pm - tm
        values = {
            "amplitude": norm(((error.square() * mask).sum(-1) / count).mean(1)),
            "complex": norm((((p - t).abs().square() * mask).sum(-1) / count).mean(1)),
            "envelope": norm(
                ((predicted_power + floor[:, None]).sqrt() - (target_power + floor[:, None]).sqrt())
                .square()
                .mean(1)
            ),
            "dynamics": norm(((error.diff(dim=1).square() * mask).sum(-1) / count).mean(1)),
        }
        weight = (0.25 if low == 0 else 1.0) / 4.25
        for name in components:
            components[name] = components[name] + weight * values[name]
        bands.append({"hz": [low, high], **values})
    per_example = sum(WEIGHTS[k] * v for k, v in components.items())
    return {
        "total": 0.5 * (per_example[~identity].mean() + per_example[identity].mean()),
        "per_example": per_example,
        "components": components,
        "bands": bands,
    }
