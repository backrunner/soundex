"""P46 separates reference-based acceptance from reference-gradient projection.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from evaluation.anchored_training import RADII, metric_guard, observations
from evaluation.batch_update_probe import average, gradients_for
from evaluation.reference_preservation import pcm_guard, reference_gradients
from evaluation.spectral_priority_probe import GUARDS, spectral_directions
from models.guarded_direction import project_direction
from scripts.audit_update_directions import restore, unit


def direction_for(gradients: dict, preservation: dict | None) -> tuple:
    if preservation is None:
        directions, certificate = spectral_directions(gradients)
        return directions["pooled_spectrum_guarded"], certificate
    proposed = -unit(unit(gradients["paired_direct_amplitude"]) + unit(gradients["paired_mr"]))
    guards = {
        **{k: gradients[k] for k in GUARDS},
        **{"reference_" + k: v for k, v in preservation.items()},
    }
    projected, certificate = project_direction(proposed, torch.stack(list(guards.values())))
    direction = unit(projected)
    return direction, {
        "guard_names": list(guards),
        "projection": certificate,
        "normalized_derivatives": {k: float(unit(g).dot(direction)) for k, g in guards.items()},
    }


def attempt(model: torch.nn.Module, items: list, anchor: tuple, bits: int, projected: bool) -> dict:
    base, _ = observations(model, items)
    gradients = average([gradients_for(model, item)[0] for item in items[:4]])
    preservation = (
        average([reference_gradients(model, item) for item in items[:4]]) if projected else None
    )
    direction, certificate = direction_for(gradients, preservation)
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    points, accepted = [], False
    try:
        for radius in RADII:
            restore(params, original, radius * direction)
            current, waves = observations(model, items)
            spectral, pcm = (
                metric_guard(base, current, anchor[0]),
                pcm_guard(waves, anchor[1], items, bits),
            )
            accepted = (
                spectral["spectral_progress"]
                and not spectral["violations"]
                and not pcm["violations"]
            )
            points.append(
                {
                    "radius": radius,
                    "current": current,
                    "metric_guard": spectral,
                    "pcm_guard": pcm,
                    "accepted": accepted,
                }
            )
            if accepted:
                break
    except BaseException:
        restore(params, original)
        raise
    if not accepted:
        restore(params, original)
        if not all(torch.equal(p, old) for p, old in zip(params, original, strict=True)):
            raise ValueError("rejected proposal changed parameters")
    return {
        "accepted": accepted,
        "base": base,
        "certificate": certificate,
        "points": points,
        "selected_radius": points[-1]["radius"] if accepted else None,
    }
