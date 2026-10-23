"""P43 spectrum-led direction; fidelity remains in finite-step assessment.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from evaluation.batch_update_probe import average, gradients_for
from models.guarded_direction import project_direction
from scripts.audit_update_directions import unit

GUARDS = (
    "paired_direct_amplitude",
    "paired_mr",
    "paired_mr_dynamics",
    "identity_mr",
    "identity_risk",
    "budget",
)


def spectral_directions(gradients: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Magnitude and MR choose the direction; no complex-gradient bisector.

    Complex error and the original total remain in the unchanged finite-step
    policies. Omitting their first-order guards does not remove their reporting
    or permit arbitrary phase damage. All original loss definitions stay intact.
    """
    proposed = -unit(unit(gradients["paired_direct_amplitude"]) + unit(gradients["paired_mr"]))
    projected, certificate = project_direction(
        proposed, torch.stack([gradients[k] for k in GUARDS])
    )
    directions = {"pooled_spectrum": proposed, "pooled_spectrum_guarded": unit(projected)}
    return directions, {
        "guard_names": GUARDS,
        "projection": certificate,
        "pooled_normalized_derivatives": {
            name: {key: float(unit(g).dot(d)) for key, g in gradients.items()}
            for name, d in directions.items()
        },
    }


def directions_for(model: torch.nn.Module, items: list[tuple]) -> tuple[dict, dict]:
    return spectral_directions(average([gradients_for(model, item)[0] for item in items]))
