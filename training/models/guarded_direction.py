"""Diagnostic Euclidean projection onto simultaneous linear descent constraints.

Related formulation: GEM, https://arxiv.org/abs/1706.08840. This is an independent
small active-set solver, not PCGrad or an implementation of continual learning.
Only first-order, same-batch constraints are enforced; finite-step and transfer
behavior must be measured separately. No model or production optimizer uses it.
SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import torch


def project_direction(
    proposed: torch.Tensor, gradients: torch.Tensor, tolerance: float = 1e-10
) -> tuple[torch.Tensor, dict]:
    """Minimize ||d-proposed||^2 subject to gradients @ d <= 0, in float64.

    Enumerate active subsets of at most eight guards and check primal/dual KKT
    conditions. Zero gradients impose no constraint. Rank-deficient subsets use
    least squares; no diagonal jitter changes the feasible set. A zero projection
    remains zero, rather than amplifying numerical residue into an update.
    """
    if (
        proposed.ndim != 1
        or gradients.ndim != 2
        or gradients.shape[1] != proposed.numel()
        or not 1 <= len(gradients) <= 8
        or not 0 < tolerance < 1e-6
        or not proposed.is_floating_point()
        or not gradients.is_floating_point()
        or not torch.isfinite(proposed).all()
        or not torch.isfinite(gradients).all()
    ):
        raise ValueError("finite floating vector and one to eight aligned guards required")
    p = proposed.detach().cpu().double().numpy()
    guards = gradients.detach().cpu().double().numpy()
    norms = np.linalg.norm(guards, axis=1)
    nonzero = norms > 0
    guards = guards[nonzero] / norms[nonzero, None]
    gram, dots = guards @ guards.T, guards @ p
    best = None
    for size in range(len(guards) + 1):
        for active in itertools.combinations(range(len(guards)), size):
            multipliers = np.zeros(len(guards))
            if active:
                indices = list(active)
                solution = np.linalg.lstsq(
                    gram[np.ix_(indices, indices)], dots[indices], rcond=None
                )[0]
                if np.min(solution) < -tolerance:
                    continue
                multipliers[indices] = np.maximum(solution, 0)
            residual = dots - gram @ multipliers
            violation = max(0.0, float(residual.max(initial=0)))
            complementarity = float(np.abs(multipliers * residual).max(initial=0))
            if violation > tolerance or complementarity > tolerance:
                continue
            distance = float(multipliers @ gram @ multipliers)
            if best is None or distance < best[0]:
                best = (distance, multipliers, violation, complementarity, active)
    if best is None:
        raise ValueError(
            "no numerically certified projection; do not use an unconstrained fallback"
        )
    distance, multipliers, violation, complementarity, active = best
    projected = p - multipliers @ guards
    norm = float(np.linalg.norm(projected))
    # An absolute floor is appropriate here only because the caller proposes unit
    # directions. Return the original norm as evidence, even when suppressing it.
    degenerate = norm <= 1e-8 * max(1.0, float(np.linalg.norm(p)))
    result = torch.from_numpy(np.zeros_like(projected) if degenerate else projected).to(proposed)
    return result, {
        "projection_norm": norm,
        "distance_squared": distance,
        "degenerate": degenerate,
        "zero_guards": int((~nonzero).sum()),
        "active_guard_indices": np.flatnonzero(nonzero)[list(active)].tolist(),
        "max_normalized_violation_float64": violation,
        "complementarity_float64": complementarity,
    }
