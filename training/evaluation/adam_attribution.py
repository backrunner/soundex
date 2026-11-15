"""Algebraic Adam attribution in canonical parameter units, without moment resets.

SPDX-License-Identifier: Apache-2.0
"""

import torch


def components(moment, variance, old_gradient, routed_gradient, *, lr, betas, eps, step):
    """Separate numerator sources and the auxiliary's denominator interaction.

    Inputs are gradients actually supplied to Adam, with common initial moments.
    This is a local first-step diagnostic, not an alternate optimizer or gradient
    projection. An effective gradient difference can include FP32 accumulation.
    """
    m, v, old, routed = [x.double() for x in (moment, variance, old_gradient, routed_gradient)]
    b1, b2 = betas
    if step < 1 or lr <= 0 or eps <= 0 or not (0 <= b1 < 1 and 0 <= b2 < 1):
        raise ValueError("valid persistent Adam hyperparameters required")
    if (v < 0).any() or not all(torch.isfinite(x).all() for x in (m, v, old, routed)):
        raise ValueError("finite gradients/moments and nonnegative variance required")
    scale = -lr / (1 - b1**step)
    denominator = (b2 * v + (1 - b2) * routed.square()).div(1 - b2**step).sqrt() + eps
    old_denominator = (b2 * v + (1 - b2) * old.square()).div(1 - b2**step).sqrt() + eps
    inherited = scale * b1 * m / denominator
    current_old = scale * (1 - b1) * old / denominator
    extra = scale * (1 - b1) * (routed - old) / denominator
    old_numerator = scale * (b1 * m + (1 - b1) * old)
    old_step = old_numerator / old_denominator
    denominator_effect = old_numerator * (1 / denominator - 1 / old_denominator)
    return {
        "inherited": inherited,
        "current_old": current_old,
        "current_extra": extra,
        "routed_step": inherited + current_old + extra,
        "old_step": old_step,
        "increment_numerator": extra,
        "increment_denominator": denominator_effect,
        "increment": extra + denominator_effect,
    }


def summarize(vectors, old_gradient, extra_gradient):
    result = {}
    for name, value in vectors.items():
        result[name] = {
            "l2": float(value.norm()),
            "dot_old": float((value * old_gradient.double()).sum()),
            "dot_extra": float((value * extra_gradient.double()).sum()),
        }
    n, d = vectors["increment_numerator"], vectors["increment_denominator"]
    result["increment_cancellation_ratio"] = (
        float((n + d).norm() / (n.norm() + d.norm())) if float(n.norm() + d.norm()) else None
    )
    return result
