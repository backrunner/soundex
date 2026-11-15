"""Exact frozen parameter-block interventions and complete three-factor attribution.

SPDX-License-Identifier: Apache-2.0
"""

import math

import numpy as np

GROUPS = ("base.", "phase_head.0.", "phase_head.2.")


def hybrid(old, routed, mask):
    if mask not in range(8) or old.keys() != routed.keys():
        raise ValueError("matching complete states and a three-bit mask required")
    result = {}
    for name, value in old.items():
        groups = [i for i, prefix in enumerate(GROUPS) if name.startswith(prefix)]
        if (
            len(groups) != 1
            or value.shape != routed[name].shape
            or value.dtype != routed[name].dtype
        ):
            raise ValueError(f"unpartitioned or incompatible state: {name}")
        result[name] = (routed[name] if mask & (1 << groups[0]) else value).detach().clone()
    return result


def shapley(values):
    """All subset marginals, never a best-subset search. None remains undefined."""
    if len(values) != 8:
        raise ValueError("all eight interventions are required")
    if any(v is None for v in values):
        return {k: None for k in (*GROUPS, "nonadditivity")}
    if not np.isfinite(values).all():
        raise ValueError("finite metric values required")
    result = {}
    for i, group in enumerate(GROUPS):
        value = 0.0
        for mask in range(8):
            if not mask & (1 << i):
                n = mask.bit_count()
                weight = math.factorial(n) * math.factorial(2 - n) / 6
                value += weight * (values[mask | (1 << i)] - values[mask])
        result[group] = value
    total = values[7] - values[0]
    if abs(sum(result.values()) - total) > 1e-12 * max(1, max(abs(v) for v in values)):
        raise ValueError("factorial attribution does not conserve the observed change")
    result["nonadditivity"] = total - sum(values[1 << i] - values[0] for i in range(3))
    return result
