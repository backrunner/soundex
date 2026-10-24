"""P49 fit-only finite-step calibration, without persistent optimizer updates.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from evaluation.batch_update_probe import average
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import metrics
from scripts.run_adapter_pilot import optimizer_for

FACTORS = tuple(0.25**i for i in range(10))
ARMIJO = 1e-4
FIXED_SCALE = 0.001


@torch.no_grad()
def place(parameters, original, delta, scale):
    for p, start, change in zip(parameters, original, delta, strict=True):
        p.copy_(start if scale == 0 else start + scale * change)


def backtrack(parameters, original, delta, derivative, baseline, evaluate):
    """Select using fit risk only; always restore parameters, including on error.

    Numerical floor is inherited from P42's eight float32 epsilons, not an
    audibility budget. Guard data cannot enter this interface. A no-op is explicit.
    """
    if not math.isfinite(derivative) or not math.isfinite(baseline):
        raise ValueError("finite local derivative and risk required")
    floor = 8 * torch.finfo(torch.float32).eps * max(abs(baseline), 1e-12)
    points, selected = [], 0.0
    try:
        if derivative < 0:
            for scale in FACTORS:
                place(parameters, original, delta, scale)
                values = evaluate()
                current = values["total"]
                if not math.isfinite(current):
                    raise ValueError("nonfinite fit risk")
                required = max(floor, -ARMIJO * scale * derivative)
                changed = any(
                    not torch.equal(p, b) for p, b in zip(parameters, original, strict=True)
                )
                accepted = changed and current <= baseline - required
                points.append(
                    {
                        "scale": scale,
                        "fit": values,
                        "required_decrease": required,
                        "accepted": accepted,
                        "changed": changed,
                    }
                )
                if accepted:
                    selected = scale
                    break
    finally:
        place(parameters, original, delta, 0)
    return {"selected_scale": selected, "numerical_floor": floor, "points": points}


def calibrate(model, fit_items):
    """One fresh P48 Adam displacement; four fit batches and no guard access."""
    if len(fit_items) != 4:
        raise ValueError("four equally weighted fit batches required")
    parameters = list(model.adapter.parameters())
    original = [p.detach().clone() for p in parameters]
    optimizer = optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    before = []
    for item in fit_items:
        values, _, _ = metrics(model, *item)
        before.append({k: float(v.detach()) for k, v in values.items()})
        (values["total"] / 4).backward()
    gradients = [p.grad.detach().clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach() - start for p, start in zip(parameters, original, strict=True)]
    if not all(torch.isfinite(d).all() for d in delta):
        place(parameters, original, delta, 0)
        raise ValueError("nonfinite Adam proposal")
    derivative = float(sum((g * d).sum() for g, d in zip(gradients, delta, strict=True)))
    place(parameters, original, delta, 0)
    optimizer.zero_grad(set_to_none=True)
    base = average(before)

    def evaluate():
        return average([measured(model, item) for item in fit_items])

    controls = {}
    try:
        for label, scale in (("adam", 1.0), ("fixed", FIXED_SCALE)):
            place(parameters, original, delta, scale)
            controls[label] = {"scale": scale, "fit": evaluate()}
        result = backtrack(parameters, original, delta, derivative, base["total"], evaluate)
    finally:
        place(parameters, original, delta, 0)
    if not all(torch.equal(p, b) for p, b in zip(parameters, original, strict=True)):
        raise ValueError("calibration changed frozen start")
    return {
        "base": base,
        "controls": controls,
        "search": result,
        "gradient_norm": float(norm),
        "derivative": derivative,
        "displacement_l2": float(sum(d.square().sum() for d in delta).sqrt()),
    }, {name: d for (name, _), d in zip(model.adapter.named_parameters(), delta, strict=True)}
