"""P56 paired/identity attribution under unchanged original-reference losses.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from evaluation.longform_review import aliases
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss, supervised_waves
from models.restoration_head import predict_budget


def assert_reserved_isolation(selected, universe, reserved):
    blocked = set().union(*(aliases(row) for row in reserved))
    previous = -1
    while len(blocked) != previous:
        previous = len(blocked)
        for row in universe:
            keys = aliases(row)
            if keys & blocked:
                blocked.update(keys)
    if any(aliases(row) & blocked for row in selected):
        raise ValueError("reserved evaluation source/work alias in TRAIN probe")
    if any(row["split"] != "train" or row["held_out"] for row in selected):
        raise ValueError("non-TRAIN probe")


def terms(model, item):
    batch = item[0].batch
    prediction, _ = predict_budget(model, batch, item[1])
    direct = direct_spectrum_loss(prediction, batch)
    rendered = multires_spectrum_loss(prediction, batch)
    identity = batch["identity"].bool()
    values = {}
    for role, mask in (("paired", ~identity), ("identity", identity)):
        values[role + "_direct"] = direct["per_example"][mask].log1p().mean()
        values[role + "_rendered"] = rendered["per_example"][mask].log1p().mean()
    return values, direct, rendered, prediction


def objective(values, identity_weight):
    if identity_weight not in (0.0, 1.0):
        raise ValueError("registered identity ablation weights are zero and one")
    # Keep the paired coefficient unchanged: zero is a diagnostic ablation,
    # never an assertion that clean fidelity is optional or a release criterion.
    return 0.5 * (
        values["paired_direct"]
        + values["paired_rendered"]
        + identity_weight * (values["identity_direct"] + values["identity_rendered"])
    )


def gradients(model, items):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    sums, reports = {}, []
    for item in items:
        values = terms(model, item)[0]
        current = {}
        for key, value in values.items():
            parts = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).detach().flatten()
                    for p, g in zip(parameters, parts, strict=True)
                ]
            )
            if not torch.isfinite(vector).all():
                raise ValueError("nonfinite component gradient")
            sums[key] = sums.get(key, 0) + vector / len(items)
            current[key] = vector
        reports.append(geometry(current))
    return sums, {"pooled": geometry(sums), "batches": reports}, named


def geometry(parts):
    pair = (parts["paired_direct"] + parts["paired_rendered"]).double()
    clean = (parts["identity_direct"] + parts["identity_rendered"]).double()
    a, b = float(pair.norm()), float(clean.norm())
    return {
        "paired_norm": a,
        "identity_norm": b,
        "identity_to_paired_norm": b / a if a else None,
        "paired_identity_cosine": float(pair.dot(clean)) / (a * b) if a * b else None,
        "component_norms": {k: float(v.double().norm()) for k, v in parts.items()},
    }


@torch.no_grad()
def measure(model, items):
    cases = []
    for item in items:
        values, direct, rendered, prediction = terms(model, item)
        batch = item[0].batch
        wave, reference = supervised_waves(prediction, batch)
        rms = (wave - reference).square().mean(-1).sqrt()
        role_scores = {}
        for role, mask in (
            ("paired", ~batch["identity"].bool()),
            ("identity", batch["identity"].bool()),
        ):
            role_scores[role] = {
                "risk": float(values[role + "_direct"] + values[role + "_rendered"]),
                "direct": float(direct["per_example"][mask].mean()),
                "mr": float(rendered["per_example"][mask].mean()),
                "pcm_rms": float(rms[mask].mean()),
                **{k: float(v[mask].mean()) for k, v in direct["components"].items()},
            }
        cases.append(role_scores)
    mean = {
        role: {k: sum(c[role][k] for c in cases) / len(cases) for k in cases[0][role]}
        for role in ("paired", "identity")
    }
    if not all(math.isfinite(v) for m in mean.values() for v in m.values()):
        raise ValueError("nonfinite measurement")
    return {"mean": mean, "cases": cases}
