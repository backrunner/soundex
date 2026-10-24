"""P50 globally separated finite-bank roles and cumulative small Adam updates.

SPDX-License-Identifier: Apache-2.0
"""

from collections import defaultdict

import numpy as np
import torch

from data.causal_study import digest
from evaluation.adapter_scale import FIXED_SCALE, place
from evaluation.batch_update_probe import average
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import metrics

STEPS = 128
ARMS = ("persistent", "reset")


def schedule_roles(specs, rows, steps=STEPS, review_batches=32):
    """Partition connected recording/source aliases before any scores are read."""
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]

    for spec in specs:
        row = rows[spec["row_id"]]
        if row["split"] != "train" or row["held_out"]:
            raise ValueError("TRAIN bank required")
        a, b = root("track:" + row["track_id"]), root("source:" + row["source_checksum"])
        parents[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for key in parents:
        members[root(key)].append(key)
    roles = {
        key: "review" if int(digest(sorted(values))[:16], 16) % 5 == 0 else "fit"
        for key, values in members.items()
    }
    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, spec in enumerate(specs):
        row = rows[spec["row_id"]]
        component = root("track:" + row["track_id"])
        pools[roles[component]][row["corpus"]][component].append(index)
    result = {}
    for role, count in (("fit", steps * 4), ("review", review_batches)):
        rng = np.random.default_rng(20261050 if role == "fit" else 20261051)
        queues, draws = {}, []
        for _ in range(count):
            draw = []
            for corpus, size in (("music_library", 4), ("speech_library", 1)):
                options = pools[role][corpus]
                if len(options) < size:
                    raise ValueError("insufficient independent recording components")
                chosen = set()
                for _ in range(size):
                    if not queues.get(corpus):
                        queues[corpus] = rng.permutation(sorted(options)).tolist()
                    # A refilled queue may start with a recording already in this draw.
                    position = next(i for i, key in enumerate(queues[corpus]) if key not in chosen)
                    key = queues[corpus].pop(position)
                    chosen.add(key)
                    draw.append(int(rng.choice(options[key])))
            draws.append(draw)
        result[role] = draws
    for field in ("track_id", "source_checksum"):
        values = {
            role: {rows[specs[i]["row_id"]][field] for draw in draws for i in draw}
            for role, draws in result.items()
        }
        if values["fit"] & values["review"]:
            raise ValueError("global fit/review overlap")
    return result


def update(model, items, optimizer):
    """Keep Adam moments; apply exactly P49's 0.001 finite displacement.

    Caller supplies a fresh optimizer only for the reset control. There is no
    guard access, veto, adaptive step selection or early stopping here.
    """
    if len(items) != 4:
        raise ValueError("four fit batches required")
    parameters = list(model.adapter.parameters())
    original = [p.detach().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    values = []
    for item in items:
        scores, _, _ = metrics(model, *item)
        values.append({key: float(value.detach()) for key, value in scores.items()})
        (scores["total"] / 4).backward()
    gradients = [p.grad.detach().clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(parameters, original, strict=True)]
    if not all(torch.isfinite(d).all() for d in delta):
        place(parameters, original, delta, 0)
        raise ValueError("nonfinite Adam proposal; optimizer state must not be reused")
    derivative = float(sum((g * d).sum() for g, d in zip(gradients, delta, strict=True)))
    place(parameters, original, delta, FIXED_SCALE)
    changed = sum(
        int(torch.count_nonzero(p.detach() != old))
        for p, old in zip(parameters, original, strict=True)
    )
    optimizer.zero_grad(set_to_none=True)
    return {
        "fit_before": average(values),
        "fit_after": average([measured(model, item) for item in items]),
        "gradient_norm": float(norm),
        "adam_derivative": derivative,
        "changed_parameters": changed,
        "adam_displacement_l2": float(sum(d.square().sum() for d in delta).sqrt()),
        "applied_displacement_l2": float(
            sum(
                (p.detach() - old).square().sum()
                for p, old in zip(parameters, original, strict=True)
            ).sqrt()
        ),
    }
