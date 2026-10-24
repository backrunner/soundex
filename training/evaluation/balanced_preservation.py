"""P53 fixed corpus-aware calibration of preservation gradients.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from evaluation.adapter_scale import FIXED_SCALE, place
from evaluation.batch_update_probe import average
from evaluation.physical_preservation import band_errors
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import gradient, metrics

ARMS = ("rms-physical", "balanced-physical", "balanced-logit")
SHARES = {"music_library": 0.8, "speech_library": 0.2}
FRACTION = 0.1


class CorpusPool:
    """Attach source metadata without changing the paired objective or tensors."""

    def __init__(self, pool):
        self.pool = pool

    def select(self, indices):
        item = self.pool.select(indices)
        item[0].corpora = [self.pool.rows[i]["corpus"] for i in indices]
        return item


def grouped_penalties(model, item):
    batch = item[0].batch
    corpora = item[0].corpora
    identity = batch["identity"].bool()
    size = len(corpora)
    if identity.tolist() != [False] * size + [True] * size:
        raise ValueError("aligned paired/identity metadata required")
    if set(corpora) != set(SHARES):
        raise ValueError("both known corpora required")
    captured = []
    handle = model.adapter.register_forward_hook(lambda _m, _args, v: captured.append(v))
    try:
        prediction, _ = predict_budget(model, batch, item[1])
    finally:
        handle.remove()
    if len(captured) != 2 or captured[0].shape[1] != 80:
        raise ValueError("separate 80-frame causal warmup required")
    logits = torch.cat(captured, 1)[:, 89:]
    if not logits.shape[1]:
        raise ValueError("supervised frames required")
    bands = band_errors(*supervised_waves(prediction, batch), batch["rate"])
    per_clip = {
        "logit": logits[identity].double().square().flatten(1).mean(-1),
        "physical": bands[identity, 1:].sum(-1),
    }
    terms = {"physical": per_clip["physical"].mean()}
    for corpus in SHARES:
        mask = torch.tensor([c == corpus for c in corpora], device=identity.device)
        for key, value in per_clip.items():
            terms[f"{key}/{corpus}"] = value[mask].mean()
    return terms


def coefficients(gradients):
    """Match RMS batch-gradient norms, retaining inter-corpus cross terms.

    Inputs are [calibration batch, parameter] gradients at the frozen parent.
    Corpus mean terms first receive 80:20 normalized-gradient shares. A common
    final multiplier matches their combined RMS norm to 10% of the unchanged
    objective. This is an initial calibration, not a bound on individual batches
    or future updates. No audio-level division or adaptive gradient normalization.
    """
    rms = {k: float(v.square().sum(-1).mean().sqrt()) for k, v in gradients.items()}
    if not all(math.isfinite(v) for v in rms.values()) or rms["common"] <= 0:
        raise ValueError("finite nonzero common gradient required")
    weights, diagnostics = {}, {}
    for arm in ARMS:
        kind = arm.split("-")[-1]
        keys = ["physical"] if arm == "rms-physical" else [f"{kind}/{c}" for c in SHARES]
        if any(rms[k] == 0 for k in keys):
            raise ValueError("inactive preservation gradient; retain preflight, do not train")
        preliminary = {
            k: (1.0 if arm == "rms-physical" else SHARES[k.split("/")[1]]) / rms[k] for k in keys
        }
        combined = sum(preliminary[k] * gradients[k] for k in keys)
        norm = float(combined.square().sum(-1).mean().sqrt())
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("cancelled preservation gradients")
        factor = FRACTION * rms["common"] / norm
        weights[arm] = {k: factor * v for k, v in preliminary.items()}
        applied = combined * factor
        common = gradients["common"]
        norms = common.norm(dim=-1)
        diagnostics[arm] = {
            "combined_rms_to_common": float(applied.square().sum(-1).mean().sqrt()) / rms["common"],
            "component_rms_to_common": {k: weights[arm][k] * rms[k] / rms["common"] for k in keys},
            "batch_ratios": [
                float(a / b) if b else None
                for a, b in zip(applied.norm(dim=-1), norms, strict=True)
            ],
            "rms_inner_product_cosine": float((applied * common).sum(-1).mean())
            / (FRACTION * rms["common"] ** 2),
        }
    return {"weights": weights, "gradient_rms": rms, "diagnostics": diagnostics}


def calibrate(model, items):
    params = list(model.adapter.parameters())
    gradients, values = {}, []
    for item in items:
        counts = {c: item[0].corpora.count(c) / len(item[0].corpora) for c in SHARES}
        if counts != SHARES:
            raise ValueError("registered 80:20 calibration mix required")
        terms = {"common": metrics(model, *item)[0]["total"], **grouped_penalties(model, item)}
        values.append({k: float(v.detach()) for k, v in terms.items()})
        for key, value in terms.items():
            gradients.setdefault(key, []).append(gradient(value, params).detach().double())
    if not values:
        raise ValueError("fit calibration batches required")
    gradients = {k: torch.stack(v) for k, v in gradients.items()}
    return {
        **coefficients(gradients),
        "pooled_gradient_norms": {k: float(g.mean(0).norm()) for k, g in gradients.items()},
        "mean_values": average(values),
        "batches": len(items),
        "shares": SHARES,
    }


def update(model, items, optimizer, arm, calibration):
    if arm not in ARMS or len(items) != 4:
        raise ValueError("registered arm and four fit batches required")
    params = list(model.adapter.parameters())
    original = [p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    values, auxiliary = [], []
    for item in items:
        scores = metrics(model, *item)[0]
        values.append({k: float(v.detach()) for k, v in scores.items()})
        terms = grouped_penalties(model, item)
        term = sum(w * terms[k] for k, w in calibration["weights"][arm].items())
        auxiliary.append(float(term.detach()))
        ((scores["total"] + term) / 4).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(params, original, strict=True)]
    if not all(torch.isfinite(d).all() for d in delta):
        place(params, original, delta, 0)
        raise ValueError("nonfinite proposal; optimizer state must not be reused")
    place(params, original, delta, FIXED_SCALE)
    optimizer.zero_grad(set_to_none=True)
    return {
        "fit_before": average(values),
        "fit_after": average([measured(model, item) for item in items]),
        "weighted_auxiliary_before": sum(auxiliary) / 4,
        "gradient_norm": float(norm),
        "changed_parameters": sum(
            int(torch.count_nonzero(p.detach() != old))
            for p, old in zip(params, original, strict=True)
        ),
        "applied_displacement_l2": float(
            sum(
                (p.detach() - old).square().sum() for p, old in zip(params, original, strict=True)
            ).sqrt()
        ),
    }
