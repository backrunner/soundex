"""Common raw review of every frozen P45 endpoint, including unchanged controls.

SPDX-License-Identifier: Apache-2.0
"""

from collections import defaultdict

import numpy as np
import torch

from evaluation.fidelity_calibration import calibrate
from models.direct_spectrum_objective import direct_spectrum_loss
from models.restoration_head import predict_budget


@torch.no_grad()
def review_case(model: torch.nn.Module, item: tuple) -> dict:
    result = calibrate(model, item)
    prediction, _ = predict_budget(model, item[0].batch, item[1])
    direct = direct_spectrum_loss(prediction, item[0].batch)
    result["direct"] = {k: v.tolist() for k, v in direct["components"].items()}
    return result


def group_values(cases: list) -> dict:
    values = defaultdict(lambda: defaultdict(list))
    for case in cases:
        samples = case["samples"]["calibration"]
        if case["identity"] != [False] * len(samples) + [True] * len(samples):
            raise ValueError("invalid paired/identity layout")
        for i, identity in enumerate(case["identity"]):
            sample = samples[i % len(samples)]
            group = sample["row"]["corpus"] + ("/identity" if identity else "/paired")
            metrics = {
                **case["paths"]["frozen_model"],
                **{"direct_" + k: v for k, v in case["direct"].items()},
            }
            for key, array in metrics.items():
                values[group][key].append(array[i])
    return values


def compare(current: list, reference: list) -> dict:
    if [c["samples"] for c in current] != [c["samples"] for c in reference]:
        raise ValueError("review examples differ")
    old, new = group_values(reference), group_values(current)
    result = {}
    for group, metrics in new.items():
        result[group] = {}
        for key, values in metrics.items():
            a, b = np.array(old[group][key]), np.array(values)
            delta = b - a
            result[group][key] = {
                "count": len(a),
                "improved": int((delta < 0).sum()),
                "unchanged": int((delta == 0).sum()),
                "worsened": int((delta > 0).sum()),
                "reference_mean": float(a.mean()),
                "current_mean": float(b.mean()),
                "mean_change": float(delta.mean()),
                "max_increase": float(delta.max()),
                "relative_mean_change": float(delta.mean() / a.mean()) if a.mean() else None,
            }
    return result


def assessment(reports: dict, starts: list[str]) -> dict:
    result = {}
    for start in starts:
        control = reports[f"{start}-stored_adam"]["cases"]
        for label, report in reports.items():
            if label.startswith(start + "-"):
                result[label] = {
                    "against_start": compare(report["cases"], reports[start]["cases"]),
                    "against_stored_adam": compare(report["cases"], control),
                }
    return result
