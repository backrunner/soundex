"""P42 evaluation of gradient pooling and finite-step diagnostic policy.

No runtime model, loss, or production optimizer uses these helpers.
SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from models.guarded_direction import project_direction
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import gradient, metrics, unit

PRIMARY = ("paired_direct_amplitude", "paired_mr")
CHECKS = (*PRIMARY, "paired_direct_complex", "paired_mr_dynamics", "identity_mr")
RADII = (0.01, 0.003, 0.001, 0.0003)
POLICIES = {"strict": 0.0, "relative_budget_0.1_percent": 0.001}


def select_roles(schedule: dict, anchor: int, rows: dict, excluded: set[int]) -> list[int]:
    """Six mutually recording/source-disjoint batches, selected without scores."""
    draws, specs = schedule["draws"], schedule["specs"]
    used = {key: set() for key in ("track_id", "source_checksum")}
    selected = []
    for role in range(6):
        for offset in range(len(draws)):
            index = (anchor + role * len(draws) // 6 + offset) % len(draws)
            if index in excluded or index in selected:
                continue
            entries = [rows[specs[j]["row_id"]] for j in draws[index]]
            if any(r["split"] != "train" or r["held_out"] for r in entries):
                raise ValueError("non-training recording in schedule")
            keys = {key: {r[key] for r in entries} for key in used}
            if any(used[key] & keys[key] for key in used):
                continue
            selected.append(index)
            for key in used:
                used[key].update(keys[key])
            break
        else:
            raise ValueError("insufficient recording-disjoint TRAIN roles")
    return selected


def average(items: list[dict]) -> dict:
    return {key: sum(item[key] for item in items) / len(items) for key in items[0]}


def gradients_for(model: torch.nn.Module, item: tuple) -> tuple[dict, torch.Tensor, torch.Tensor]:
    params = list(model.parameters())
    values, direct, pair = metrics(model, *item)
    gradients = {key: gradient(value, params).detach() for key, value in values.items()}
    denominator = 1 + direct["per_example"].detach()
    amp = gradient(0.5 * (direct["components"]["amplitude"] / denominator)[pair].mean(), params)
    comp = gradient(0.05 * (direct["components"]["complex"] / denominator)[pair].mean(), params)
    return gradients, amp.detach(), comp.detach()


def directions_for(model: torch.nn.Module, items: list[tuple]) -> tuple[dict, dict]:
    gradients, amps, comps = [], [], []
    for item in items:
        g, a, c = gradients_for(model, item)
        gradients.append(g)
        amps.append(a)
        comps.append(c)
    single = -unit(unit(amps[0]) + unit(comps[0]))
    single, single_certificate = project_direction(single, torch.stack(list(gradients[0].values())))
    pooled = average(gradients)
    bisector = -unit(unit(torch.stack(amps).mean(0)) + unit(torch.stack(comps).mean(0)))
    guarded, certificate = project_direction(bisector, torch.stack(list(pooled.values())))
    directions = {
        "single_guarded": unit(single),
        "pooled_bisector": bisector,
        "pooled_guarded": unit(guarded),
    }
    report = {
        "single_certificate": single_certificate,
        "pooled_certificate": certificate,
        "pooled_normalized_derivatives": {
            name: {key: float(unit(value).dot(direction)) for key, value in pooled.items()}
            for name, direction in directions.items()
        },
    }
    return directions, report


def error_rms(error: torch.Tensor, rates: torch.Tensor, low_hz: float = 0) -> torch.Tensor:
    """Unwindowed one-sided Parseval error RMS in physical PCM units."""
    length = error.shape[-1]
    spectrum = torch.fft.rfft(error.double()) / length
    sides = torch.full((spectrum.shape[-1],), 2.0, dtype=torch.float64, device=error.device)
    sides[0] = 1
    if length % 2 == 0:
        sides[-1] = 1
    hz = torch.arange(len(sides), device=error.device)[None] * rates.reshape(-1, 1) / length
    return (spectrum.abs().square() * sides * (hz >= low_hz)).sum(-1).sqrt()


def pcm_errors(model: torch.nn.Module, item: tuple) -> dict:
    objective, context = item
    with torch.no_grad():
        prediction, _ = predict_budget(model, objective.batch, context)
        wave, reference = supervised_waves(prediction, objective.batch)
        identity = objective.batch["identity"].bool()
        result = {}
        for low, band in ((0, "full"), (8000, "hf8k")):
            values = error_rms(wave - reference, objective.batch["rate"], low)
            for mask, role in ((identity, "identity"), (~identity, "paired")):
                selected = values[mask].tolist()
                result[f"{role}_{band}"] = {
                    "per_example_rms": selected,
                    "mean_rms": sum(selected) / len(selected),
                    "per_example_dbfs": [20 * math.log10(v) if v > 0 else None for v in selected],
                }
        return result


def evaluate_roles(model: torch.nn.Module, items: list[tuple]) -> dict:
    values = [measured(model, item) for item in items]
    return {
        "fit": average(values[:4]),
        "fit_batches": values[:4],
        "guard": values[4],
        "transfer": values[5],
    }


def step_policy(base: dict, current: dict, relative_budget: float) -> dict:
    """Fit spectral progress plus source/guard protection; never read transfer.

    Eight float32 epsilons of each metric's own scale are numerical allowance,
    not perceptual tolerance. The separate 0.1% policy is a fixed sensitivity
    control. No-op or sub-allowance spectral changes cannot count as progress.
    """
    if not 0 <= relative_budget <= 0.001:
        raise ValueError("unregistered step budget")

    def allowance(value: float) -> float:
        return 8 * torch.finfo(torch.float32).eps * max(abs(value), 1e-12)

    primary = all(
        current["fit"][key] < base["fit"][key] - allowance(base["fit"][key]) for key in PRIMARY
    )
    violations = []
    for role in ("fit", "guard"):
        keys = base[role] if role == "fit" else CHECKS
        for key in keys:
            old, new = base[role][key], current[role][key]
            if not math.isfinite(new) or new - old > relative_budget * abs(old) + allowance(old):
                violations.append({"role": role, "metric": key, "change": new - old})
    return {
        "accepted": primary and not violations,
        "spectral_progress": primary,
        "violations": violations,
    }


def choose_step(points: list[dict], policy: str) -> int | None:
    """Largest predeclared radius that passes, or a recorded no-op."""
    return next(
        (i for i, point in enumerate(points) if point["policies"][policy]["accepted"]), None
    )
