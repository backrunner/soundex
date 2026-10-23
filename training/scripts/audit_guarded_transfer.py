#!/usr/bin/env python3
"""P41 reversible guarded-update and recording-disjoint TRAIN transfer audit.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_probe import probe_inputs
from models.fit_diagnostic import BURN
from models.guarded_direction import project_direction
from models.restoration_head import load_restoration_head
from scripts.audit_update_directions import gradient, metrics, restore, unit
from scripts.run_consistency_followup import read, registered


def transfer_index(schedule: dict, source_index: int, rows: dict) -> int:
    """Choose by schedule geometry only, never by losses or gradient alignment."""
    draws, specs = schedule["draws"], schedule["specs"]

    def keys(index: int, field: str) -> set:
        return {rows[specs[j]["row_id"]][field] for j in draws[index]}

    for offset in range(len(draws)):
        index = (source_index + len(draws) // 2 + offset) % len(draws)
        if all(
            not keys(source_index, field) & keys(index, field)
            for field in ("track_id", "source_checksum")
        ):
            return index
    raise ValueError("no recording/source-disjoint transfer draw")


def history(model: torch.nn.Module, objective: object, context: torch.Tensor) -> torch.Tensor:
    _, _, condition = probe_inputs(objective.batch, context)
    with torch.no_grad():
        _, state = model(
            objective.batch["x"][:, :, :BURN],
            context[:, :BURN],
            condition[:, :BURN],
            model.initial_state(len(context), device=context.device),
        )
    return state.detach().clone()


def measured(model: torch.nn.Module, item: tuple, state: torch.Tensor | None = None) -> dict:
    with torch.no_grad():
        values, _, _ = metrics(model, *item, state)
    result = {k: float(v) for k, v in values.items()}
    if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
        raise ValueError("nonfinite metric")
    return result


def probe(model: torch.nn.Module, checkpoint: dict, source: tuple, transfer: tuple) -> dict:
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    state = history(model, *source)
    values, direct, pair = metrics(model, *source, state)
    base = {"source": {k: float(v.detach()) for k, v in values.items()}}
    base["transfer"] = measured(model, transfer)
    gradients = {k: gradient(v, params) for k, v in values.items()}
    denominator = 1 + direct["per_example"].detach()
    amp = gradient(0.5 * (direct["components"]["amplitude"] / denominator)[pair].mean(), params)
    comp = gradient(0.05 * (direct["components"]["complex"] / denominator)[pair].mean(), params)
    bisector = -unit(unit(amp) + unit(comp))
    projected, certificate = project_direction(bisector, torch.stack(list(gradients.values())))
    optimizer = torch.optim.Adam(params)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    offset = 0
    for p in params:
        p.grad = gradients["total"][offset : offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    optimizer.step()
    adam_delta = torch.cat(
        [(p.detach() - b).flatten() for p, b in zip(params, original, strict=True)]
    )
    restore(params, original)
    for p in params:
        p.grad = None
    directions = {
        "stored_adam": unit(adam_delta),
        "full_bisector": bisector,
        "guarded_bisector": unit(projected),
    }
    results = {}
    try:
        for name, direction in directions.items():
            derivatives = {k: float(v.dot(direction)) for k, v in gradients.items()}
            normalized = {k: float(unit(v).dot(direction)) for k, v in gradients.items()}
            points, fixed = [], []
            for radius in (-0.001, 0.001, 0.01):
                restore(params, original, radius * direction)
                if radius != 0.01:
                    fixed.append(measured(model, source, state))
                if radius < 0:
                    continue
                responses = {
                    "source": measured(model, source),
                    "transfer": measured(model, transfer),
                }
                points.append(
                    {
                        "radius": radius,
                        "values": responses,
                        "changes": {
                            role: {k: v - base[role][k] for k, v in response.items()}
                            for role, response in responses.items()
                        },
                    }
                )
            results[name] = {
                "norm": float(direction.norm()),
                "source_derivatives": derivatives,
                "source_normalized_derivatives": normalized,
                "source_fixed_history_central_slope": {
                    k: (fixed[1][k] - fixed[0][k]) / 0.002 for k in values
                },
                "source_fixed_history_values": fixed,
                "points": points,
            }
    finally:
        restore(params, original)
    if not all(torch.equal(p, b) for p, b in zip(params, original, strict=True)):
        raise ValueError("frozen endpoint changed")
    return {"base": base, "guard_certificate": certificate, "directions": results}


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    study, output = args.study.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("fresh output required")
    config, endpoints = read(study / "registration.json"), read(study / "review-registration.json")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("completed study required")
    selected = {k: v for k, v in endpoints.items() if k.startswith("control-")}
    indices = [int(i * (config["steps"] - 1) / 15) for i in range(16)]
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    schedules = {
        str(seed): read(Path(config["schedule_files"][str(seed)])) for seed in config["seeds"]
    }
    pairs = {
        seed: [{"source": i, "transfer": transfer_index(schedule, i, rows)} for i in indices]
        for seed, schedule in schedules.items()
    }
    for seed, schedule in schedules.items():
        for pair in pairs[seed]:
            for index in pair.values():
                for j in schedule["draws"][index]:
                    row = rows[schedule["specs"][j]["row_id"]]
                    if row["split"] != "train" or row["held_out"]:
                        raise ValueError("non-training recording in diagnostic")
    root = Path(__file__).resolve().parents[1]
    hashes = dict(config["input_hashes"])
    for name, digest in config["source_hashes"].items():
        if name.startswith(("models/", "data/")):
            hashes[str(root / name)] = digest
    executed = [
        Path(__file__),
        root / "models/guarded_direction.py",
        root / "scripts/audit_update_directions.py",
    ]
    for path in [
        *executed,
        study / "registration.json",
        study / "review-registration.json",
        *(Path(p) for p in config["schedule_files"].values()),
    ]:
        hashes[str(path.resolve())] = sha256_file(path)
    for endpoint in selected.values():
        hashes[endpoint["path"]] = endpoint["sha256"]
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift: {name}")
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in executed:
        shutil.copy2(path, snapshot / path.name)
    registered(
        {
            "scope": "P41 reversible source/transfer TRAIN diagnostics; no saved updates or independent generalization claim",
            "pairs": pairs,
            "hashes": hashes,
            "endpoints": selected,
            "directions": ["stored_adam", "full_bisector", "guarded_bisector"],
            "radii_l2": [0.001, 0.01],
            "guards": "all eight unchanged P40 raw metrics, source batch only; simultaneous Euclidean projection, normalize after projection; no line search",
            "transfer": "first schedule draw from half-cycle offset with no source checksum or track overlap with its source draw; never chosen by score; seen during original training",
            "history": "gradients/fixed-history central slopes use original detached warmup; actual responses recompute warmup for source and transfer",
            "decision": "report every metric, source/transfer, radius and seed; no training launch from this diagnostic alone; a matched training proposal requires both spectrum progress and protected finite-step transfer, not only first-order feasibility",
        },
        output / "registration.json",
    )
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    reports = {}
    for seed, schedule in schedules.items():
        pool = LazyPool(schedule["specs"], clips)
        label = f"control-{config['steps']}-{seed}"
        checkpoint = torch.load(selected[label]["path"], map_location="cpu", weights_only=False)
        model = load_restoration_head(checkpoint).eval()
        cases = []
        for pair in pairs[seed]:
            batches = {role: pool.select(schedule["draws"][i]) for role, i in pair.items()}
            case = probe(model, checkpoint, batches["source"], batches["transfer"])
            samples = {
                role: [
                    {"spec": schedule["specs"][j], "row": rows[schedule["specs"][j]["row_id"]]}
                    for j in schedule["draws"][index]
                ]
                for role, index in pair.items()
            }
            cases.append({"indices": pair, "samples": samples, **case})
            write_json_report(cases, output / f"{label}.json")
            print(label, pair, "complete", flush=True)
        reports[label] = cases
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift after probe: {name}")
    write_json_report(
        {
            "status": "complete",
            "cases": reports,
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
