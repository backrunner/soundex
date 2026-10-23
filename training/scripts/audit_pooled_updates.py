#!/usr/bin/env python3
"""P42 pooled updates with separate fit, step-guard and transfer recordings.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.batch_update_probe import (
    POLICIES,
    RADII,
    choose_step,
    directions_for,
    evaluate_roles,
    pcm_errors,
    select_roles,
    step_policy,
)
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head
from scripts.audit_update_directions import restore
from scripts.run_consistency_followup import read, registered
from scripts.summarize_guarded_transfer import coverage


def probe(
    model: torch.nn.Module,
    items: list[tuple],
    direction_builder: Callable = directions_for,
) -> dict:
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    base = evaluate_roles(model, items)
    base_pcm = pcm_errors(model, items[5])
    directions, certificate = direction_builder(model, items[:4])
    results = {}
    try:
        for name, direction in directions.items():
            points = []
            for radius in RADII:
                restore(params, original, radius * direction)
                current = evaluate_roles(model, items)
                points.append(
                    {
                        "radius": radius,
                        "values": current,
                        "transfer_pcm": pcm_errors(model, items[5]),
                        "policies": {
                            key: step_policy(base, current, budget)
                            for key, budget in POLICIES.items()
                        },
                    }
                )
            results[name] = {
                "norm": float(direction.norm()),
                "points": points,
                "selected": {key: choose_step(points, key) for key in POLICIES},
            }
    finally:
        restore(params, original)
    if not all(torch.equal(p, b) for p, b in zip(params, original, strict=True)):
        raise ValueError("frozen model changed")
    return {
        "base": base,
        "base_transfer_pcm": base_pcm,
        "certificate": certificate,
        "directions": results,
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    prior, output = args.prior.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("fresh output directory required")
    previous = read(prior / "registration.json")
    if read(prior / "summary.json")["status"] != "complete":
        raise ValueError("completed P41 evidence required")
    hashes = dict(previous["hashes"])
    study_path = next(
        Path(p).parent for p in hashes if p.endswith("study/review-registration.json")
    )
    config = read(study_path / "registration.json")
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    schedules = {str(seed): read(Path(path)) for seed, path in config["schedule_files"].items()}
    groups = {}
    for seed, schedule in schedules.items():
        excluded = {i for pair in previous["pairs"][seed] for i in pair.values()}
        groups[seed] = [
            select_roles(schedule, int((i + 0.5) * config["steps"] / 16), rows, excluded)
            for i in range(16)
        ]
    root = Path(__file__).resolve().parents[1]
    executed = [
        Path(__file__),
        root / "evaluation/batch_update_probe.py",
        root / "scripts/summarize_guarded_transfer.py",
        root / "scripts/audit_guarded_transfer.py",
    ]
    for path in [*executed, prior / "registration.json", prior / "summary.json"]:
        hashes[str(path.resolve())] = sha256_file(path)
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input/source drift: {path}")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for path in executed:
        shutil.copy2(path, output / "source" / path.name)
    registered(
        {
            "scope": "P42 reversible diagnostic on frozen small P38 controls; no trained candidate or independent test",
            "hashes": hashes,
            "endpoints": previous["endpoints"],
            "groups": groups,
            "roles": "first four draws estimate gradients; fifth checks step; sixth evaluates transfer only. All six mutually track/source disjoint within a case; draws from original TRAIN schedule, excluding P41 draw indices. Across cases recordings can recur.",
            "directions": ["single_guarded", "pooled_bisector", "pooled_guarded"],
            "radii_l2_descending": RADII,
            "policies": POLICIES,
            "selection": "first passing radius: pooled-fit amplitude and MR decrease beyond 8 float32 eps of metric scale; all eight fit metrics and five guard metrics bounded by the registered relative budget plus numerical allowance; reject all yields explicit no-op, never recovery credit; transfer never selects the step",
            "pooling": "average four raw metric gradients and four exact P40 amplitude/complex carriers BEFORE normalization; guards protect means, not every recording. Single control uses first batch but same four-batch finite-step assessment.",
            "history": "detached warmup for gradients; recompute real warmup for all finite-step responses",
            "pcm": "raw OLA error against original paired PCM on common supervised support, no deployed DSP; full-band and >=8k one-sided Parseval RMS with correct DC/Nyquist factors; zero dBFS encoded null",
            "decision": "compare all cases, seeds and fixed radii; separate accepted-only results from all-case no-op-inclusive outcomes. Strict and 0.1% policies are diagnostic sensitivity controls, not audibility or production gates. No full training launch from acceptance counts alone.",
        },
        output / "registration.json",
    )
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    reports, coverage_reports = {}, {}
    for seed, schedule in schedules.items():
        pool = LazyPool(schedule["specs"], clips)
        label = f"control-{config['steps']}-{seed}"
        checkpoint = torch.load(
            previous["endpoints"][label]["path"], map_location="cpu", weights_only=False
        )
        model = load_restoration_head(checkpoint).eval()
        cases = []
        for indices in groups[seed]:
            items = [pool.select(schedule["draws"][index]) for index in indices]
            samples = [
                [
                    {"spec": schedule["specs"][j], "row": rows[schedule["specs"][j]["row_id"]]}
                    for j in schedule["draws"][index]
                ]
                for index in indices
            ]
            roles = {
                "fit": [sample for batch in samples[:4] for sample in batch],
                "guard": samples[4],
                "transfer": samples[5],
            }
            cases.append({"indices": indices, "samples": roles, **probe(model, items)})
            write_json_report(cases, output / f"{label}.json")
            print(label, indices[0], "complete", flush=True)
        reports[label] = cases
        coverage_reports[label] = {
            role: coverage(cases, role) for role in ("fit", "guard", "transfer")
        }
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input/source drift after probe: {path}")
    write_json_report(
        {
            "status": "complete",
            "registration_sha256": sha256_file(output / "registration.json"),
            "cases": reports,
            "coverage": coverage_reports,
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
