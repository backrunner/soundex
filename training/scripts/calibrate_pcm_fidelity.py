#!/usr/bin/env python3
"""P44 frozen PCM calibration on geometry-selected TRAIN intervals.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, allowed_starts, protected_intervals
from data.causal_study import load_sources
from evaluation.fidelity_calibration import BITS, calibrate
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head
from scripts.run_consistency_followup import read, registered
from scripts.summarize_guarded_transfer import coverage


def select_calibration(schedule: dict, rows: dict, protected_specs: list, count: int = 32) -> list:
    """No scores: scan from uniform offsets, exclude prior and newly chosen support."""
    protected = protected_intervals(protected_specs, rows)
    chosen = []
    for index in range(count):
        for offset in range(len(schedule["draws"])):
            draw = (int((index + 0.25) * len(schedule["draws"]) / count) + offset) % len(
                schedule["draws"]
            )
            specs = [schedule["specs"][j] for j in schedule["draws"][draw]]
            if all(
                rows[s["row_id"]]["split"] == "train"
                and not rows[s["row_id"]]["held_out"]
                and any(
                    a <= s["start"] <= b for a, b in allowed_starts(rows[s["row_id"]], protected)
                )
                for s in specs
            ):
                chosen.append(draw)
                for key, spans in protected_intervals(specs, rows).items():
                    protected[key].extend(spans)
                break
        else:
            raise ValueError("insufficient untouched diagnostic intervals")
    return chosen


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    study, prior, output = args.study.resolve(), args.prior.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("fresh output required")
    config, endpoints = read(study / "registration.json"), read(study / "review-registration.json")
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    old = read(prior / "summary.json")
    if old["status"] != "complete":
        raise ValueError("completed P43 required")
    protected = [*config["development"], *config["shifted"]]
    for report in (old, read(args.p41 / "summary.json")):
        protected.extend(
            sample["spec"]
            for cases in report["cases"].values()
            for case in cases
            for samples in case["samples"].values()
            for sample in samples
        )
    # P38 seeds share exactly the same schedule; one common calibration selection.
    if len(set(config["schedule_hashes"].values())) != 1:
        raise ValueError("common schedule required")
    schedule = read(Path(next(iter(config["schedule_files"].values()))))
    selected = select_calibration(schedule, rows, protected)
    root = Path(__file__).resolve().parents[1]
    sources = (
        sorted(root.glob("models/*.py"))
        + sorted(root.glob("data/*.py"))
        + [
            Path(__file__),
            root / "evaluation/fidelity_calibration.py",
            root / "scripts/summarize_guarded_transfer.py",
            root / "evaluation/reporting.py",
            root / "scripts/run_consistency_followup.py",
        ]
    )
    frozen = {k: v for k, v in endpoints.items() if k.startswith("control-")}
    hashes = dict(config["input_hashes"])
    for name, digest in config["source_hashes"].items():
        if name.startswith(("models/", "data/")) and sha256_file(root / name) != digest:
            raise ValueError(f"P38 source drift: {name}")
    for p in [
        *sources,
        study / "registration.json",
        study / "review-registration.json",
        prior / "summary.json",
        args.p41 / "summary.json",
        *(Path(p) for p in config["schedule_files"].values()),
    ]:
        hashes[str(p.resolve())] = sha256_file(p)
    for endpoint in frozen.values():
        hashes[endpoint["path"]] = endpoint["sha256"]
    for p, digest in hashes.items():
        if sha256_file(Path(p)) != digest:
            raise ValueError(f"input drift: {p}")
    output.mkdir(parents=True)
    for p in sources:
        destination = output / "source" / p.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    registered(
        {
            "scope": "P44 absolute fidelity calibration, no updates or acceptance threshold selection",
            "hashes": hashes,
            "endpoints": frozen,
            "indices": selected,
            "precision_bits": BITS,
            "precision_meaning": "rounding-only lattice, half-step 2^-bits; no clipping/dither, not audibility",
            "isolation": "full source-time support excludes P39/P41, P42/P43 and P38 development/shifted support across codecs/views/rates; selected draws mutually time-disjoint. Recordings can recur; original P38 TRAIN draws, not independent test.",
            "decision": "report all per-example absolute/relative errors, numerical roundtrips, precision rulers and coverage. No tolerance selected from candidate scores; no new weights or production qualification.",
        },
        output / "registration.json",
    )
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    pool, reports = LazyPool(schedule["specs"], clips), {}
    for label, endpoint in frozen.items():
        model = load_restoration_head(torch.load(endpoint["path"], weights_only=False)).eval()
        cases = []
        for index in selected:
            specs = [schedule["specs"][i] for i in schedule["draws"][index]]
            cases.append(
                {
                    "index": index,
                    "samples": {
                        "calibration": [{"spec": s, "row": rows[s["row_id"]]} for s in specs]
                    },
                    **calibrate(model, pool.select(schedule["draws"][index])),
                }
            )
            write_json_report(cases, output / f"{label}.json")
            print(label, index, "complete", flush=True)
        reports[label] = cases
    for p, digest in hashes.items():
        if sha256_file(Path(p)) != digest:
            raise ValueError(f"input drift after calibration: {p}")
    write_json_report(
        {
            "status": "complete",
            "registration_sha256": sha256_file(output / "registration.json"),
            "cases": reports,
            "coverage": {k: coverage(v, "calibration") for k, v in reports.items()},
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "prior", "p41", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
