#!/usr/bin/env python3
"""Check P17 receipts and apply the registered training-only interpretation rule."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report

TOLERANCE = 1e-6


def read(path):
    return json.loads(path.read_text())


def values(report):
    return {
        **report["weighted"],
        "auxiliary": report["auxiliary"],
        "combined": report["combined"],
    }


def native_delta(reference, candidate):
    return {
        role: {
            metric: float(
                np.mean(
                    [
                        candidate[i]["output_band_metrics"][metric]
                        - reference[i]["output_band_metrics"][metric]
                        for i in indices
                    ]
                )
            )
            for metric in ("high_sc", "high_flux", "high_energy_error_db", "band_0_4000_sc")
        }
        for role, indices in (("paired", range(5)), ("identity", range(5, 10)))
    }


def run(root):
    registration = read(root / "registration.json")
    if registration["rates"] != [0.001, 0.0001] or registration["milestones"] != [
        320,
        384,
        448,
        512,
    ]:
        raise ValueError("unexpected registered experiment")
    for name, expected in registration["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"input drift: {name}")
    source = root.parent / "source/training"
    for name, expected in registration["source_hashes"].items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"source drift: {name}")
    summary = read(root / "summary.json")
    if summary["registration_sha256"] != sha256_file(root / "registration.json"):
        raise ValueError("registration drift")
    results, native_records, report_hashes = [], [], {}
    for seed in registration["seeds"]:
        start_path = root / f"start-{seed}.json"
        start = read(start_path)
        report_hashes[str(start_path.relative_to(root))] = sha256_file(start_path)
        native_records.extend(start["native"])
        arms = {}
        for rate in registration["rates"]:
            directory = root / f"seed-{seed}-lr-{rate}"
            receipt = read(directory / "complete.json")
            for name, expected in receipt["files"].items():
                if sha256_file(directory / name) != expected:
                    raise ValueError(f"completed output drift: {directory / name}")
            report_hashes[str((directory / "complete.json").relative_to(root))] = sha256_file(
                directory / "complete.json"
            )
            reports = []
            for step in registration["milestones"]:
                evaluation = read(directory / f"evaluation-{step}.json")
                checkpoint = directory / f"step-{step}.pth"
                registered = read(directory / f"registration-{step}.json")
                if (
                    evaluation["step"] != step
                    or registered["step"] != step
                    or not (
                        evaluation["checkpoint_sha256"]
                        == registered["checkpoint_sha256"]
                        == sha256_file(checkpoint)
                    )
                ):
                    raise ValueError("checkpoint/score mismatch")
                logged = next(
                    r
                    for r in summary["results"]
                    if r["seed"] == seed and r["rate"] == rate and r["step"] == step
                )
                if any(logged[k] != v for k, v in values(evaluation).items()):
                    raise ValueError("summary differs from evaluation")
                reports.append(evaluation)
            end = reports[-1]
            native_records.extend(end["native"])
            arms[str(rate)] = {
                "end": values(end),
                "end_minus_start": {k: v - values(start)[k] for k, v in values(end).items()},
                "max_main_above_start": max(
                    r["weighted"]["total"] - start["weighted"]["total"] for r in reports
                ),
                "main_above_start_count": sum(
                    r["weighted"]["total"] > start["weighted"]["total"] + TOLERANCE for r in reports
                ),
                "native_end_minus_start": native_delta(start["native"], end["native"]),
            }
        low, control = arms["0.0001"]["end"], arms["0.001"]["end"]
        checks = {
            "main_below_start_and_control": low["total"]
            < min(values(start)["total"], control["total"]) - TOLERANCE,
            "combined_below_start_and_control": low["combined"]
            < min(values(start)["combined"], control["combined"]) - TOLERANCE,
            "paired_not_worse_than_start": low["paired_restoration"]
            <= values(start)["paired_restoration"] + TOLERANCE,
            "identity_not_worse_than_start": low["identity_preservation"]
            <= values(start)["identity_preservation"] + TOLERANCE,
        }
        results.append(
            {
                "seed": seed,
                "start": values(start),
                "arms": arms,
                "checks": checks,
                "supports_smaller_updates": all(checks.values()),
            }
        )
    if len(native_records) != 60:
        raise ValueError("requires 60 native replays")
    report = {
        "scope": registration["scope"],
        "registration_sha256": sha256_file(root / "registration.json"),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "report_hashes": report_hashes,
        "results": results,
        "both_seeds_support_smaller_updates": all(r["supports_smaller_updates"] for r in results),
        "native_count": len(native_records),
        "native_proxy_max_error": max(r["proxy_max_error"] for r in native_records),
    }
    write_json_report(report, root / "comparisons.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
