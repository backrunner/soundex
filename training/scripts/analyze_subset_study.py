#!/usr/bin/env python3
"""Summarize frozen P12 checkpoints against identical per-region input baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from scripts.run_subset_study import aggregate


def compare(candidate, baseline, tracks=None):
    if tracks is not None:
        candidate = [r for r in candidate if r["track_id"] in tracks]
        baseline = [r for r in baseline if r["track_id"] in tracks]
    keys = [(r["track_id"], r["spec"]) for r in candidate]
    if keys != [(r["track_id"], r["spec"]) for r in baseline]:
        raise ValueError("candidate/baseline cases differ")
    current, initial = aggregate(candidate), aggregate(baseline)
    return {
        "rows": len(candidate),
        "recordings": len({r["track_id"] for r in candidate}),
        "baseline": initial,
        "candidate": current,
        "delta": {k: current["weighted"][k] - v for k, v in initial["weighted"].items()},
        "score_change_percent": 100
        * (current["weighted"]["total"] / initial["weighted"]["total"] - 1),
    }


def run(root):
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("finish all registered runs before aggregate analysis")
    small = {s["row_id"] for s in config["train_pools"]["5"]}
    common = {
        r["track_id"] for r in baseline["anchor"]["proxy"]["rows"] if r["spec"]["row_id"] in small
    }
    results = []
    for receipt in summary["runs"]:
        directory = root / f"size-{receipt['size']}-{receipt['seed']}"
        for name, expected in receipt["hashes"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"completed artifact changed: {name}")
        for step in sorted({config["common_steps"], receipt["steps"]}):
            report = json.loads((directory / f"evaluation-{step}.json").read_text())
            if report["checkpoint_sha256"] != sha256_file(directory / f"step-{step}.pth"):
                raise ValueError("evaluation checkpoint mismatch")
            groups = report["groups"]
            comparisons = {}
            for group in ("anchor", "shifted"):
                rows = groups[group]["proxy"]["rows"]
                initial = baseline[group]["proxy"]["rows"]
                comparisons["common_" + group] = compare(rows, initial, common)
                comparisons["fitted_" + group] = compare(
                    rows, initial, set(groups[group]["proxy"]["fitted_track_ids"])
                )
            comparisons["development"] = compare(
                groups["development"]["proxy"]["rows"], baseline["development"]["proxy"]["rows"]
            )
            native = {}
            for group in groups:
                a, b = groups[group]["native"], baseline[group]["native"]
                if len(a) != 10 or len(b) != 10:
                    raise ValueError("registered native subsets require five pairs")
                native[group] = {"proxy_max_error": max(r["proxy_max_error"] for r in a)}
                for label, indices in (("paired", range(5)), ("identity", range(5, 10))):
                    native[group][label] = {
                        k: {
                            "mean": float(
                                np.mean(
                                    [
                                        a[i]["output_band_metrics"][k]
                                        - b[i]["output_band_metrics"][k]
                                        for i in indices
                                    ]
                                )
                            ),
                            "per_clip": [
                                a[i]["output_band_metrics"][k] - b[i]["output_band_metrics"][k]
                                for i in indices
                            ],
                        }
                        for k in ("high_sc", "high_flux", "high_energy_error_db", "band_0_4000_sc")
                    }
            results.append(
                {
                    "size": receipt["size"],
                    "seed": receipt["seed"],
                    "step": step,
                    "common_update": step == config["common_steps"],
                    "equal_exposure": step == receipt["steps"],
                    "proxy": comparisons,
                    "native": native,
                }
            )
    write_json_report(
        {
            "scope": config["scope"],
            "config_sha256": sha256_file(root / "config.json"),
            "baseline_sha256": sha256_file(root / "baseline.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "results": results,
        },
        root / "scaling-comparisons.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
