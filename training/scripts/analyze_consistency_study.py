#!/usr/bin/env python3
"""P16 matched original / consistency-auxiliary comparison; no checkpoint selection."""

import argparse
import json
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_subset_study import run as analyze_subset


def run(root):
    configs, results = [], []
    for label in ("control", "consistency"):
        study = root / label
        analyze_subset(study)  # Verify every checkpoint, evaluation and schedule receipt.
        configs.append(json.loads((study / "config.json").read_text()))
        results.append(json.loads((study / "scaling-comparisons.json").read_text()))
    if configs[0]["sizes"] != [40] or configs[1]["sizes"] != [40]:
        raise ValueError("P16 requires the same 40 recordings")
    if any(c["crop_sampling"]["positions_per_recording"] != 8 for c in configs):
        raise ValueError("P16 keeps eight-position sampling in both arms")
    if configs[0]["consistency_weight"] != 0 or configs[1]["consistency_weight"] <= 0:
        raise ValueError("expected zero and positive auxiliary coefficient")
    ignored = {"consistency_weight", "calibration_sha256"}
    if {k: v for k, v in configs[0].items() if k not in ignored} != {
        k: v for k, v in configs[1].items() if k not in ignored
    }:
        raise ValueError("non-auxiliary configuration differs")
    for seed in configs[0]["seeds"]:
        schedules = [
            json.loads((root / label / f"size-40-{seed}" / "draws.json").read_text())
            for label in ("control", "consistency")
        ]
        if schedules[0]["crop_draws"] != schedules[1]["crop_draws"]:
            raise ValueError("recording schedules differ")
    a = {(r["seed"], r["step"]): r for r in results[0]["results"]}
    b = {(r["seed"], r["step"]): r for r in results[1]["results"]}
    if a.keys() != b.keys():
        raise ValueError("milestones differ")
    comparisons = []
    for key in sorted(a):
        groups = {}
        for group in ("fitted_anchor", "fitted_shifted", "development"):
            fixed, multi = a[key]["proxy"][group], b[key]["proxy"][group]
            if fixed["baseline"] != multi["baseline"]:
                raise ValueError("common baseline differs")
            groups[group] = {
                "baseline": fixed["baseline"]["weighted"],
                "control": fixed["candidate"]["weighted"],
                "consistency": multi["candidate"]["weighted"],
                "consistency_minus_control": {
                    k: multi["candidate"]["weighted"][k] - v
                    for k, v in fixed["candidate"]["weighted"].items()
                },
            }
        comparisons.append(
            {
                "seed": key[0],
                "step": key[1],
                "groups": groups,
                "native": {"control": a[key]["native"], "consistency": b[key]["native"]},
            }
        )
    write_json_report(
        {
            "scope": "single factor: reference-aware synthesis consistency auxiliary; fixed original score; no promotion",
            "source_reports": {
                label: sha256_file(root / label / "scaling-comparisons.json")
                for label in ("control", "consistency")
            },
            "results": comparisons,
        },
        root / "consistency-comparisons.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
