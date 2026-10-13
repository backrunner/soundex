#!/usr/bin/env python3
"""P13 matched fixed-crop / finite-bank comparison; no checkpoint selection."""

import argparse
import json
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_subset_study import run as analyze_subset


def run(root):
    configs, results = [], []
    for label in ("fixed", "multiposition"):
        study = root / label
        analyze_subset(study)  # Verify every checkpoint, evaluation and schedule receipt.
        configs.append(json.loads((study / "config.json").read_text()))
        results.append(json.loads((study / "scaling-comparisons.json").read_text()))
    if configs[0]["sizes"] != [40] or configs[1]["sizes"] != [40]:
        raise ValueError("P13 requires the same 40 recordings")
    if [c["crop_sampling"]["positions_per_recording"] for c in configs] != [1, 8]:
        raise ValueError("P13 requires fixed and eight-position arms")
    if {k: v for k, v in configs[0].items() if k != "crop_sampling"} != {
        k: v for k, v in configs[1].items() if k != "crop_sampling"
    }:
        raise ValueError("non-crop configuration differs")
    for seed in configs[0]["seeds"]:
        schedules = [
            json.loads((root / label / f"size-40-{seed}" / "draws.json").read_text())
            for label in ("fixed", "multiposition")
        ]
        if schedules[0]["draws"] != schedules[1]["draws"]:
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
                "fixed": fixed["candidate"]["weighted"],
                "multiposition": multi["candidate"]["weighted"],
                "multiposition_minus_fixed": {
                    k: multi["candidate"]["weighted"][k] - v
                    for k, v in fixed["candidate"]["weighted"].items()
                },
            }
        comparisons.append(
            {
                "seed": key[0],
                "step": key[1],
                "groups": groups,
                "native": {"fixed": a[key]["native"], "multiposition": b[key]["native"]},
            }
        )
    write_json_report(
        {
            "scope": "single factor: bounded crop coverage; reused development, no promotion",
            "source_reports": {
                label: sha256_file(root / label / "scaling-comparisons.json")
                for label in ("fixed", "multiposition")
            },
            "results": comparisons,
        },
        root / "crop-comparisons.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
