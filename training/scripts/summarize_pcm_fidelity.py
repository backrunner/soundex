#!/usr/bin/env python3
"""Summarize P44 by corpus/role without turning precision rulers into pass gates.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import read


def distribution(values: list[float]) -> dict:
    array = np.asarray(values)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("nonempty finite measurements required")
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def summarize(path: Path) -> dict:
    raw = read(path)
    if raw["status"] != "complete":
        raise ValueError("complete frozen calibration required")
    result = {"input_sha256": sha256_file(path), "endpoints": {}}
    for label, cases in raw["cases"].items():
        buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for case in cases:
            samples = case["samples"]["calibration"]
            if case["identity"] != [False] * len(samples) + [True] * len(samples):
                raise ValueError("unexpected paired/identity ordering")
            for index, identity in enumerate(case["identity"]):
                sample = samples[index % len(samples)]
                corpus = sample["row"]["corpus"]
                role = "identity" if identity else "paired"
                for group in (f"{corpus}/{role}", f"all/{role}"):
                    for name, metrics in case["paths"].items():
                        for metric, values in metrics.items():
                            buckets[group][name][metric].append(values[index])
                    for metric, values in case["model_minus_passthrough"].items():
                        buckets[group]["model_minus_passthrough"][metric].append(values[index])
                    for metric, values in case["reference"].items():
                        buckets[group]["reference"][metric].append(values[index])
        result["endpoints"][label] = {
            "coverage": raw["coverage"][label],
            "groups": {
                group: {
                    name: {k: distribution(v) for k, v in metrics.items()}
                    for name, metrics in paths.items()
                }
                for group, paths in buckets.items()
            },
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    destination = args.run / "analysis.json"
    if destination.exists():
        raise ValueError("refuse to overwrite calibration analysis")
    write_json_report(summarize(args.run / "summary.json"), destination)
