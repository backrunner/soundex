#!/usr/bin/env python3
"""Summarize fixed-radius and selected-step outcomes, including no-ops.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.batch_update_probe import CHECKS, POLICIES
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import read


def changes(base: list[dict], values: list[dict]) -> dict:
    if not base:
        return {"count": 0, "metrics": {}, "all_five_improved": 0}
    result, joint = {}, np.ones(len(base), dtype=bool)
    for key in base[0]:
        before = np.array([b[key] for b in base])
        delta = np.array([v[key] - b[key] for v, b in zip(values, base, strict=True)])
        nonzero = before != 0
        result[key] = {
            "improved": int((delta < 0).sum()),
            "unchanged": int((delta == 0).sum()),
            "worsened": int((delta > 0).sum()),
            "median_base": float(np.median(before)),
            "median_change": float(np.median(delta)),
            "median_percent_change_nonzero_base": float(
                np.median(100 * delta[nonzero] / before[nonzero])
            )
            if nonzero.any()
            else None,
            "percent_omitted_zero_base": int((~nonzero).sum()),
            "min_change": float(delta.min()),
            "max_change": float(delta.max()),
        }
        if key in CHECKS:
            joint &= delta < 0
    return {"count": len(base), "metrics": result, "all_five_improved": int(joint.sum())}


def pcm_changes(cases: list[dict], selected: list[dict | None]) -> dict:
    if not cases:
        return {}
    result = {}
    for key in cases[0]["base_transfer_pcm"]:
        before = np.array([c["base_transfer_pcm"][key]["mean_rms"] for c in cases])
        after = np.array(
            [
                (p["transfer_pcm"] if p else c["base_transfer_pcm"])[key]["mean_rms"]
                for c, p in zip(cases, selected, strict=True)
            ]
        )
        delta = after - before
        result[key] = {
            "definition": "mean of per-example raw PCM error RMS, not RMS of a mean waveform",
            "improved": int((delta < 0).sum()),
            "unchanged": int((delta == 0).sum()),
            "worsened": int((delta > 0).sum()),
            "median_base_rms": float(np.median(before)),
            "median_after_rms": float(np.median(after)),
            "median_change_rms": float(np.median(delta)),
            "min_change_rms": float(delta.min()),
            "max_change_rms": float(delta.max()),
        }
    return result


def summarize(path: Path) -> dict:
    raw = read(path)
    if raw["status"] != "complete":
        raise ValueError("completed P42 run required")
    result = {"input_sha256": sha256_file(path), "endpoints": {}}
    for label, cases in raw["cases"].items():
        output = {}
        for name in cases[0]["directions"]:
            fixed = []
            for i in range(len(cases[0]["directions"][name]["points"])):
                points = [c["directions"][name]["points"][i] for c in cases]
                fixed.append(
                    {
                        "radius": points[0]["radius"],
                        **{
                            role: changes(
                                [c["base"][role] for c in cases],
                                [p["values"][role] for p in points],
                            )
                            for role in ("fit", "guard", "transfer")
                        },
                        "transfer_pcm": pcm_changes(cases, points),
                    }
                )
            policies = {}
            for policy in POLICIES:
                selected = [
                    None
                    if c["directions"][name]["selected"][policy] is None
                    else c["directions"][name]["points"][c["directions"][name]["selected"][policy]]
                    for c in cases
                ]
                policies[policy] = {
                    "accepted": sum(p is not None for p in selected),
                    "radii": dict(Counter(str(p["radius"]) if p else "no_op" for p in selected)),
                    "all_cases": changes(
                        [c["base"]["transfer"] for c in cases],
                        [
                            p["values"]["transfer"] if p else c["base"]["transfer"]
                            for c, p in zip(cases, selected, strict=True)
                        ],
                    ),
                    "accepted_only": changes(
                        [
                            c["base"]["transfer"]
                            for c, p in zip(cases, selected, strict=True)
                            if p is not None
                        ],
                        [p["values"]["transfer"] for p in selected if p is not None],
                    ),
                    "all_case_transfer_pcm": pcm_changes(cases, selected),
                }
            output[name] = {"fixed": fixed, "policies": policies}
        result["endpoints"][label] = output
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    destination = args.run / "analysis.json"
    if destination.exists():
        raise ValueError("refuse to overwrite analysis")
    write_json_report(summarize(args.run / "summary.json"), destination)
