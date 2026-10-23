#!/usr/bin/env python3
"""Summarize every P41 seed, role and radius without selecting a winner."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from data.causal_study import source_origin
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import read

CHECKS = [
    "paired_direct_amplitude",
    "paired_direct_complex",
    "paired_mr",
    "paired_mr_dynamics",
    "identity_mr",
]


def unique_seconds(samples: list, first_hop: int, end_hop: int) -> float:
    """Union physical-source intervals across codecs, rates and channel views."""
    intervals = defaultdict(list)
    for sample in samples:
        row, spec = sample["row"], sample["spec"]
        origin = source_origin(row) + spec["start"]
        intervals[row["source_checksum"]].append(
            (
                Fraction(origin + first_hop * 128, row["sample_rate"]),
                Fraction(origin + end_hop * 128, row["sample_rate"]),
            )
        )
    duration = Fraction(0)
    for spans in intervals.values():
        left, right = sorted(spans)[0]
        for a, b in sorted(spans)[1:]:
            if a > right:
                duration += right - left
                left, right = a, b
            else:
                right = max(right, b)
        duration += right - left
    return float(duration)


def coverage(cases: list, role: str) -> dict:
    samples = [sample for case in cases for sample in case["samples"][role]]
    return {
        "paired_crops": len(samples),
        "unique_recordings": len({s["row"]["track_id"] for s in samples}),
        # Prepared PCM has 128 leading padding samples. MR uses padded
        # [89*128, 255*128); direct frames 89..255 span source [88, 256) hops.
        "unique_mr_supervised_seconds": unique_seconds(samples, 88, 254),
        "unique_direct_analysis_support_seconds": unique_seconds(samples, 88, 256),
        "corpus": dict(Counter(s["row"]["corpus"] for s in samples)),
        "codec": dict(Counter(s["row"]["codec_id"] for s in samples)),
        "channel_view": dict(Counter(s["row"]["channel_role"] for s in samples)),
        "source_channels": dict(Counter(str(s["row"]["source_channels"]) for s in samples)),
        "music_genres": dict(
            Counter(
                s["row"]["source_metadata"].get("genre", "unknown")
                for s in samples
                if s["row"]["corpus"] == "music_library"
            )
        ),
        "music_regions": dict(
            Counter(
                s["row"]["source_metadata"].get("music_region", "unknown")
                for s in samples
                if s["row"]["corpus"] == "music_library"
            )
        ),
    }


def summarize(path: Path) -> dict:
    raw = read(path)
    if raw["status"] != "complete":
        raise ValueError("completed diagnostic required")
    output = {"input_sha256": sha256_file(path), "checks": CHECKS, "endpoints": {}}
    for label, cases in raw["cases"].items():
        result = {
            "coverage": {role: coverage(cases, role) for role in ("source", "transfer")},
            "directions": {},
        }
        for name in cases[0]["directions"]:
            points = []
            for i in range(2):
                point = {"radius": cases[0]["directions"][name]["points"][i]["radius"]}
                for role in ("source", "transfer"):
                    changes = {
                        k: np.array(
                            [c["directions"][name]["points"][i]["changes"][role][k] for c in cases]
                        )
                        for k in cases[0]["base"][role]
                    }
                    metrics = {}
                    for key, delta in changes.items():
                        base = np.array([c["base"][role][key] for c in cases])
                        nonzero = base != 0
                        metrics[key] = {
                            "improved": int((delta < 0).sum()),
                            "unchanged": int((delta == 0).sum()),
                            "worsened": int((delta > 0).sum()),
                            "median_base": float(np.median(base)),
                            "median_absolute_change": float(np.median(delta)),
                            "median_percent_change_nonzero_base": float(
                                np.median(100 * delta[nonzero] / base[nonzero])
                            )
                            if nonzero.any()
                            else None,
                            "percent_omitted_zero_base": int((~nonzero).sum()),
                            "min_absolute_change": float(delta.min()),
                            "max_absolute_change": float(delta.max()),
                        }
                    point[role] = {
                        "metrics": metrics,
                        "all_five_strictly_improved": int(
                            np.stack([changes[k] < 0 for k in CHECKS]).all(0).sum()
                        ),
                        "all_eight_nonincreasing": int(
                            np.stack([v <= 0 for v in changes.values()]).all(0).sum()
                        ),
                    }
                points.append(point)
            result["directions"][name] = points
        result["guarded_fixed_history_increases_at_0.001"] = {
            k: sum(
                c["directions"]["guarded_bisector"]["source_fixed_history_values"][1][k]
                > c["base"]["source"][k]
                for c in cases
            )
            for k in cases[0]["base"]["source"]
        }
        result["max_normalized_guard_violation_float32"] = max(
            max(c["directions"]["guarded_bisector"]["source_normalized_derivatives"].values())
            for c in cases
        )
        output["endpoints"][label] = result
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    destination = args.run / "analysis.json"
    if destination.exists():
        raise ValueError("refuse to overwrite analysis")
    write_json_report(summarize(args.run / "summary.json"), destination)
