#!/usr/bin/env python3
"""Summarize frozen P19 failure concentration, generalization and synthesis stages."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from scripts.audit_coverage_failure import read


def mean(values):
    return float(np.mean(values))


def run(args):
    receipt = read(args.audit / "summary.json")
    registration = read(args.audit / "registration.json")
    if sha256_file(args.audit / "registration.json") != receipt["registration_sha256"]:
        raise ValueError("audit registration drift")
    for name, expected in registration["source_hashes"].items():
        if sha256_file(Path(__file__).resolve().parents[1] / name) != expected:
            raise ValueError(f"audit source drift: {name}")
    for name, expected in registration["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"audit input drift: {name}")
    stages = {}
    for name, expected in receipt["reports"].items():
        if sha256_file(args.audit / name) != expected:
            raise ValueError("stage report drift")
        records = read(args.audit / name)["records"]
        rows = [r for r in records if r["role"] == "identity" and r["corpus"] == "music_library"]
        stages[name] = {
            "music_clips": len(rows),
            "means": {
                stage: mean([sum(b["total"] for b in r["stages"][stage]) for r in rows])
                for stage in ("raw", "blended", "ola", "gated", "limited")
            },
            "limited_bands": np.mean(
                [[b["total"] for b in r["stages"]["limited"]] for r in rows], axis=0
            ).tolist(),
            "active_fraction": mean([r["active_fraction"] for r in rows]),
            "cutoff_median": float(np.median([r["cutoff_mean"] for r in rows])),
            "floor_dominated_by_band": [
                sum(r["stages"]["limited"][i]["floor_dominated"] for r in rows) for i in range(5)
            ],
            "codec_counts": dict(Counter(r["codec_id"] for r in rows)),
            "channel_counts": dict(Counter(r["channel_role"] for r in rows)),
            "wave_superposition_max_error": max(
                r["transfer"]["wave_superposition_max_error"] for r in records
            ),
            "spectrum_superposition_max_error": max(
                r["transfer"]["spectrum_superposition_max_error"] for r in records
            ),
        }
    initial = read(args.study / "review/initial/proxy.json")["groups"]["development"]["rows"]
    comparisons = {}
    for label in registration["checkpoints"]:
        if not label.startswith("wide-"):
            continue
        _, step, seed = label.split("-")
        proxy = read(args.study / "review" / label / "proxy.json")
        current = proxy["groups"]["development"]["rows"]
        if [r["spec"] for r in current] != [r["spec"] for r in initial]:
            raise ValueError("case order drift")
        music = [
            (a, b) for a, b in zip(initial, current, strict=True) if a["corpus"] == "music_library"
        ]
        paired = np.array(
            [
                b["metrics"]["paired_restoration"] - a["metrics"]["paired_restoration"]
                for a, b in music
            ]
        )
        identity = np.array([b["metrics"]["identity_preservation"] for a, b in music])
        delta = (paired + identity) / 2
        indices = np.argsort(-delta)[:5]
        top = [
            {
                "spec": music[i][1]["spec"],
                "track_id": music[i][1]["track_id"],
                "genre": music[i][1]["genre"],
                "paired_delta": float(paired[i]),
                "identity": float(identity[i]),
                "net_increase_share": float(delta[i] / delta.sum()),
            }
            for i in indices
        ]
        updates = read(args.study / f"wide-{seed}/updates.json")[: int(step)]
        grads = np.array([v["preclip_gradient_norm"] for v in updates])
        train = read(args.study / f"wide-{seed}" / f"training-{step}.json")
        comparisons[label] = {
            "paired_improved": int(sum(paired < -1e-6)),
            "paired_worse": int(sum(paired > 1e-6)),
            "paired_delta_mean": mean(paired),
            "paired_delta_median": float(np.median(paired)),
            "identity_mean": mean(identity),
            "total_delta_mean": mean(delta),
            "identity_share_of_net_increase": float(
                identity.sum() / (identity.sum() + paired.sum())
            ),
            "top_cases": top,
            "top_two_share_of_net_increase": sum(t["net_increase_share"] for t in top[:2]),
            "gradient_clipped_updates": int(sum(grads > 5)),
            "updates": len(updates),
            "gradient_quantiles_50_95_99_max": np.quantile(grads, [0.5, 0.95, 0.99, 1]).tolist(),
            "full_training_corpus": train["arm_bank"]["corpus"],
            "shifted_same_recording": proxy["groups"]["shifted"]["corpus"],
        }
    native = read(args.audit / "native-interventions.json")
    if native["audit_sha256"] != sha256_file(args.audit / "summary.json"):
        raise ValueError("native receipt drift")
    write_json_report(
        {
            "scope": "post-hoc descriptive mechanism audit; no candidate tuning, no quality promotion",
            "audit_sha256": sha256_file(args.audit / "summary.json"),
            "comparisons_sha256": sha256_file(args.study / "comparisons.json"),
            "native_sha256": sha256_file(args.audit / "native-interventions.json"),
            "analyzer_sha256": sha256_file(Path(__file__)),
            "stages": stages,
            "comparisons": comparisons,
            "native_count": native["native_count"],
            "native_max_error": native["maximum_proxy_error"],
        },
        args.audit / "analysis.json",
    )
    print(
        {
            "stage_reports": len(stages),
            "native_count": native["native_count"],
            "native_max_error": native["maximum_proxy_error"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    run(parser.parse_args())
