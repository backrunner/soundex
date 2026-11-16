"""Descriptive P82 whole-grid summary after every registered probe completes.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file
from models.phase_relations import KINDS
from models.realized_phase_relations import DOMAINS
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def stats(values):
    finite = [v for v in values if v is not None]
    if not all(np.isfinite(v) for v in finite):
        raise ValueError("nonfinite summary input")
    return {
        "count": len(values),
        "undefined": len(values) - len(finite),
        "negative": sum(v < 0 for v in finite),
        "zero": sum(v == 0 for v in finite),
        "median": float(np.median(finite)) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def pair(values, a, b):
    return values[a + "|" + b] if a + "|" + b in values else values[b + "|" + a]


def run(root, output):
    if output.exists():
        raise ValueError("fresh descriptive summary required")
    checked_receipt(root / "complete.json")
    config = read(root / "registration.json")
    hashes = {**read(root / "complete.json")["files"]}
    hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    output.mkdir(parents=True)
    source = output / "summary-source.py"
    source.write_text(Path(__file__).read_text())
    hashes[str(source)] = sha256_file(source)
    registered(
        {
            "hashes": hashes,
            "scope": "All32 completed frozen probes and40 saved native correspondence cases. Descriptive gradient summaries, no optimizer, source/candidate selection, quality gate or new native execution.",
        },
        output / "registration.json",
    )
    probes = [
        read(root / "probes" / origin / f"{draw}.json")
        for origin, draw in itertools.product(config["endpoints"], config["fit"])
    ]
    native = [
        read(root / "native" / origin / arm / f"{draw}.json")
        for origin, arm, draw in itertools.product(
            config["endpoints"], ("circular_relation", "wrapped_relation"), config["native_draws"]
        )
    ]
    arms = [f"{d}/{k}" for d, k in itertools.product(DOMAINS, KINDS)]
    targets = ("old", "fused_mr", "fused_hf_magnitude", "fused_complex")
    groups, strata, placement = {}, {}, {}
    for origin in ("all", *config["endpoints"]):
        selected = [p for p in probes if origin == "all" or p["origin"] == origin]
        for group in ("all", "base", "phase_head"):
            for arm in arms:
                label = f"{origin}/{group}/{arm}"
                values = [p["parameters"][group] for p in selected]
                groups[label] = {
                    "norm_ratio_to_old": stats(
                        [
                            v["norms"][arm] / v["norms"]["old"] if v["norms"]["old"] else None
                            for v in values
                        ]
                    ),
                    "auxiliary_cosines": {
                        t: stats([pair(v["cosines"], arm, t) for v in values]) for t in targets
                    },
                    "old_plus_auxiliary_cosines": {
                        t: stats([v["old_plus_auxiliary_cosines"][arm][t] for v in values])
                        for t in targets
                    },
                }
            for kind, (a, b) in itertools.product(
                KINDS,
                (("direct", "raw"), ("direct", "fused"), ("raw", "fused"), ("legacy", "direct")),
            ):
                placement[f"{origin}/{group}/{kind}/{a}-vs-{b}"] = stats(
                    [
                        pair(p["parameters"][group]["cosines"], a + "/" + kind, b + "/" + kind)
                        for p in selected
                    ]
                )
        for stratum, arm in itertools.product(probes[0]["strata"], arms):
            values = [p["strata"][stratum] for p in selected]
            strata[f"{origin}/{stratum}/{arm}"] = {
                "auxiliary_cosines": {
                    t: stats([v["cosines"][arm][t] for v in values]) for t in targets
                },
                "old_plus_auxiliary_cosines": {
                    t: stats([v["old_plus_auxiliary_cosines"][arm][t] for v in values])
                    for t in targets
                },
            }
    phase = {
        kind: {
            "risk_difference": stats(
                [n["phase_correspondence"][kind]["risk_difference"][0] for n in native]
            ),
            "risk_bound": stats([n["phase_correspondence"][kind]["risk_bound"][0] for n in native]),
            "supported_prediction_zero_count": sum(
                n["phase_correspondence"][kind]["supported_prediction_zero_count"][0]
                for n in native
            ),
        }
        for kind in KINDS
    }
    sources = {s["row"]["source_checksum"] for p in probes for s in p["samples"]}
    registered(
        {
            "groups": groups,
            "strata": strata,
            "placement_cosines": placement,
            "gradient_component_max_error": max(
                v["sum_gradient_max_error"] for p in probes for v in p["components"].values()
            ),
            "native": {
                "cases": len(native),
                "sources": sorted({n["source"] for n in native}),
                "pcm_max": max(n["pcm_max"] for n in native),
                "pcm_rms_max": max(n["pcm_rms"] for n in native),
                "phase": phase,
                "historical_scores_exact": all(n["historical_scores_exact"] for n in native),
            },
            "fit_recordings": len(sources),
            "fit_recording_hashes": sorted(sources),
            "qualified": False,
            "scope": "Euclidean derivatives and same-output numerical correspondence only. Counts are endpoint/draw probes, not independent recordings or audible gains. No finite Adam update, new candidate or full training result.",
        },
        output / "assessment.json",
    )
    verify(hashes)
    registered(
        {"files": {str(p): sha256_file(p) for p in output.rglob("*") if p.is_file()}},
        output / "complete.json",
    )
    print("summarized all32 probes and40 native correspondence cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
