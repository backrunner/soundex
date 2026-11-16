"""Summarize every frozen P83 trajectory without selecting a winner or a gate.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
from pathlib import Path

import torch

from evaluation.reporting import sha256_file
from evaluation.restoration_evidence import direction
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_realized_relation_updates import compare, compare_fit
from scripts.summarize_realized_relations import stats


def metric(value):
    baseline = value.get("control_mean")
    return {
        **value,
        "direction": direction(value),
        "percent_of_control_mean": (100 * value["mean_delta"] / baseline if baseline else None),
        "ci_percent_of_control_mean": (
            [100 * v / baseline for v in value["paired_source_bootstrap_95"]] if baseline else None
        ),
    }


def run(root, output):
    plan = read(output / "registration.json")
    verify(plan["hashes"])
    if (output / "assessment.json").exists():
        raise ValueError("completed summary must not be overwritten")
    checked_receipt(root / "complete.json")
    config = read(root / "registration.json")
    hashes = dict(read(root / "complete.json")["files"])
    hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    hashes.update(plan["hashes"])
    hashes[str(output / "registration.json")] = sha256_file(output / "registration.json")
    registered(
        {
            "hashes": hashes,
            "scope": "Descriptive review of all fixed16-step endpoints, all origins/arms/strata. No new training, selection, score threshold, timing run or promotion.",
        },
        output / "input-receipt.json",
    )
    updates, deltas = {}, {}
    frozen = read(root / "frozen.json")
    for origin, arm in itertools.product(config["endpoints"], config["arms"]):
        name = origin + "/" + arm
        steps = [read(root / "updates" / name / f"{n:02d}.json") for n in range(1, 17)]
        updates[name] = {
            k: stats([s[k] for s in steps])
            for k in (
                "actual_directional_derivative",
                "old_directional_derivative",
                "auxiliary_directional_derivative",
                "auxiliary_gradient_norm",
                "gradient_norm",
                "changed_parameters",
            )
        }
        updates[name]["clipped_steps"] = sum(s["gradient_norm"] > 5 for s in steps)
        updates[name]["steps"] = [
            {
                k: s[k]
                for k in ("step", "old_directional_derivative", "auxiliary_directional_derivative")
            }
            for s in steps
        ]
        candidate = torch.load(
            frozen["endpoints"][name]["path"], weights_only=False, map_location="cpu"
        )["model"]
        for label, control_entry in (
            ("old", frozen["endpoints"][origin + "/old"]),
            ("parent", config["endpoints"][origin]),
        ):
            control = torch.load(control_entry["path"], weights_only=False, map_location="cpu")[
                "model"
            ]
            groups = {}
            for group in ("base", "phase_head"):
                values = [
                    candidate[k].double() - v.double()
                    for k, v in control.items()
                    if k.startswith(group + ".")
                ]
                groups[group] = {
                    "canonical_l2": float(sum(v.square().sum() for v in values).sqrt()),
                    "canonical_max": max(float(v.abs().max()) for v in values),
                    "changed_state_elements": sum(int(torch.count_nonzero(v)) for v in values),
                }
            deltas[name + "-vs-" + label] = groups
    checks = [read(p) for p in sorted((root / "relation-correspondence").rglob("*.json"))]
    assessment = compare(read(root / "records.json"), config, seed=plan["bootstrap_seed"])
    fit = compare_fit(read(root / "fit-records.json"), config, seed=plan["bootstrap_seed"])
    registered(
        {
            "updates": updates,
            "canonical_state_differences": deltas,
            "control_assessment": read(root / "control-assessment.json"),
            "step2_old_controls_exact": frozen["step2_old_controls_exact"],
            "native": {
                "checks": len(checks),
                "pcm_max": max(v["pcm_max"] for v in checks),
                "pcm_rms_max": max(v["pcm_rms"] for v in checks),
                "historical_scores_exact": all(v["historical_scores_exact"] for v in checks),
                "phase": {
                    kind: {
                        field: stats([v["phase_correspondence"][kind][field][0] for v in checks])
                        for field in (
                            "risk_difference",
                            "risk_bound",
                            "supported_prediction_zero_count",
                        )
                    }
                    for kind in ("circular", "wrapped")
                },
            },
            "comparisons": {
                k: {name: metric(v) for name, v in values.items()}
                for k, values in assessment["comparisons"].items()
            },
            "fit_comparisons": {
                k: {name: metric(v) for name, v in values.items()}
                for k, values in fit["comparisons"].items()
            },
            "interpretations": assessment["interpretations"],
            "bootstrap_seed": plan["bootstrap_seed"],
            "registered_seed83_retained": str(root / "assessment.json"),
            "qualified": False,
            "scope": config["scope"],
        },
        output / "assessment.json",
    )
    verify(hashes)
    registered(
        {"files": {str(p): sha256_file(p) for p in output.iterdir() if p.is_file()}},
        output / "complete.json",
    )
    print("All16 endpoints summarized; no quality promotion.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.run.resolve(), args.output.resolve())
