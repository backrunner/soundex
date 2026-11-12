"""Matched-study completion audit from frozen endpoints, scores and timing traces.

No optimizer updates, model selection, rerendering or timing retries.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import math
import shutil
import statistics
from pathlib import Path

import torch

from data.causal_study import digest
from evaluation.band_attribution import deadline_accounting
from evaluation.reporting import sha256_file
from scripts.assess_band_phase import compare, records
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def summarize(root, output, *, study="p72", control="small", candidate="double"):
    if output.exists():
        raise ValueError("preserve previous audits; fresh destination required")
    consumed = []

    def checked(path):
        consumed.append(path)
        return read(path)

    for name in (
        f"{study}-complete.json",
        "complete.json",
        "preflight-complete.json",
        "longform-recheck/complete.json",
    ):
        checked_receipt(root / name)
        consumed.append(root / name)
    config = checked(root / "registration.json")
    if control == candidate or set(config["arms"]) != {control, candidate}:
        raise ValueError("named comparators must match the registered arms")
    endpoints = checked(root / "endpoints.json")
    if set(endpoints) != {f"{seed}-{kind}" for seed in config["seeds"] for kind in config["arms"]}:
        raise ValueError("incomplete endpoint set")
    report = {
        "study": study,
        "control": control,
        "candidate": candidate,
        "endpoints": {},
        "source_review": {},
        "native_roles": 0,
        "late_hops": [],
        "fit_history": {},
        "fit_loss": {},
    }
    for label, entry in endpoints.items():
        path = Path(entry["path"])
        consumed.append(path)
        verify({str(path): entry["sha256"]})
        saved = torch.load(path, map_location="cpu", weights_only=False)
        history = saved["history"]
        if (
            saved["step"] != config["steps"]
            or len(history) != config["steps"]
            or saved["signature"]
            != digest({"config": config, "kind": saved["kind"], "seed": saved["seed"]})
            or label != f"{saved['seed']}-{saved['kind']}"
        ):
            raise ValueError("endpoint budget or signature mismatch")
        tensors = list(saved["model"].values()) + [
            v
            for state in saved["optimizer"]["state"].values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ]
        if not all(torch.isfinite(v).all() for v in tensors):
            raise ValueError("nonfinite final model/optimizer")
        if not all(
            math.isfinite(v) for h in history for v in h.values() if isinstance(v, (int, float))
        ):
            raise ValueError("nonfinite optimization history")
        report["endpoints"][label] = {
            "sha256": entry["sha256"],
            "steps": saved["step"],
            "finite_model_optimizer": True,
            "finite_history": True,
            "optimization_hours": sum(h["seconds"] for h in history) / 3600,
            "last_four_256_loss_means": [
                statistics.mean(
                    h["objective_before"] for h in history[-256 * i : len(history) - 256 * (i - 1)]
                )
                for i in (4, 3, 2, 1)
            ],
        }
        for directory, key in (
            ("fit-history-review", "fit_history"),
            ("fit-loss-review", "fit_loss"),
        ):
            diagnostic = root / directory / f"{label}.json"
            if diagnostic.exists():
                report[key][label] = checked(diagnostic)
    for path in sorted((root / "native-review").glob("*/report.json")):
        value = checked(path)
        verify(value["artifacts"])
        report["native_roles"] += sum(len(case["identity"]) for case in value["cases"])
    for seed in config["seeds"]:
        native, raw = {}, {}
        for kind in config["arms"]:
            label = f"{seed}-{kind}"
            native[kind] = records(
                checked(root / "native-review" / label / "report.json"), "native"
            )
            raw[kind] = records(checked(root / "raw-review" / (label + ".json")), "scores")
        selected = {(r["case"], r["role"]) for r in native[control]}
        matched = {
            k: [r for r in rows if (r["case"], r["role"]) in selected] for k, rows in raw.items()
        }
        report["source_review"][str(seed)] = {
            "raw_full": compare(raw[control], raw[candidate]),
            "raw_native_population": compare(matched[control], matched[candidate]),
            "native": compare(native[control], native[candidate]),
        }
    panel = checked(root / "panel-assessment.json")
    report["longform_comparisons"] = panel["comparisons"]
    costs = checked(root / "cost-summary.json")
    report["runtime"] = {}
    for label in endpoints:
        values = [r for key, r in costs.items() if key.startswith(label + "/")]
        report["runtime"][label] = {
            "hops": sum(r["measured_hops"] for r in values),
            "max_compute_ms": max(r["stages"]["total_ms"]["max_ms"] for r in values),
            "over_5ms": sum(r["stages"]["total_ms"]["over_5ms"] for r in values),
            "service_over_hop": sum(r["stages"]["total_ms"]["service_over_hop"] for r in values),
            "paced_late_hops": sum(r["paced_late_hops"] for r in values),
            "max_native_peak_rss_bytes": max(r["process_peak_rss_bytes"] for r in values),
            "paced": [r for r in values if r["paced"]],
        }
    for key, value in costs.items():
        path = root / "native-cost" / key / "raw.json"
        if sha256_file(path) != value["raw_sha256"]:
            raise ValueError("native timing drift")
        raw = checked(path)
        if not value["paced"]:
            continue
        # Timing excludes 100 hops; raw warmup_hops is the model's 80-hop history.
        timing = deadline_accounting(raw)
        if (
            timing["measured_hops"] != value["measured_hops"]
            or timing["late_hops"] != value["paced_late_hops"]
        ):
            raise ValueError("paced trace/summary mismatch")
        for row in timing["outliers"]:
            i = row["hop"]
            report["late_hops"].append(
                {
                    "case": key,
                    "raw_hop_index": i,
                    "finish_lateness_ms": row["finish_late_ms"],
                    "previous_finish_lateness_ms": row["previous_finish_late_ms"],
                    "next_finish_lateness_ms": (
                        raw["paced_finish_lateness_ms"][i + 1]
                        if i + 1 < len(raw["paced_finish_lateness_ms"])
                        else None
                    ),
                    "wake_lateness_ms": raw["wake_lateness_ms"][i],
                    "stages": dict(zip(raw["timing_columns"], raw["timings"][i], strict=True)),
                    "scope": "Observed delayed wake plus compute; not an identified OS/root cause or actual-device xrun.",
                }
            )
    report["scope"] = (
        "Completed matched training audit; quality and timing outcomes are recorded, not automatically qualified. No promotion, independent test, listening or convergence claim. Original records and every adverse source retained. Fit diagnostics are bound at audit time, not retroactively part of the original completion receipt."
    )
    report["selection"] = config["selection"]
    output.mkdir(parents=True)
    frozen = output / "summarize_band_capacity.py"
    shutil.copy2(Path(__file__).resolve(), frozen)
    registered(report, output / "summary.json")
    consumed += [frozen, output / "summary.json"]
    registered({"files": {str(p): sha256_file(p) for p in consumed}}, output / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study", default="p72")
    parser.add_argument("--control", default="small")
    parser.add_argument("--candidate", default="double")
    args = parser.parse_args()
    summarize(
        args.root.resolve(),
        args.output.resolve(),
        study=args.study,
        control=args.control,
        candidate=args.candidate,
    )
