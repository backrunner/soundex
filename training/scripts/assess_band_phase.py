"""Summarize fixed P65 fit endpoints and diagnose the phase head's history use.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_training import objective_value
from models.direct_spectrum_objective import direct_spectrum_loss
from models.restoration_head import predict_budget
from scripts.run_band_phase_pilot import SEEDS, model_for
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def repeated_current(_module, inputs):
    """Ablate only the new head's previous-frame carrier, not backbone history."""
    features = inputs[0].clone()
    carrier = features[..., :129]
    features[..., 3 * 129 : 4 * 129] = carrier.square() / (1 + carrier.square())
    features[..., 4 * 129 : 5 * 129] = 0
    return (features,)


def records(report, key):
    result = []
    for case in report["cases"]:
        samples = case["samples"]["calibration"]
        for index, identity in enumerate(case["identity"]):
            row = samples[index % len(samples)]["row"]
            result.append(
                {
                    "case": case["index"],
                    "role": index,
                    "source": row["source_checksum"],
                    "source_id": row["source_id"],
                    "codec": row["codec_id"],
                    "group": row["corpus"] + ("/identity" if identity else "/paired"),
                    "scores": {k: v[index] for k, v in case[key].items()},
                }
            )
    return result


def compare(before, after):
    if len(before) != len(after):
        raise ValueError("unmatched case counts")
    grouped, cases = defaultdict(list), []
    for a, b in zip(before, after, strict=True):
        if {k: v for k, v in a.items() if k != "scores"} != {
            k: v for k, v in b.items() if k != "scores"
        }:
            raise ValueError("unmatched source/role/order")
        case = {**a, "before": a["scores"], "after": b["scores"]}
        del case["scores"]
        case["delta"] = {k: b["scores"][k] - v for k, v in a["scores"].items()}
        cases.append(case)
        grouped[a["group"]].append(case)
    summary = {}
    for group, values in grouped.items():
        sources = sorted({r["source"] for r in values})
        metrics = {}
        for metric in values[0]["before"]:
            means = {}
            for side in ("before", "after"):
                means[side] = sum(
                    sum(r[side][metric] for r in values if r["source"] == source)
                    / sum(r["source"] == source for r in values)
                    for source in sources
                ) / len(sources)
            metrics[metric] = {
                **means,
                "delta": means["after"] - means["before"],
                "relative_percent": 100 * (means["after"] / means["before"] - 1)
                if means["before"]
                else None,
                "worse_cases": [
                    {k: r[k] for k in ("case", "role", "source_id", "codec")}
                    for r in values
                    if r["delta"][metric] > 0
                ],
            }
        summary[group] = {"cases": len(values), "sources": len(sources), "metrics": metrics}
    return {"source_equal_summary": summary, "cases": cases}


@torch.no_grad()
def assess(root):
    destination = root / "assessment.json"
    if destination.exists():
        raise ValueError("assessment exists; preserve recorded evidence")
    checked_receipt(root / "complete.json")
    config, endpoints = read(root / "registration.json"), read(root / "endpoints.json")
    rows, _ = load_sources(list(map(Path, config["manifests"])))
    schedule = read(Path(config["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(config["detector"]), root / "cache", capacity=100),
    )
    items = [pool.select(draw) for draw in schedule["draws"][:8]]
    report = {
        "scope": "TRAIN fit diagnostic and same-crop phase-carrier ablation; not independent testing, history quality proof or model promotion",
        "seeds": {},
    }
    for seed in SEEDS:
        result = {}
        for path, key in (("raw-review", "scores"), ("native-after", "native")):

            def filename(arm, path=path, seed=seed):
                return (
                    root
                    / path
                    / (
                        f"{seed}-{arm}.json"
                        if path == "raw-review"
                        else f"{seed}-{arm}/report.json"
                    )
                )

            result[path] = compare(
                records(read(filename("zero")), key), records(read(filename("phase")), key)
            )
        native_indices = {r["case"] for r in result["native-after"]["cases"]}
        matched = [
            [
                r
                for r in records(read(root / "raw-review" / f"{seed}-{arm}.json"), "scores")
                if r["case"] in native_indices
            ]
            for arm in ("zero", "phase")
        ]
        result["raw-native-matched"] = compare(*matched)
        zero = model_for(config["parents"][str(seed)], seed)
        model = model_for(config["parents"][str(seed)], seed)
        endpoint = torch.load(endpoints[str(seed)]["path"], map_location="cpu", weights_only=False)
        if endpoint["step"] != config["steps"]:
            raise ValueError("incomplete trajectory")
        model.load_state_dict(endpoint["model"])
        diagnostics = []
        for index, item in enumerate(items):
            batch = item[0].batch
            parent, _ = predict_budget(zero, batch, item[1])
            predicted, _ = predict_budget(model, batch, item[1])
            if not torch.equal(parent[:, 0], predicted[:, 0]):
                raise ValueError("predicted magnitude drift")
            common = float(objective_value(model, item, "common"))
            handle = model.phase_head.register_forward_pre_hook(repeated_current)
            try:
                ablated, _ = predict_budget(model, batch, item[1])
                ablated_common = float(objective_value(model, item, "common"))
            finally:
                handle.remove()
            diagnostics.append(
                {
                    "index": index,
                    "common_zero": float(objective_value(zero, item, "common")),
                    "common_trained": common,
                    "common_repeated_current": ablated_common,
                    "phase_rotation_rms_rad": float(
                        (predicted[:, 1, 89:] - parent[:, 1, 89:]).square().mean().sqrt()
                    ),
                    "history_phase_difference_rms_rad": float(
                        (predicted[:, 1, 89:] - ablated[:, 1, 89:]).square().mean().sqrt()
                    ),
                    "direct_zero": float(direct_spectrum_loss(parent, batch)["total"]),
                    "direct_trained": float(direct_spectrum_loss(predicted, batch)["total"]),
                }
            )
        result["fit_and_history"] = diagnostics
        result["update_history"] = {
            "steps": endpoint["step"],
            "finite_updates": all(
                torch.isfinite(torch.tensor(h["objective_before"])).item()
                for h in endpoint["history"]
            ),
            "uphill_directions": sum(
                h["actual_directional_derivative"] > 0 for h in endpoint["history"]
            ),
            "mean_update_seconds": sum(h["seconds"] for h in endpoint["history"])
            / len(endpoint["history"]),
        }
        report["seeds"][str(seed)] = result
    write_json_report(report, destination)
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in (destination, root / "complete.json", Path(__file__).resolve())
            }
        },
        root / "assessment-receipt.json",
    )
    print("P65 fit and history assessment complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assess(args.run.resolve())
