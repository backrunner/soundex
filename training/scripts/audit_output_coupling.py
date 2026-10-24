#!/usr/bin/env python3
"""P47 frozen-output band attribution and recorded budget-gate ablation.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.fidelity_calibration import assess_wave
from evaluation.output_coupling import (
    PARTS,
    PATHS,
    budget_ablation,
    budget_coordinate_control,
    compare_outputs,
)
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import supervised_waves
from models.restoration_head import load_restoration_head, predict_budget
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered


def aggregate(cases: list, original: list) -> dict:
    values = defaultdict(lambda: defaultdict(list))
    before = defaultdict(lambda: defaultdict(list))
    violations = {path: {str(b): Counter() for b in (16, 20, 24)} for path in PATHS}
    decomposition = defaultdict(lambda: defaultdict(float))
    for case, anchor in zip(cases, original, strict=True):
        samples = case["samples"]["calibration"]
        for path, observation in case["paths"].items():
            for bits, fails in observation["violations"].items():
                violations[path][bits].update(v["role"] for v in fails)
            metrics = {
                **observation["metrics"],
                **{"direct_" + k: v for k, v in observation["direct"].items()},
                **{
                    f"band_{band['hz'][0]}_{k}": band[k]
                    for band in observation["direct_bands"]
                    for k in ("amplitude", "complex")
                },
            }
            for i in range(2 * len(samples)):
                group = samples[i % len(samples)]["row"]["corpus"] + (
                    "/identity" if i >= len(samples) else "/paired"
                )
                for metric, array in metrics.items():
                    key = path + "/" + group
                    values[key][metric].append(array[i])
                    before[key][metric].append(anchor["metrics"][metric][i])
        parts = case["decomposition"]["parts"]
        for i, rms in enumerate(case["decomposition"]["total_low4k"]["rms"]):
            group = samples[i % len(samples)]["row"]["corpus"] + (
                "/identity" if i >= len(samples) else "/paired"
            )
            decomposition[group]["total_power_sum"] += rms * rms
            for part in PARTS:
                decomposition[group][part + "_power_sum"] += parts[part]["low4k"]["rms"][i] ** 2
                decomposition[group][part + "_projection_sum"] += (
                    parts[part]["low4k_projection_share"][i] * rms * rms
                )
    changes = {}
    for group, metrics in values.items():
        changes[group] = {}
        for metric, array in metrics.items():
            old = before[group][metric]
            mean, baseline = sum(array) / len(array), sum(old) / len(old)
            changes[group][metric] = {
                "count": len(array),
                "current_mean": mean,
                "initial_mean": baseline,
                "mean_change": mean - baseline,
                "relative_mean_change": mean / baseline - 1 if baseline else None,
                "improved": sum(a < b for a, b in zip(array, old, strict=True)),
                "worsened": sum(a > b for a, b in zip(array, old, strict=True)),
            }
    for values in decomposition.values():
        power = values["total_power_sum"]
        values["projection_shares"] = {
            part: values[part + "_projection_sum"] / power if power else None for part in PARTS
        }
    return {"changes": changes, "violations": violations, "decomposition": decomposition}


def prepare(prior: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("fresh evidence directory required; preserve earlier attempts")
    config, receipt = read(prior / "registration.json"), read(prior / "complete.json")
    verify(config["hashes"])
    hashes = {str(prior / name): expected for name, expected in receipt["files"].items()}
    hashes[str(prior / "registration.json")] = receipt["registration_sha256"]
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    endpoints = {
        k: v
        for k, v in read(prior / "review-registration.json").items()
        if "-reference_" in k or k.endswith("-stored_adam") or k in config["endpoints"]
    }
    for label, entry in endpoints.items():
        hashes[entry["path"]] = entry["sha256"]
        report = prior / "review" / f"{label}.json"
        hashes[str(report)] = read(prior / "summary.json")["review_files"][label]
        if "-reference_" in label:
            updates = Path(entry["path"]).with_name("updates.json")
            hashes[str(updates)] = sha256_file(updates)
    verify(hashes)
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for file in sorted(source.rglob("*.py")):
        if any(p in (".venv", "__pycache__") for p in file.relative_to(source).parts):
            continue
        hashes[str(file)] = sha256_file(file)
        destination = output / "source" / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
    for file in (source / "evaluation").glob("*.json"):
        hashes[str(file)] = sha256_file(file)
        shutil.copy2(file, output / "source/evaluation" / file.name)
    registration = {
        "format": "soundex-output-coupling-p47-v1",
        "prior": str(prior),
        "hashes": hashes,
        "prior_input_hashes": config["hashes"],
        "endpoints": endpoints,
        "starts": config["endpoints"],
        "calibration": config["calibration"],
        "paths": PATHS,
        "parts_hz": {k: [a, b if b != float("inf") else "nyquist"] for k, (a, b) in PARTS.items()},
        "scope": "post-hoc mechanism audit on all 32 reused P44 TRAIN batches, all twelve P46 endpoints and both matched Adam controls; no new training, endpoint selection, independent test, native DSP or listening claim",
        "output_intervention": "replace candidate feature bins below 4k, or below 8k, with the corresponding frozen start; no clean target used; hard masks are attribution only, not deployment-ready crossovers",
        "decomposition": "linear float64 synthesis of complex feature changes in three disjoint bands; original float32 render discrepancy and cross terms retained; long-FFT low4k projection is offline",
        "budget_ablation": "remove only budget metric violations from recorded proposals, retain spectral progress and every other PCM/metric guard; count added attempts once across recorded radii; not a retrained counterfactual",
        "budget_gauge": "reversible output hooks double positive residual-scope budget and halve bounded decoder coordinates; residual is unchanged in real arithmetic; record actual float32 PCM difference and auxiliary risk, not a fitted parameter model",
    }
    registered(registration, output / "registration.json")
    return registration


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    prior, output = args.prior.resolve(), args.output.resolve()
    config = prepare(prior, output)
    paths = [Path(p) for p in config["prior_input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    original_cases = next(iter(read(Path(config["calibration"]))["cases"].values()))
    specs = [s["spec"] for case in original_cases for s in case["samples"]["calibration"]]
    pool = LazyPool(specs, clips)
    items = [pool.select(list(range(i * 5, i * 5 + 5))) for i in range(len(original_cases))]
    budget = {}
    for label, endpoint in config["endpoints"].items():
        if "-reference_" in label:
            history = read(Path(endpoint["path"]).with_name("updates.json"))
            budget[label] = budget_ablation(history)
    write_json_report(budget, output / "budget-ablation.json")
    reports, raw_hashes = {}, {}
    for seed, endpoint in config["starts"].items():
        initial = load_restoration_head(
            torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        ).eval()
        anchors, anchor_metrics = [], []
        for item in items:
            batch = item[0].batch
            prediction, _ = predict_budget(initial, batch, item[1])
            anchors.append(prediction)
            wave, reference = supervised_waves(prediction, batch)
            direct = direct_spectrum_loss(prediction, batch)
            anchor_metrics.append(
                {
                    "metrics": {
                        **assess_wave(wave, reference, batch["rate"]),
                        **{"direct_" + k: v.tolist() for k, v in direct["components"].items()},
                        **{
                            f"band_{b['hz'][0]}_{k}": b[k].tolist()
                            for b in direct["bands"]
                            for k in ("amplitude", "complex")
                        },
                    }
                }
            )
        for label, entry in config["endpoints"].items():
            if not label.startswith(seed + "-"):
                continue
            model = load_restoration_head(
                torch.load(entry["path"], map_location="cpu", weights_only=False)
            ).eval()
            if model.budget_scope != "residual":
                raise ValueError("budget gauge diagnostic requires residual scope")
            old = read(prior / "review" / f"{label}.json")["cases"]
            cases = []
            for index, (item, anchor) in enumerate(zip(items, anchors, strict=True)):
                prediction, predicted_budget = predict_budget(model, item[0].batch, item[1])
                result = compare_outputs(anchor, prediction, item)
                if (
                    result["paths"]["full"]["metrics"] != old[index]["paths"]["frozen_model"]
                    or result["paths"]["full"]["direct"] != old[index]["direct"]
                ):
                    raise ValueError(f"P46 raw review does not reproduce: {label} case {index}")
                gauge_prediction, gauge_budget = budget_coordinate_control(model, item)
                gauge_wave, _ = supervised_waves(gauge_prediction, item[0].batch)
                original_wave, _ = supervised_waves(prediction, item[0].batch)
                result["budget_gauge"] = {
                    "original_auxiliary": float(0.1 * budget_risk(predicted_budget, item[0].batch)),
                    "double_scale_half_coordinates_auxiliary": float(
                        0.1 * budget_risk(gauge_budget, item[0].batch)
                    ),
                    "pcm_max_difference": float((gauge_wave - original_wave).abs().max()),
                    "pcm_rms_difference": (gauge_wave.double() - original_wave.double())
                    .square()
                    .mean(-1)
                    .sqrt()
                    .tolist(),
                }
                cases.append(
                    {
                        "index": original_cases[index]["index"],
                        "samples": original_cases[index]["samples"],
                        **result,
                    }
                )
            path = output / "cases" / f"{label}.json"
            write_json_report(cases, path)
            raw_hashes[str(path)] = sha256_file(path)
            reports[label] = aggregate(cases, anchor_metrics)
            write_json_report(
                {"completed": len(reports), "total": len(config["endpoints"]) - 2, "last": label},
                output / "progress.json",
            )
            print(label, "complete; raw review exactly reproduced", flush=True)
    verify(config["hashes"])
    verify(config["prior_input_hashes"])
    write_json_report(
        {
            "reports": reports,
            "raw_files": raw_hashes,
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )
    registered(
        {
            "summary_sha256": sha256_file(output / "summary.json"),
            "budget_ablation_sha256": sha256_file(output / "budget-ablation.json"),
        },
        output / "complete.json",
    )
    write_json_report({"stage": "complete", "endpoints": len(reports)}, output / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
