#!/usr/bin/env python3
"""P54 fit-only persistent-Adam calibration and matched broad-crop continuation.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import importlib.metadata
import shutil
import time
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, make_broad_schedule
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import assert_frozen, load_endpoint
from evaluation.adapter_runtime import export, native_review
from evaluation.batch_update_probe import average
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_training import OBJECTIVES, objective_value, update
from models.spectral_adapter import load_adapter
from scripts.audit_guarded_transfer import measured
from scripts.compare_historical_spectra import aggregate, scores
from scripts.run_adapter_pilot import checkpoint
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_guarded_transfer import coverage

RATES = (1e-7, 1e-6, 1e-5)
CALIBRATION_STEPS = 16
TRAINING_STEPS = 4096


def training_roles(rows, old):
    """Connected recording/source aliases, within the existing P50 fit role only."""
    samples = [s for c in old["cases"]["fit"] for s in c["samples"]["calibration"]]
    allowed = {key: {s["row"][key] for s in samples} for key in ("track_id", "source_checksum")}
    review = [s for c in old["cases"]["review"] for s in c["samples"]["calibration"]]
    excluded = {key: {s["row"][key] for s in review} for key in allowed}
    if any(allowed[k] & excluded[k] for k in allowed):
        raise ValueError("existing recording/source split leaks")
    eligible = {k: r for k, r in rows.items() if all(r[f] in allowed[f] for f in allowed)}
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]

    for row in eligible.values():
        a, b = root("track:" + row["track_id"]), root("source:" + row["source_checksum"])
        parents[max(a, b)] = min(a, b)
    result = {"optimization": {}, "calibration": {}}
    for key, row in eligible.items():
        component = root("track:" + row["track_id"])
        role = "calibration" if int(digest(component)[:16], 16) % 5 == 0 else "optimization"
        result[role][key] = row
    for field in allowed:
        left = {r[field] for r in result["optimization"].values()}
        right = {r[field] for r in result["calibration"].values()}
        if left & right or (left | right) & excluded[field]:
            raise ValueError("new role split leaks")
    return result


def prepare(prior, output):
    path = output / "registration.json"
    if path.exists():
        config = read(path)
        verify(config["hashes"])
        if config["prior"] != str(prior):
            raise ValueError("prior changed")
        return config
    old = read(prior / "registration.json")
    verify(read(prior / "complete.json")["files"])
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    partitions = training_roles(rows, old)
    schedules, exposure = {}, {}
    for role, count, seed in (
        ("optimization", TRAINING_STEPS * 4, 20261054),
        ("calibration", 16, 20261055),
    ):
        specs, draws, recordings = make_broad_schedule(partitions[role], [], count, seed)
        schedules[role] = {"specs": specs, "draws": draws}
        samples = [{"spec": s, "row": rows[s["row_id"]]} for s in specs]
        exposure[role] = coverage([{"samples": {"registered": samples}}], "registered")
        exposure[role]["eligible_recordings"] = recordings
    schedule = output / "schedule.json"
    registered(schedules, schedule)
    root = Path(__file__).resolve().parents[1]
    hashes = {str(schedule): sha256_file(schedule)}
    for source in root.rglob("*.py"):
        if any(p in {".venv", "__pycache__", "outputs"} for p in source.relative_to(root).parts):
            continue
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(target)] = sha256_file(target)
        hashes[str(source)] = sha256_file(source)
    for path_to_bind in [
        prior / "registration.json",
        prior / "complete.json",
        *map(Path, old["manifests"]),
        Path(old["detector"]),
        Path(old["binary"]),
        Path(old["schedule"]),
        *(Path(e["path"]) for e in old["parents"].values()),
        *(Path(e["path"]) for e in old["starts"].values()),
        root / "requirements.txt",
        root / "constraints-cpu-py312.txt",
    ]:
        hashes[str(path_to_bind)] = sha256_file(path_to_bind)
    config = {
        "format": "soundex-spectral-training-p54-v1",
        "prior": str(prior),
        "parents": old["parents"],
        "starts": old["starts"],
        "manifests": old["manifests"],
        "detector": old["detector"],
        "binary": old["binary"],
        "review_schedule": old["schedule"],
        "review_cases": old["cases"]["review"],
        "schedule": str(schedule),
        "coverage": exposure,
        "hashes": hashes,
        "steps": TRAINING_STEPS,
        "calibration_steps": CALIBRATION_STEPS,
        "rates": RATES,
        "objectives": OBJECTIVES,
        "packages": {k: importlib.metadata.version(k) for k in ("auraloss", "torch", "numpy")},
        "optimizer": "True Adam lr from fixed grid, betas .9/.999 eps 1e-8, clip 5, persistent moments. Four equal batches per update. No post-step multiplier, line search, gradient projection, veto or reset during a trajectory.",
        "calibration": "Complete all 12 trajectories (two seeds x two objectives x three rates), 16 steps each. Freeze endpoint hashes before scoring 16 source/recording-disjoint TRAIN calibration batches. Select ONE shared rate minimizing worst endpoint/start own-objective ratio over both seeds/objectives, requiring every ratio <1. Tie picks smaller rate. No paired metric or PCM veto; retain their regressions. If no rate descends for all four, stop with diagnosis; no silent fallback.",
        "training": "Reset all four trajectories to the same P50 parents and fresh moments after calibration. Same selected rate, 4096 steps and 16384 newly drawn broad-crop batches each, including real warmup. 32 times the prior continuation budget tests cumulative optimization and broader exposure; not proof of convergence or a full-backbone training run. No changed data, objective or budget after review.",
        "objective": "Common versus auraloss rendered spectral term only; same direct HF amplitude/complex/envelope/dynamics target, equal paired/identity reduction. Omit constant frozen-budget auxiliary during optimization. Always rescore both with both objectives and historical common metrics; no cross-objective total comparison.",
        "review": "Finish and hash all four endpoints before historical TRAIN review. Same model-only and native DSP materials, original reference, all per-example counterexamples. These sources were used by older backbone training and diagnostics; no independent generalization, blind-listening, stereo or device claim. No automatic weight promotion.",
        "question": "Does persistent Adam at a TRAIN-calibrated actual learning rate learn more than tiny 128-step updates, and does auraloss help under matched rate/budget? Not a repeat of reset first-step backtracking or added preservation penalties.",
    }
    registered(config, path)
    return config


@torch.no_grad()
def measure(model, items, objective):
    return {
        "objective": sum(float(objective_value(model, item, objective)) for item in items)
        / len(items),
        "common": average([measured(model, item) for item in items]),
    }


def trajectory(config, pool, seed, objective, rate, steps, directory):
    signature = digest(
        {"config": config, "seed": seed, "objective": objective, "rate": rate, "steps": steps}
    )
    final, receipt = directory / "endpoint.pth", directory / "complete.json"
    if receipt.exists():
        result = read(receipt)
        if result["signature"] != signature or result["sha256"] != sha256_file(final):
            raise ValueError("completed trajectory drift")
        return {"path": str(final), "sha256": result["sha256"]}
    directory.mkdir(parents=True, exist_ok=True)
    latest, latest_receipt = directory / "latest.pth", directory / "latest.json"
    saved = None
    if latest.exists():
        if read(latest_receipt)["sha256"] != sha256_file(latest):
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, weights_only=False, map_location="cpu")
        if saved["signature"] != signature or saved["step"] != len(saved["history"]):
            raise ValueError("resume registration/history drift")
    model = (load_adapter(saved) if saved else load_endpoint(config["parents"][seed])).eval()
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-8)
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
    initial = torch.load(config["starts"][seed]["path"], weights_only=False, map_location="cpu")
    draws = read(Path(config["schedule"]))["optimization"]["draws"]
    for index in range(len(history), steps):
        started = time.monotonic()
        items = [pool.select(d) for d in draws[4 * index : 4 * index + 4]]
        result = update(model, items, optimizer, objective)
        history.append({"step": index + 1, **result, "seconds": time.monotonic() - started})
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == steps:
            assert_frozen(model, initial)
            atomic_save(checkpoint(model, initial, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, latest_receipt)
            write_json_report(history, directory / "updates.json")
            progress = {
                "trajectory": directory.name,
                "step": index + 1,
                "steps": steps,
                "seconds_per_step": history[-1]["seconds"],
            }
            write_json_report(progress, directory.parents[1] / "progress.json")
            print(progress, flush=True)
    assert_frozen(model, initial)
    atomic_save(checkpoint(model, initial, optimizer, history, signature), final)
    result = {"signature": signature, "sha256": sha256_file(final)}
    registered(result, receipt)
    return {"path": str(final), "sha256": result["sha256"]}


def calibrate(config, optimization, calibration, output):
    path = output / "calibration.json"
    if path.exists():
        if read(output / "calibration.receipt.json")["sha256"] != sha256_file(path):
            raise ValueError("calibration drift")
        return read(path)
    entries = {}
    for seed in config["parents"]:
        for objective in OBJECTIVES:
            for rate in config["rates"]:
                label = f"{seed}-{objective}-{rate:g}"
                entries[label] = {
                    **trajectory(
                        config,
                        optimization,
                        seed,
                        objective,
                        rate,
                        config["calibration_steps"],
                        output / "calibration-training" / label,
                    ),
                    "seed": seed,
                    "objective": objective,
                    "rate": rate,
                }
    registered(entries, output / "calibration-endpoints.json")
    schedule = read(Path(config["schedule"]))["calibration"]
    items = [calibration.select(d) for d in schedule["draws"]]
    starts, measured_endpoints = {}, {}
    for seed, entry in config["parents"].items():
        model = load_endpoint(entry)
        starts[seed] = {objective: measure(model, items, objective) for objective in OBJECTIVES}
    for label, entry in entries.items():
        measured_endpoints[label] = measure(load_endpoint(entry), items, entry["objective"])
    worst = {}
    for rate in config["rates"]:
        ratios = [
            measured_endpoints[k]["objective"] / starts[e["seed"]][e["objective"]]["objective"]
            for k, e in entries.items()
            if e["rate"] == rate
        ]
        worst[rate] = max(ratios)
    eligible = [rate for rate in config["rates"] if worst[rate] < 1]
    selected = min(eligible, key=lambda rate: (worst[rate], rate)) if eligible else None
    result = {
        "starts": starts,
        "endpoints": measured_endpoints,
        "entries": entries,
        "worst_objective_ratio": worst,
        "selected_shared_rate": selected,
        "scope": "TRAIN calibration only, after all 12 fixed endpoints; not test quality or convergence.",
    }
    registered(result, path)
    registered({"sha256": sha256_file(path)}, output / "calibration.receipt.json")
    return result


@torch.no_grad()
def review(config, entries, output):
    from models.multires_spectrum import supervised_waves
    from models.restoration_head import predict_budget

    registered(entries, output / "review-registration.json")
    verify({e["path"]: e["sha256"] for e in entries.values()})
    rows, _ = load_sources(list(map(Path, config["manifests"])))
    schedule = read(Path(config["review_schedule"]))
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(config["detector"]), output / "review-cache", capacity=320),
    )
    items = [pool.select(d) for d in schedule["review"]]
    all_entries = {
        **{seed + "-parent": entry for seed, entry in config["parents"].items()},
        **entries,
    }
    graphs, summary = {}, {}
    for label, entry in all_entries.items():
        model = load_endpoint(entry)
        export(model, output / "exports" / label)
        graphs[label] = output / "exports" / label / "diagnostic.onnx"
        cases = []
        for index, item in enumerate(items):
            batch = item[0].batch
            prediction, _ = predict_budget(model, batch, item[1])
            wave, target = supervised_waves(prediction, batch)
            cases.append(
                {
                    "index": index,
                    "samples": config["review_cases"][index]["samples"],
                    "corpora": [
                        rows[schedule["specs"][i]["row_id"]]["corpus"]
                        for i in schedule["review"][index]
                    ],
                    "identity": batch["identity"].bool().tolist(),
                    "scores": scores(wave, target, batch["rate"]),
                    "common": measured(model, item),
                }
            )
        write_json_report(
            {"checkpoint": entry, "cases": cases}, output / "review" / f"{label}.json"
        )
        summary[label] = aggregate(cases)
    native_review(
        Path(config["binary"]), graphs, items, config["review_cases"], output / "native-review"
    )
    write_json_report(summary, output / "summary.json")
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file()
                and p.name not in {"run.lock", "complete.json", "progress.json"}
                and "feature-cache" not in p.parts
                and "review-cache" not in p.parts
            }
        },
        output / "complete.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), output)
        if any(importlib.metadata.version(k) != v for k, v in config["packages"].items()):
            raise ValueError("registered package version changed")
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            return
        if args.prepare_only:
            print("registered", config["coverage"], flush=True)
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))
        clips = BoundedEnvelopeClips(
            rows, Path(config["detector"]), output / "feature-cache", capacity=100
        )
        optimization = LazyPool(schedule["optimization"]["specs"], clips)
        calibration = LazyPool(schedule["calibration"]["specs"], clips)
        result = calibrate(config, optimization, calibration, output)
        print("calibration complete", result["selected_shared_rate"], flush=True)
        if args.calibrate_only:
            return
        rate = result["selected_shared_rate"]
        if rate is None:
            raise ValueError("no shared rate descends on TRAIN calibration; retain and diagnose")
        entries = {}
        for seed in config["parents"]:
            for objective in OBJECTIVES:
                label = f"{seed}-{objective}"
                entries[label] = trajectory(
                    config,
                    optimization,
                    seed,
                    objective,
                    rate,
                    config["steps"],
                    output / "training" / label,
                )
        review(config, entries, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    run(parser.parse_args())
