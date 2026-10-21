#!/usr/bin/env python3
"""P34 one small architecture, single-seed training-only risk balance diagnosis."""

import argparse
import fcntl
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import (
    BoundedEnvelopeClips,
    LazyPool,
    allowed_starts,
    assert_isolated,
    protected_intervals,
)
from data.causal_study import load_sources
from evaluation.head_export import export_and_measure
from evaluation.joint_review import joint_review
from evaluation.multires_review import review as spectrum_review
from evaluation.reporting import sha256_file, write_json_report
from models.risk_balance import training_terms
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.bench_context_deployment import run as bench_native
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_direct_spectrum_study import MODEL, train
from scripts.run_head_study import progress

ARMS = ("control", "compressed", "compressed_identity")


def protocol(parent, *, preflight=False, followup=False, overshoot_guard=False):
    steps = 2 if preflight else 1024
    if overshoot_guard and not followup:
        raise ValueError("overshoot guard requires a frozen followup control")
    arms = (
        (("compressed_identity_guard" if overshoot_guard else "compressed_identity_mid"),)
        if followup
        else ARMS
    )
    return {
        "format": "soundex-risk-balance-diagnostic-v1",
        "scope": "single architecture/seed fixed train-crop diagnosis; not convergence or capacity ranking",
        "review_scope": "fitted and unfitted TRAINING-domain crops, separated by registered scope; no independent test or promotion",
        "arms": list(arms),
        "seeds": [20261012],
        "steps": steps,
        "primary_endpoint": steps,
        "smoke": preflight,
        "followup": followup,
        "overshoot_guard": overshoot_guard,
        "model_configs": {arm: dict(MODEL) for arm in arms},
        "loss_routes": {arm: {arm: 1} for arm in arms},
        "rate": 1e-4,
        "weight": 0.0,
        "multires_loss": parent["multires_loss"],
        "paced_arms": list(arms),
        "optimization": "fresh shared initialization; Adam 1e-4 no decay, clip 5; identical cyclic batches, 4 music/1 speech and identities; true 80-frame no-grad warmup; paired coefficient .5 fixed",
        "contrasts": "control=direct+MR+0.1 budget; compressed=log1p of each per-example direct/MR risk; compressed_identity changes identity coefficient .5 to .05; followup compressed_identity_mid uses .15; budget unchanged",
        "selection": "all final endpoints frozen before scoring; identical fixed budgets; no mid-run selection or validation access; common uncompressed MR v1, recording and crop summaries, every outlier retained",
        "stop_policy": "nonfinite/integrity/runtime failure only; no early performance stopping",
    }


def register(args, root):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        if (
            config["smoke"] != args.preflight
            or config.get("followup", False) != args.followup
            or config.get("overshoot_guard", False) != args.overshoot_guard
        ):
            raise ValueError("preflight/full drift")
        verify_inputs(config)
        return config
    parent_path = args.parent.resolve()
    parent = read(parent_path / "registration.json")
    if parent["format"] != "soundex-broad-spectrum-v1":
        raise ValueError("P33 parent required")
    if read(parent_path / "progress.json")["stage"] != "complete":
        raise ValueError("completed P33 parent required")
    config = protocol(
        parent,
        preflight=args.preflight,
        followup=args.followup,
        overshoot_guard=args.overshoot_guard,
    )
    if args.overshoot_guard:
        config["contrasts"] += (
            "; guard adds .1 times paired coefficient .5 times HF deficit-bin positive overshoot risk, fixed MR windows/scales; no gradient below reference"
        )
    required = {
        "manifest.jsonl",
        "soundex-gain-data",
        "soundex-prediction-replay",
        "soundex-analysis-trace",
        "replay-contract.onnx",
    }
    inputs = {p: h for p, h in parent["input_hashes"].items() if Path(p).name in required}
    schedule_path = Path(parent["schedule_files"]["20261012"])
    if sha256_file(schedule_path) != parent["input_hashes"][str(schedule_path)]:
        raise ValueError("schedule drift")
    schedule = read(schedule_path)
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    count = len(schedule["draws"])
    fit_draws = [i * (count - 1) // 15 for i in range(16)]
    transfer_draws = [(i + count // 32) % count for i in fit_draws]
    if args.preflight:
        fit_draws, transfer_draws = fit_draws[:1], transfer_draws[:1]
    fit = [schedule["specs"][i] for d in fit_draws for i in schedule["draws"][d]]
    protected_fit = protected_intervals(fit, rows)
    # Before scoring, advance overlapping proposals by schedule position only.
    # Keep corpus ratio and source-time separation; recordings may still recur.
    selected_draws = []
    for proposal in transfer_draws:
        for offset in range(count):
            candidate = (proposal + offset) % count
            if candidate in fit_draws or candidate in selected_draws:
                continue
            members = [schedule["specs"][i] for i in schedule["draws"][candidate]]
            if all(
                any(
                    a <= s["start"] <= b
                    for a, b in allowed_starts(rows[s["row_id"]], protected_fit)
                )
                for s in members
            ):
                selected_draws.append(candidate)
                break
        else:
            raise ValueError("no source-time isolated training review batch")
    transfer_draws = selected_draws
    transfer = [schedule["specs"][i] for d in transfer_draws for i in schedule["draws"][d]]
    if any(rows[s["row_id"]]["split"] != "train" for s in fit + transfer):
        raise ValueError("training-only diagnosis required")
    assert_isolated(fit + transfer, rows, protected_intervals(parent["development"], rows))
    assert_isolated(transfer, rows, protected_fit)
    config.update(
        fit_specs=fit,
        transfer_specs=transfer,
        fit_draw_indices=fit_draws,
        transfer_draw_indices=transfer_draws,
        specs=fit + transfer,
        development=fit + transfer,  # Legacy review slot; all rows are TRAINING.
        cache_path=str(root / "feature-cache"),
        review_groups=["fitted"] * len(fit) + ["unfitted_train"] * len(transfer),
    )
    config["coverage"] = {
        scope: {
            corpus: coverage([s for s in specs if rows[s["row_id"]]["corpus"] == corpus], rows)
            for corpus in ("music_library", "speech_library")
        }
        for scope, specs in (("fitted", fit), ("unfitted_train", transfer))
    }
    if args.followup:
        if args.control_study is None:
            raise ValueError("completed equal-budget control study required")
        control_root = args.control_study.resolve()
        control = read(control_root / "registration.json")
        if read(control_root / "progress.json")["stage"] != "complete":
            raise ValueError("control study incomplete")
        for key in ("steps", "seeds", "fit_specs", "transfer_specs", "multires_loss"):
            if control[key] != config[key]:
                raise ValueError(f"unequal control {key}")
        if any(value != MODEL for value in control["model_configs"].values()):
            raise ValueError("different control architecture")
        config["control_assessment"] = str(control_root / "assessment.json")
        config["review_scope"] += (
            "; followup selected after first-round results: diagnostic development, not fresh holdout"
        )
        for path in (
            control_root / "assessment.json",
            control_root / "registration.json",
            control_root / "review-registration.json",
        ):
            inputs[str(path)] = sha256_file(path)
        for endpoint in read(control_root / "review-registration.json").values():
            inputs[endpoint["path"]] = endpoint["sha256"]
    for p in (parent_path / "registration.json", schedule_path, args.native_binary.resolve()):
        inputs[str(p)] = sha256_file(p)
    for s in fit + transfer:
        row = rows[s["row_id"]]
        for role in ("clean", "degraded"):
            inputs[str(Path(row["_root"]) / row[f"{role}_path"])] = row[f"{role}_checksum"]
    source = Path(__file__).resolve().parents[1]
    config["source_hashes"] = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    config["input_hashes"] = inputs
    verify_inputs(config)
    for name in config["source_hashes"]:
        dest = root.parent / "source/training" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dest)
    registered(config, destination)
    return config


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args, root)
        paths = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            Path(config["cache_path"]),
            capacity=320,
        )
        pool = LazyPool(config["fit_specs"], clips)
        # Finite diagnostic pool: retain immutable tensors, no random crop or disk-cache growth.
        batches = [pool.select(list(range(i, i + 5))) for i in range(0, len(pool.rows), 5)]
        seed = config["seeds"][0]
        for arm in config["arms"]:
            progress(root, stage="training_start", arm=arm, seed=seed)
            train(arm, seed, root, config, batches, training_terms=training_terms)
        endpoints = {}
        for arm in config["arms"]:
            path = root / f"{arm}-{seed}/step-{config['steps']}.pth"
            endpoints[f"{arm}-{config['steps']}-{seed}"] = {
                "arm": arm,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        registered(endpoints, root / "review-registration.json")
        for label, value in endpoints.items():
            progress(root, stage="export", label=label)
            export_and_measure(Path(value["path"]), root / "exports" / label, iterations=100)
        pools = {"development": LazyPool(config["specs"], clips)}
        progress(root, stage="identity_dsp_control")
        joint_review(
            "initial",
            None,
            root,
            pools,
            next(p for p in paths if p.name == "soundex-prediction-replay"),
            next(p for p in paths if p.name == "replay-contract.onnx"),
            observed_binary=next(p for p in paths if p.name == "soundex-analysis-trace"),
        )
        write_json_report(
            {"status": "complete", "scope": "training endpoints frozen; native evaluation follows"},
            root / "summary.json",
        )
        progress(root, stage="native_pcm_review")
        audit_native(
            SimpleNamespace(study=root, output=root / "native-audit", binary=args.native_binary)
        )
        progress(root, stage="native_spectrum_review")
        spectrum_review(root, args.native_binary)
        if not args.preflight:
            progress(root, stage="native_benchmark")
            bench_native(
                SimpleNamespace(study=root, output=root / "native-cost", binary=args.native_binary)
            )
        verify_inputs(config)
        progress(root, stage="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "native-binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--overshoot-guard", action="store_true")
    parser.add_argument("--control-study", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        write_json_report(
            {"stage": "failed", "error": traceback.format_exc()},
            args.output / "pipeline-error.json",
        )
        raise
