#!/usr/bin/env python3
"""P22: matched train-only capacity extension selected by CPU cost, never dev loss."""

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.coverage_study import PoolView
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.head_study import evaluate
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_head_study import progress, train
from scripts.run_subset_study import Pool
from scripts.screen_latency_capacity import qualifies


def register(parent, screen, root):
    config = read(parent / "registration.json")
    screening = read(screen / "summary.json")
    widths = sorted(int(w) for w, r in screening["results"].items() if qualifies(r["report"]))
    if widths != sorted(screening["eligible_widths"]) or not any(w > 224 for w in widths):
        raise ValueError("no larger capacity passed cost screen, or screening drift")
    configs = {"anchored": {"arm": "anchored", "width": 144, "residual_slope": 1.0}}
    configs.update(
        {f"scaled{w}": {"arm": "complex", "width": w, "residual_slope": 1 / 64} for w in widths}
    )
    inputs = dict(config["input_hashes"])
    for path in (
        parent / "registration.json",
        screen / "summary.json",
        screen / "registration.json",
    ):
        inputs[str(path.resolve())] = sha256_file(path)
    source = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    config = {
        **{k: config[k] for k in ("seeds", "bank", "common", "weight", "schedule_hashes")},
        "format": "soundex-latency-capacity-v1",
        "scope": "512-step train-only optimization calibration; no held-out quality or promotion claim",
        "steps": 512,
        "endpoints": [512],
        "score_interval": 512,
        "rate": 1e-4,
        "model_configs": configs,
        "arms": list(configs),
        "source_hashes": sources,
        "input_hashes": inputs,
        "selection": "all CPU-cost passing sizes, same two seeds/draws/steps/objective; stop for diagnosis after calibration, no automatic long run or winner selection",
    }
    for seed in config["seeds"]:
        path = parent / f"draws-{seed}.json"
        inputs[str(path.resolve())] = sha256_file(path)
        registered(read(path), root / path.name)
    registered(config, root / "registration.json")
    verify_inputs(config)
    return config


def summarize(root, config):
    initial = read(root / "initial-bank.json")
    results = []
    for seed in config["seeds"]:
        control = read(root / f"anchored-{seed}/training-512.json")
        for arm in config["arms"]:
            directory = root / f"{arm}-{seed}"
            final = read(directory / "training-512.json")
            start = read(directory / "training-0.json")["common"]
            updates = read(directory / "updates.json")
            clipped = float(np.mean([u["preclip_gradient_norm"] > 5 for u in updates]))
            checks = {"clipping_at_most_25_percent": clipped <= 0.25}
            for scope, before in (("common", start), ("arm_bank", initial)):
                after, reference = final[scope]["weighted"], control[scope]["weighted"]
                for metric in ("total", "paired_restoration"):
                    checks[f"{scope}_{metric}_improved"] = (
                        after[metric] < before["weighted"][metric] - 1e-6
                    )
                checks[f"{scope}_identity_not_worse"] = (
                    after["identity_preservation"] <= reference["identity_preservation"] + 1e-6
                )
            checks["common_combined_improved"] = (
                final["common"]["combined"] < start["combined"] - 1e-6
            )
            report = read(root / "exports" / f"{arm}-{seed}/report.json")
            checks["trained_cpu_cost_passed"] = qualifies(report)
            results.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "clipped_fraction": clipped,
                    "scores": final,
                    "checkpoint_sha256": sha256_file(directory / "step-512.pth"),
                }
            )
    return {"status": "calibration_complete", "scope": config["scope"], "results": results}


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args.parent, args.screen, root)
        if args.prepare_only:
            return
        if args.wait_for:
            while True:
                state = read(args.wait_for)
                if state["status"] == "complete":
                    break
                if state["status"] == "failed":
                    raise RuntimeError("preceding pipeline failed; inspect before training")
                progress(root, stage="queued", waiting_for=str(args.wait_for))
                time.sleep(30)
        verify_inputs(config)
        inputs = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
        detector = next(p for p in inputs if p.name == "soundex-gain-data")
        parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
        clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
        progress(root, stage="prepare_training_bank", positions=len(config["bank"]))
        pool = Pool(config["bank"], clips)
        common = PoolView(pool, config["common"])
        initial = root / "initial-bank.json"
        if not initial.exists():
            write_json_report(evaluate(None, pool, config["weight"]), initial)
        for seed in config["seeds"]:
            for arm in config["arms"]:
                train(arm, seed, root, config, pool, common)
                export_and_measure(
                    root / f"{arm}-{seed}/step-512.pth",
                    root / "exports" / f"{arm}-{seed}",
                    experimental_unbounded=True,
                    iterations=10000,
                )
        verify_inputs(config)
        write_json_report(summarize(root, config), root / "summary.json")
        progress(root, stage="calibration_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
