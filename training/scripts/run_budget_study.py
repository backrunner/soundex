#!/usr/bin/env python3
"""P24: full-budget 640-wide loss / generation-budget factorial comparison."""

import argparse
import fcntl
import gc
from pathlib import Path

import torch

from data.causal_study import digest, load_sources
from data.coverage_study import PoolView
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.head_study import review
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_head_study import run as analyze
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered, verify_receipt
from scripts.run_head_study import progress, train
from scripts.run_subset_study import Pool

STEPS = 16192
ENDPOINTS = [2048, 8192, STEPS]
ARMS = ["joint", "control", "balanced", "budget"]
COMPARISONS = [
    ("control", "balanced"),
    ("control", "budget"),
    ("control", "joint"),
    ("balanced", "joint"),
    ("budget", "joint"),
]


def protocol(parent):
    base = {"arm": "complex", "width": 640, "residual_slope": 1 / 64}
    return {
        **{
            k: parent[k]
            for k in (
                "bank",
                "common",
                "seeds",
                "weight",
                "rate",
                "development",
                "shifted",
                "tolerance",
            )
        },
        "format": "soundex-budget-factorial-v1",
        "scope": "same-width factorial; reused development, not independent test; no promotion",
        "arms": ARMS,
        "steps": STEPS,
        "endpoints": ENDPOINTS,
        "primary_endpoint": STEPS,
        "score_interval": 2048,
        "smoke": False,
        "model_configs": {
            arm: {**base, "generation_budget": arm in ("budget", "joint")} for arm in ARMS
        },
        "balanced_arms": ["balanced", "joint"],
        "budget_weights": {"budget": 0.1, "joint": 0.1},
        "native_count": (1 + len(ARMS) * len(parent["seeds"]) * len(ENDPOINTS))
        * len(parent["development"])
        * 2,
        "joint_native_count": (2 + len(ARMS) * len(parent["seeds"]) * len(ENDPOINTS))
        * len(parent["development"])
        * 2,
        "selection": "all 8 fresh trajectories complete 16192 updates; freeze 24 endpoints before review; primary final endpoint; no best-checkpoint selection or early loss gate",
        "optimization": "Adam 1e-4, no decay, clip norm 5; same frozen 4 music + 1 speech draws and equal paired/identity risk",
        "objective": "control/budget: historical main+complex auxiliary; balanced/joint: band+global reference power normalization in main AND auxiliary, plus 0.25 full-scale smooth PCM RMS; budget/joint add 0.1 log1p residual-energy supervision",
        "budget": "5 input-only softplus band budgets, initial 1, continuous frequency interpolation; no hard ceiling, target not in forward; raw clean complex residual / input RMS is teacher only",
        "evaluation": "all arms under unchanged historical relative metrics, common balanced proxy, absolute native PCM/dBFS, low/HF error, temporal and energy; paired original clean and identity-DSP separately; clip/recording, music/speech separately; no deletion/threshold tuning/automatic release",
        "stop_policy": "nonfinite, source/input drift or runtime error only; budget completion is not proof of convergence",
    }


def register(parent_path, root):
    parent = read(parent_path / "registration.json")
    if read(parent_path / "summary.json")["status"] != "complete":
        raise ValueError("P23 incomplete")
    config = protocol(parent)
    inputs = dict(parent["input_hashes"])
    for path in (parent_path / "registration.json", parent_path / "summary.json"):
        inputs[str(path.resolve())] = sha256_file(path)
    for seed in config["seeds"]:
        path = parent_path / f"draws-{seed}.json"
        draws = read(path)["draws"]
        if len(draws) != STEPS or digest(draws) != parent["schedule_hashes"][str(seed)]:
            raise ValueError("parent schedule drift")
        inputs[str(path.resolve())] = sha256_file(path)
        registered({"draws": draws}, root / path.name)
        config.setdefault("schedule_hashes", {})[str(seed)] = digest(draws)
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
    registered(config, root / "registration.json")
    return config


def freeze_endpoints(root, config):
    checkpoints = {}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            directory = root / f"{arm}-{seed}"
            verify_receipt(directory, "files")
            if len(read(directory / "updates.json")) != config["steps"]:
                raise ValueError("full training budget incomplete")
            for step in config["endpoints"]:
                path = directory / f"step-{step}.pth"
                expected = read(directory / f"registration-{step}.json")
                checksum = sha256_file(path)
                if expected != {"step": step, "checkpoint_sha256": checksum}:
                    raise ValueError("endpoint registration drift")
                checkpoints[f"{arm}-{step}-{seed}"] = {
                    "arm": arm,
                    "path": str(path),
                    "sha256": checksum,
                }
    registered(checkpoints, root / "review-registration.json")
    return checkpoints


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args.parent, root)
        if args.prepare_only:
            return
        inputs = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
        detector = next(p for p in inputs if p.name == "soundex-gain-data")
        parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
        clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
        progress(root, stage="prepare_training_bank", positions=len(config["bank"]))
        pool = Pool(config["bank"], clips)
        common = PoolView(pool, config["common"])
        for seed in config["seeds"]:
            for arm in config["arms"]:
                train(arm, seed, root, config, pool, common)
        del common, pool
        gc.collect()
        verify_inputs(config)
        checkpoints = freeze_endpoints(root, config)
        pools = {name: Pool(config[name], clips) for name in ("development", "shifted")}
        binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
        onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
        review("initial", None, None, root, pools, binary, onnx, config["weight"])
        joint_review("initial", None, root, pools, binary, onnx)
        for label, value in checkpoints.items():
            review(
                label,
                Path(value["path"]),
                value["arm"],
                root,
                pools,
                binary,
                onnx,
                config["weight"],
            )
            joint_review(label, Path(value["path"]), root, pools, binary, onnx)
            progress(root, stage="export_and_measure", case=label)
            export_and_measure(
                Path(value["path"]),
                root / "exports" / label,
                experimental_unbounded=True,
                iterations=10000,
            )
        verify_inputs(config)
        write_json_report(
            {
                "status": "complete",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )
        analyze(root, comparisons=COMPARISONS, source=Path(__file__).resolve().parents[1])
        progress(root, stage="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
