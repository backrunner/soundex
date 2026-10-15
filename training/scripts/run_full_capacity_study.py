#!/usr/bin/env python3
"""P23: full-budget capacity comparison, without the 512-step optimization gate."""

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
from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_head_study import run as analyze
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered, verify_receipt
from scripts.run_head_study import progress, train
from scripts.run_subset_study import Pool

STEPS = 16192
ENDPOINTS = [2048, 8192, STEPS]
ARMS = ["scaled640", "scaled384", "scaled224", "anchored"]
COMPARISONS = [("anchored", arm) for arm in ARMS[:-1]] + [
    ("scaled224", "scaled384"),
    ("scaled384", "scaled640"),
]


def protocol(parent, original):
    for key in ("bank", "common", "seeds", "weight", "rate"):
        if parent[key] != original[key]:
            raise ValueError(f"parent training definition differs: {key}")
    if set(parent["arms"]) != set(ARMS):
        raise ValueError("capacity arms changed")
    for arm, width in zip(ARMS, (640, 384, 224, 144), strict=True):
        expected = {
            "arm": "anchored" if arm == "anchored" else "complex",
            "width": width,
            "residual_slope": 1.0 if arm == "anchored" else 1 / 64,
        }
        if parent["model_configs"][arm] != expected:
            raise ValueError("capacity model definition changed")
    return {
        **{k: parent[k] for k in ("bank", "common", "seeds", "weight", "rate", "model_configs")},
        **{k: original[k] for k in ("development", "shifted", "tolerance")},
        "format": "soundex-full-capacity-v1",
        "scope": "full-budget matched capacity study; reused development review, not independent test; no automatic promotion",
        "arms": ARMS,
        "steps": STEPS,
        "endpoints": ENDPOINTS,
        "primary_endpoint": STEPS,
        "score_interval": 2048,
        "smoke": False,
        "native_count": (1 + len(ARMS) * len(parent["seeds"]) * len(ENDPOINTS))
        * len(original["development"])
        * 2,
        "initialization": "fresh shared-seed identity initialization, not selected calibration weights",
        "selection": "all arms/seeds run full budget, no train-loss or 512-step early stop; freeze all endpoints before any reused development score; intermediate endpoints describe trajectory, not best-checkpoint selection",
        "optimization": "unchanged main plus clean-complex auxiliary, Adam constant 1e-4, no weight decay, clip norm 5; 4 music + 1 speech, paired/identity equal",
        "stop_policy": "stop on nonfinite loss/gradient, input/source drift, or execution failure; retain recovery checkpoint; no automatic quality promotion",
    }


def register(parent_path, original_path, root):
    parent, original = [read(p / "registration.json") for p in (parent_path, original_path)]
    if read(parent_path / "summary.json")["status"] != "calibration_complete":
        raise ValueError("parent calibration incomplete")
    config = protocol(parent, original)
    inputs = {**parent["input_hashes"], **original["input_hashes"]}
    for path in (
        parent_path / "registration.json",
        parent_path / "summary.json",
        original_path / "registration.json",
        original_path / "summary.json",
    ):
        inputs[str(path.resolve())] = sha256_file(path)
    for seed in config["seeds"]:
        parent_draws = read(parent_path / f"draws-{seed}.json")["draws"]
        path = original_path / f"draws-{seed}.json"
        draws = read(path)["draws"]
        if len(draws) != STEPS or draws[: parent["steps"]] != parent_draws:
            raise ValueError("full schedule length or calibration prefix changed")
        if digest(draws) != original["schedule_hashes"][str(seed)]:
            raise ValueError("original full schedule hash changed")
        inputs[str(path.resolve())] = sha256_file(path)
        prefix = parent_path / f"draws-{seed}.json"
        inputs[str(prefix.resolve())] = sha256_file(prefix)
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
    registered(config, root / "registration.json")
    verify_inputs(config)
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
        config = register(args.parent, args.p20, root)
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
    parser.add_argument("--p20", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
