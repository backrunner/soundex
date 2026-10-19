#!/usr/bin/env python3
"""P31 small 2x2 phase-input / residual-coordinate learnability diagnostic."""

import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import RestorationHead
from scripts.bench_context_deployment import run as bench_native
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_direct_spectrum_study import MODEL
from scripts.run_direct_spectrum_study import run as run_fit
from scripts.run_head_study import progress

ARMS = ("relative", "relative_phase", "cartesian", "cartesian_phase")


def protocol(parent, *, preflight=False):
    steps = 4 if preflight else 512
    return {
        **{k: parent[k] for k in ("specs", "weight", "rate", "direct_loss", "optimization")},
        "format": "soundex-phase-coordinates-fit-v1",
        "scope": "fixed P29 TRAINING crops; small-model early learnability only, not convergence/capacity/generalization or release",
        "review_scope": "all fixed endpoints on reused fitted TRAINING crops, not independent evaluation",
        "arms": list(ARMS),
        "seeds": parent["seeds"][:1] if preflight else parent["seeds"],
        "steps": steps,
        "primary_endpoint": steps,
        "smoke": preflight,
        "model_configs": {
            arm: {
                **MODEL,
                "phase_input": arm.endswith("_phase"),
                "residual_coordinates": "global_cartesian"
                if arm.startswith("cartesian")
                else "input_relative",
            }
            for arm in ARMS
        },
        "optimization_arms": dict.fromkeys(ARMS, "raw"),
        "development": parent["specs"],
        "shifted": [],
        "paced_arms": list(ARMS),
        "contrasts": "2x2: observed complex carrier input off/on and input-relative/global-Cartesian residual coordinates; same direct raw loss plus 0.1 budget teacher; no deployed loss gradient",
        "initialization": "same seed and all common tensors; zero decoder and zero phase projection give exact identity; phase input adds 258*96 parameters, an explicit capacity confound",
        "selection": "all eight final endpoints frozen before common scoring; no mid-run selection, extension, default replacement or cross-recording claim",
        "stop_policy": "nonfinite, integrity or runtime failure only; no early performance stopping",
        "timing_policy": "after all fits and native review, all endpoints: full Rust compute, mono/stereo, 44.1/48 kHz and paced stereo 48 kHz; <5 ms service and 2.667 ms hop reported separately; fixed 256-frame buffer excluded from compute",
    }


def register(args, root):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["smoke"] != args.preflight:
            raise ValueError("preflight/full protocol drift")
        verify_inputs(config)
        return config
    parent_path = args.parent.resolve()
    parent = read(parent_path / "registration.json")
    if (
        parent["format"] != "soundex-direct-spectrum-fit-v1"
        or read(parent_path / "progress.json")["stage"] != "complete"
    ):
        raise ValueError("completed P29 parent required")
    for rel, expected in parent["source_hashes"].items():
        if sha256_file(parent_path.parent / "source/training" / rel) != expected:
            raise ValueError("parent frozen source drift")
    config = protocol(parent, preflight=args.preflight)
    inputs = dict(parent["input_hashes"])
    for p in (
        parent_path / "registration.json",
        parent_path / "progress.json",
        parent_path / "coverage.json",
        args.native_binary.resolve(),
    ):
        inputs[str(p)] = sha256_file(p)
    config["input_hashes"] = inputs
    config["cache_path"] = str(root / "feature-cache")
    source = Path(__file__).resolve().parents[1]
    config["source_hashes"] = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    config["parameters"] = {}
    for seed in config["seeds"]:
        models = {}
        for arm, options in config["model_configs"].items():
            torch.manual_seed(seed)
            models[arm] = RestorationHead(**options)
            config["parameters"][arm] = sum(p.numel() for p in models[arm].parameters())
        for arm in ARMS[1:]:
            for key, value in models[ARMS[0]].state_dict().items():
                if not torch.equal(value, models[arm].state_dict()[key]):
                    raise ValueError("unmatched common initialization")
    verify_inputs(config)
    registered(config, destination)
    registered(read(parent_path / "coverage.json"), root / "coverage.json")
    return config


def run(args):
    run_fit(
        args,
        register_study=register,
        completion_stage="complete" if args.preflight else "native_review_complete",
    )
    if args.prepare_only or args.preflight:
        return
    progress(args.output, stage="native_compute_benchmark")
    bench_native(
        SimpleNamespace(
            study=args.output, output=args.output / "native-cost", binary=args.native_binary
        )
    )
    progress(args.output, stage="complete", preflight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        write_json_report(
            {"stage": "failed", "error": traceback.format_exc()},
            args.output / "pipeline-error.json",
        )
        raise
