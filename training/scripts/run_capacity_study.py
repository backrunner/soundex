#!/usr/bin/env python3
"""P21: train-only residual calibration, then a frozen three-width comparison."""

import argparse
import fcntl
import gc
from pathlib import Path

import numpy as np
import torch

from artifact_contract import MAX_PARAMETERS
from data.causal_study import digest, load_sources
from data.coverage_study import PoolView
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.head_study import evaluate, review
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import RestorationHead
from scripts.analyze_head_study import run as analyze
from scripts.run_consistency_followup import read, registered
from scripts.run_head_study import progress, train
from scripts.run_subset_study import Pool


def variants():
    return {
        "anchored": {"arm": "anchored", "width": 144, "residual_slope": 1.0},
        "unscaled": {"arm": "complex", "width": 144, "residual_slope": 1.0},
        "lowrate": {"arm": "complex", "width": 144, "residual_slope": 1.0},
        **{
            f"scaled{width}": {"arm": "complex", "width": width, "residual_slope": 1 / 64}
            for width in (96, 144, 224)
        },
    }


def register(args, root):
    parent = read(args.p20 / "registration.json")
    inputs = dict(parent["input_hashes"])
    for path in (
        args.p20 / "registration.json",
        args.p20 / "summary.json",
        *args.p20.glob("draws-*.json"),
    ):
        inputs[str(path.resolve())] = sha256_file(path)
    for name, expected in inputs.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("registered input drift")
    source = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    model_configs = variants()
    parameters = {
        name: sum(p.numel() for p in RestorationHead(**value).parameters())
        for name, value in model_configs.items()
    }
    if max(parameters.values()) > MAX_PARAMETERS:
        raise ValueError("parameter budget exceeded")
    config = {
        "format": "soundex-capacity-study-v1",
        "scope": "train-only calibration followed by reused development review; no promotion",
        "model_configs": model_configs,
        "parameters": parameters,
        "rates": {name: (1e-4 / 64 if name == "lowrate" else 1e-4) for name in model_configs},
        "rate": 1e-4,
        "weight": parent["weight"],
        "seeds": parent["seeds"],
        "bank": parent["bank"],
        "common": parent["common"],
        "development": parent["development"],
        "shifted": parent["shifted"],
        "input_hashes": inputs,
        "source_hashes": sources,
        "smoke": False,
        "tolerance": 1e-6,
        "calibration_steps": 512,
        "full_steps": 8192,
        "full_endpoints": [2048, 8192],
        "full_arms": ["anchored", "scaled96", "scaled144", "scaled224"],
        "calibration_gate": "both seeds at every scaled width: common total, paired and combined plus full-bank total/paired below initialization; common and full-bank identity <= same-seed anchored + 1e-6; clipping <=25%; parameters <=2M; ONNX <8MiB and stereo combined p99 <128/48ms. Process screen only, not a quality/promotion gate.",
        "selection": "No best checkpoint/width or development selection. Full fresh comparison runs only if all three scaled widths pass train-only gate; otherwise stop with diagnostics.",
    }
    for phase, steps, arms, endpoints in (
        ("calibration", 512, list(model_configs), [512]),
        ("full", 8192, config["full_arms"], config["full_endpoints"]),
    ):
        directory = root / phase
        directory.mkdir(exist_ok=True)
        phase_config = {
            **config,
            "arms": arms,
            "steps": steps,
            "endpoints": endpoints,
            "score_interval": 512 if phase == "calibration" else 2048,
            "native_count": (1 + len(arms) * len(endpoints) * len(config["seeds"]))
            * len(config["development"])
            * 2,
        }
        for seed in config["seeds"]:
            draws = read(args.p20 / f"draws-{seed}.json")["draws"][:steps]
            if len(draws) != steps:
                raise ValueError("insufficient frozen draws")
            registered({"draws": draws}, directory / f"draws-{seed}.json")
            phase_config.setdefault("schedule_hashes", {})[str(seed)] = digest(draws)
        registered(phase_config, directory / "registration.json")
    registered(config, root / "registration.json")
    return config


def calibration_checks(config, root):
    results = []
    initial_bank = read(root / "initial-bank.json")
    for name in ("scaled96", "scaled144", "scaled224"):
        for seed in config["seeds"]:
            directory = root / f"{name}-{seed}"
            initial = read(directory / "training-0.json")["common"]
            final_bank = read(directory / "training-512.json")["arm_bank"]
            control_bank = read(root / f"anchored-{seed}/training-512.json")["arm_bank"]
            final = read(directory / "training-512.json")["common"]
            control = read(root / f"anchored-{seed}/training-512.json")["common"]
            updates = read(directory / "updates.json")
            clipped = float(np.mean([r["preclip_gradient_norm"] > 5 for r in updates]))
            benchmark = read(root / "exports" / f"{name}-{seed}" / "report.json")
            checks = {
                "main_improved": final["weighted"]["total"] < initial["weighted"]["total"] - 1e-6,
                "paired_improved": final["weighted"]["paired_restoration"]
                < initial["weighted"]["paired_restoration"] - 1e-6,
                "combined_improved": final["combined"] < initial["combined"] - 1e-6,
                "identity_not_worse_than_anchored": final["weighted"]["identity_preservation"]
                <= control["weighted"]["identity_preservation"] + 1e-6,
                "full_bank_main_improved": final_bank["weighted"]["total"]
                < initial_bank["weighted"]["total"] - 1e-6,
                "full_bank_paired_improved": final_bank["weighted"]["paired_restoration"]
                < initial_bank["weighted"]["paired_restoration"] - 1e-6,
                "full_bank_identity_not_worse": final_bank["weighted"]["identity_preservation"]
                <= control_bank["weighted"]["identity_preservation"] + 1e-6,
                "clipping_at_most_25_percent": clipped <= 0.25,
                "stereo_graph_within_hop": benchmark["cost"]["2"]["combined"]["p99_ms"] < 128 / 48,
            }
            results.append(
                {
                    "arm": name,
                    "seed": seed,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "clipped_fraction": clipped,
                    "checkpoint_sha256": sha256_file(directory / "step-512.pth"),
                    "benchmark_sha256": sha256_file(
                        root / "exports" / f"{name}-{seed}" / "report.json"
                    ),
                }
            )
    return {
        "scope": "training optimization screen, not held-out quality or listening",
        "results": results,
        "passed": all(r["passed"] for r in results),
    }


def verify_inputs(config):
    source = Path(__file__).resolve().parents[1]
    for name, expected in config["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("input drift")
    for name, expected in config["source_hashes"].items():
        if sha256_file(source / name) != expected:
            raise ValueError("source drift")


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args, root)
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
        initial_bank = root / "calibration/initial-bank.json"
        if not initial_bank.exists():
            write_json_report(evaluate(None, pool, config["weight"]), initial_bank)
        for phase in ("calibration", "full"):
            directory = root / phase
            phase_config = read(directory / "registration.json")
            progress(root, stage=phase, steps=phase_config["steps"])
            for seed in config["seeds"]:
                for arm in phase_config["arms"]:
                    train(arm, seed, directory, phase_config, pool, common)
            if phase == "calibration":
                for seed in config["seeds"]:
                    for arm in phase_config["arms"]:
                        export_and_measure(
                            directory / f"{arm}-{seed}/step-512.pth",
                            directory / "exports" / f"{arm}-{seed}",
                        )
                gate = calibration_checks(config, directory)
                registered(gate, root / "calibration-decision.json")
                verify_inputs(config)
                if not gate["passed"]:
                    progress(
                        root,
                        stage="calibration_rejected",
                        reason="train-only screen failed; full training not started",
                    )
                    write_json_report(
                        {
                            "status": "calibration_rejected",
                            "decision_sha256": sha256_file(root / "calibration-decision.json"),
                        },
                        root / "summary.json",
                    )
                    return
        del common, pool
        gc.collect()
        directory = root / "full"
        checkpoints = {
            f"{arm}-{step}-{seed}": {
                "arm": arm,
                "path": str(directory / f"{arm}-{seed}/step-{step}.pth"),
                "sha256": sha256_file(directory / f"{arm}-{seed}/step-{step}.pth"),
            }
            for arm in config["full_arms"]
            for seed in config["seeds"]
            for step in config["full_endpoints"]
        }
        registered(checkpoints, directory / "review-registration.json")
        pools = {name: Pool(config[name], clips) for name in ("development", "shifted")}
        binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
        onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
        review("initial", None, None, directory, pools, binary, onnx, config["weight"])
        for label, value in checkpoints.items():
            checkpoint = Path(value["path"])
            review(
                label, checkpoint, value["arm"], directory, pools, binary, onnx, config["weight"]
            )
            export_and_measure(checkpoint, directory / "exports" / label)
        verify_inputs(config)
        write_json_report(
            {
                "status": "complete",
                "registration_sha256": sha256_file(directory / "registration.json"),
                "review_registration_sha256": sha256_file(directory / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(directory)): sha256_file(p)
                    for p in directory.rglob("complete.json")
                },
            },
            directory / "summary.json",
        )
        analyze(
            directory,
            source=Path(__file__).resolve().parents[1],
            comparisons=[("anchored", name) for name in ("scaled96", "scaled144", "scaled224")]
            + [("scaled96", "scaled144"), ("scaled144", "scaled224")],
        )
        write_json_report(
            {
                "status": "complete",
                "full_summary_sha256": sha256_file(directory / "summary.json"),
                "comparisons_sha256": sha256_file(directory / "comparisons.json"),
            },
            root / "summary.json",
        )
        progress(root, stage="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
