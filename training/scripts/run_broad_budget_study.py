#!/usr/bin/env python3
"""P26: full-budget data coverage x generation-budget-scope comparison."""

import argparse
import fcntl
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import (
    BoundedEnvelopeClips,
    LazyPool,
    assert_isolated,
    make_broad_schedule,
    protected_intervals,
    repair_finite_bank,
)
from data.causal_study import digest, load_sources
from data.weak_band_study import audit_groups
from evaluation.head_export import export_and_measure
from evaluation.head_study import review
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_head_study import run as analyze
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.bench_context_deployment import run as bench_native
from scripts.run_budget_study import ENDPOINTS, STEPS, freeze_endpoints
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_head_study import progress, train

ARMS = ["finite_global", "broad_global", "finite_residual", "broad_residual"]
COMPARISONS = [
    ("finite_global", "broad_global"),
    ("finite_residual", "broad_residual"),
    ("finite_global", "finite_residual"),
    ("broad_global", "broad_residual"),
]


def register(parent_path, root, native_binary):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        verify_inputs(config)
        return config
    parent = read(parent_path / "registration.json")
    if read(parent_path / "summary.json")["status"] != "complete":
        raise ValueError("completed parent review required")
    inputs = dict(parent["input_hashes"])
    for path in (parent_path / "registration.json", parent_path / "summary.json", native_binary):
        inputs[str(path.resolve())] = sha256_file(path)
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    split_audit = audit_groups(rows)
    probes = parent["development"] + parent["shifted"]
    protected = protected_intervals(probes, rows)
    bank, repairs = repair_finite_bank(parent["bank"], rows, protected)
    mapping = {digest(old): new for old, new in zip(parent["bank"], bank, strict=True)}
    common = [mapping[digest(spec)] for spec in parent["common"]]
    assert_isolated(common, rows, protected)
    config = {
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
        "format": "soundex-broad-budget-v1",
        "bank": bank,
        "common": common,
        "scope": "coverage x budget scope; reused development, no independent test or promotion",
        "arms": ARMS,
        "paced_arms": ARMS,
        "steps": STEPS,
        "endpoints": ENDPOINTS,
        "primary_endpoint": STEPS,
        "native_count": (1 + len(ARMS) * len(parent["seeds"]) * len(ENDPOINTS))
        * len(parent["development"])
        * 2,
        "joint_native_count": (2 + len(ARMS) * len(parent["seeds"]) * len(ENDPOINTS))
        * len(parent["development"])
        * 2,
        "score_interval": 2048,
        "score_entire_bank": False,
        "smoke": False,
        "schedule_groups": {arm: arm.split("_")[0] for arm in ARMS},
        "model_configs": {
            arm: {
                "arm": "complex",
                "width": 640,
                "residual_slope": 1 / 64,
                "generation_budget": True,
                "budget_scope": arm.split("_")[1],
            }
            for arm in ARMS
        },
        "budget_weights": dict.fromkeys(ARMS, 0.1),
        "selection": "8 fresh trajectories, each 16192 updates; freeze all 24 endpoints before reused review; final endpoint primary, no early selection",
        "optimization": "Adam 1e-4, no decay, norm clip 5; historical main + fixed complex auxiliary + 0.1 budget auxiliary, identical in all arms",
        "sampling": "4 distinct music recordings + 1 speech per update; equal paired/identity risk; zero gain augmentation; finite preserves P24 index draws with geometry-only crop repairs; broad cycles recording permutations, manifest-weighted rows per recording, uniform legal crop starts",
        "coverage_factor": "bundles source-time, codec, channel view and speech-recording coverage; cannot attribute effect to a single subfactor",
        "isolation": "manifest recording/source/work-group split checks; whole input windows excluding all development/shifted source-time intervals across codec, view and rate",
        "training_scores": "fixed common 320-crop training bank only, same all arms; not full-bank risk or validation",
        "cache": "128 prepared clips LRU, existing disk cache read only; PCM checksum on first read; strict finite aligned mono geometry",
        "stop_policy": "nonfinite, integrity drift or runtime error only; budget completion does not prove convergence",
        "evaluation": "unchanged relative and absolute PCM, HF/low/transient/identity metrics, observed Rust synthesis, all final native Rust-context ORT DSP; listening remains required",
        "schedule_hashes": {},
        "schedule_files": {},
    }
    report = {"split_groups": split_audit, "finite_repairs": repairs, "finite": {}, "broad": {}}
    for corpus in ("music_library", "speech_library"):
        report["finite"][corpus] = coverage(
            [s for s in config["bank"] if rows[s["row_id"]]["corpus"] == corpus], rows
        )
    for seed in config["seeds"]:
        original = parent_path / f"draws-{seed}.json"
        finite = read(original)["draws"]
        if len(finite) != STEPS or digest(finite) != parent["schedule_hashes"][str(seed)]:
            raise ValueError("parent finite schedule drift")
        inputs[str(original.resolve())] = sha256_file(original)
        broad, draws, eligible = make_broad_schedule(rows, probes, STEPS, seed)
        report["broad"][str(seed)] = {"eligible_recordings": eligible, "corpora": {}}
        for corpus in ("music_library", "speech_library"):
            report["broad"][str(seed)]["corpora"][corpus] = coverage(
                [s for s in broad if rows[s["row_id"]]["corpus"] == corpus], rows
            )
        for name, specs, schedule in (("finite", config["bank"], finite), ("broad", broad, draws)):
            key = f"{name}-{seed}"
            path = root / f"draws-{key}.json"
            registered({"specs": specs, "draws": schedule}, path)
            inputs[str(path)] = sha256_file(path)
            config["schedule_hashes"][key] = digest(schedule)
            config["schedule_files"][key] = str(path)
    registered(report, root / "coverage.json")
    inputs[str(root / "coverage.json")] = sha256_file(root / "coverage.json")
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
    registered(config, destination)
    return config


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args.parent.resolve(), root, args.native_binary.resolve())
        if args.prepare_only:
            return
        inputs = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
        detector = next(p for p in inputs if p.name == "soundex-gain-data")
        parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
        clips = BoundedEnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
        common = LazyPool(config["common"], clips)
        for seed in config["seeds"]:
            for arm in config["arms"]:
                key = f"{config['schedule_groups'][arm]}-{seed}"
                pool = LazyPool(read(Path(config["schedule_files"][key]))["specs"], clips)
                progress(
                    root, stage="training_start", arm=arm, seed=seed, positions=len(pool.specs)
                )
                train(arm, seed, root, config, pool, common)
        verify_inputs(config)
        checkpoints = freeze_endpoints(root, config)
        pools = {name: LazyPool(config[name], clips) for name in ("development", "shifted")}
        binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
        observer = next(p for p in inputs if p.name == "soundex-analysis-trace")
        onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
        progress(root, stage="frozen_review", endpoints=len(checkpoints))
        for label, value in {"initial": None, **checkpoints}.items():
            checkpoint = Path(value["path"]) if value else None
            review(
                label,
                checkpoint,
                value["arm"] if value else None,
                root,
                pools,
                binary,
                onnx,
                config["weight"],
                observed_binary=observer,
            )
            joint_review(label, checkpoint, root, pools, binary, onnx, observed_binary=observer)
            if checkpoint:
                progress(root, stage="export_and_measure", case=label)
                export_and_measure(
                    checkpoint,
                    root / "exports" / label,
                    experimental_unbounded=True,
                    iterations=10000,
                )
        verify_inputs(config)
        write_json_report(
            {
                "status": "complete",
                "scope": "training and frozen endpoint review; see native-audit separately",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )
        analyze(root, comparisons=COMPARISONS, source=Path(__file__).resolve().parents[1])
        progress(root, stage="native_final_review")
        audit_native(
            SimpleNamespace(study=root, output=root / "native-audit", binary=args.native_binary)
        )
        progress(root, stage="native_compute_benchmark")
        bench_native(
            SimpleNamespace(study=root, output=root / "native-cost", binary=args.native_binary)
        )
        progress(
            root,
            stage="complete",
            native_roles=len(config["arms"])
            * len(config["seeds"])
            * len(config["development"])
            * 2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-binary", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        write_json_report(
            {"stage": "failed", "error": traceback.format_exc()},
            args.output / "pipeline-error.json",
        )
        raise
