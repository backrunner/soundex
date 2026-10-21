#!/usr/bin/env python3
"""P35: matched broad-coverage risk compression, one small architecture, two seeds."""

import argparse
import fcntl
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from data.weak_band_study import audit_groups
from evaluation.head_export import export_and_measure
from evaluation.joint_review import joint_review
from evaluation.multires_review import review as spectrum_review
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import RestorationHead
from models.risk_balance import reconstruction_risk
from scripts.analyze_broad_risk_study import run as assess
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.bench_context_deployment import run as bench_native
from scripts.run_broad_spectrum_study import ScheduledBatches, review_inventory, validate_schedule
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_direct_spectrum_study import MODEL, train
from scripts.run_head_study import progress
from scripts.run_music_coverage_study import tensor_digest

ARMS = ("control", "compressed_identity_mid")
SEEDS = (20261012, 20261013)
STEPS = 16192


def training_terms(prediction, budget, objective, weight):
    """Exactly the P34 control/mid objectives; do not build unused guard graphs."""
    del weight
    batch = objective.batch
    direct = direct_spectrum_loss(prediction, batch)
    multires = multires_spectrum_loss(prediction, batch)
    args = (direct["per_example"], multires["per_example"], batch["identity"])
    auxiliary = 0.1 * budget_risk(budget, batch)
    return {
        "control": reconstruction_risk(*args) + auxiliary,
        "compressed_identity_mid": reconstruction_risk(*args, compress=True, identity_weight=0.3)
        + auxiliary,
        "raw": direct["total"],
        "multires": multires["total"],
        "budget": auxiliary,
    }


def protocol(parent, *, preflight=False):
    steps = 4 if preflight else STEPS
    return {
        "format": "soundex-broad-risk-v1",
        "scope": "broad training, recording-isolated REUSED validation; no independent test or promotion",
        "review_scope": "all frozen final endpoints on reused validation; common MR v1 primary, raw/native separately; PCM auxiliary",
        "arms": list(ARMS),
        "seeds": list(SEEDS[:1] if preflight else SEEDS),
        "steps": steps,
        "primary_endpoint": steps,
        "smoke": preflight,
        "model_configs": {arm: dict(MODEL) for arm in ARMS},
        "parameters": dict.fromkeys(ARMS, 431015),
        "loss_routes": {arm: {arm: 1.0} for arm in ARMS},
        "direct_loss": parent["direct_loss"],
        "multires_loss": parent["multires_loss"],
        "weight": parent["weight"],
        "rate": 1e-4,
        "specs": parent["development"],
        "development": parent["development"],
        "shifted": [],
        "paced_arms": list(ARMS),
        "optimization": "fresh shared initialization within seed; Adam 1e-4, no decay, clip 5; 80 no-grad warmup frames; identical broad draws, 4 music + 1 speech with paired/identity roles",
        "contrast": "control = uncompressed direct+MR plus unchanged .1 budget risk (P33 mr_add); candidate = per-example log1p(direct)+log1p(MR), paired coefficient .5, identity coefficient .15, unchanged .1 budget risk. Joint compression/reweighting contrast, not individual attribution. No overshoot guard.",
        "selection": "freeze all four final endpoints before validation; no intermediate validation or best-checkpoint selection. Report every seed, corpus, role and recording. Continue candidate research only if native paired music MR improves versus matched control in BOTH seeds, with lower median recording error, no corpus-level identity or dynamics regression. Otherwise diagnose; never automatic production promotion.",
        "stop_policy": "nonfinite, integrity or runtime failure only; fixed budget is not proof of convergence; no unequal extension or early performance stopping",
        "sampling": "same full P33 schedules; 20% speech experimental ratio; no new sources or gain augmentation; coverage counts actual source-time supervision excluding warmup/overlap",
        "evaluation_limits": "historically reused validation, not independent test; views do not certify simultaneous stereo fidelity; no blind listening or physical-device certification",
    }


def register(args, root):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["smoke"] != args.preflight or config["format"] != "soundex-broad-risk-v1":
            raise ValueError("protocol drift")
        verify_inputs(config)
        return config
    parent_root = args.parent.resolve()
    parent = read(parent_root / "registration.json")
    if parent["format"] != "soundex-broad-spectrum-v1" or parent["steps"] != STEPS:
        raise ValueError("full P33 parent required")
    if read(parent_root / "progress.json")["stage"] != "complete":
        raise ValueError("completed parent required")
    for name, expected in parent["source_hashes"].items():
        if sha256_file(parent_root.parent / "source/training" / name) != expected:
            raise ValueError("parent frozen source drift")
    config = protocol(parent, preflight=args.preflight)
    required = {
        "manifest.jsonl",
        "soundex-gain-data",
        "soundex-prediction-replay",
        "soundex-analysis-trace",
        "replay-contract.onnx",
    }
    inputs = {p: h for p, h in parent["input_hashes"].items() if Path(p).name in required}
    for path in (
        parent_root / "registration.json",
        parent_root / "progress.json",
        args.native_binary.resolve(),
    ):
        inputs[str(path)] = sha256_file(path)
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    if args.preflight:
        config["development"] = config["specs"] = [
            next(
                s
                for s in parent["development"]
                if rows[s["row_id"]]["corpus"] == corpus
                and rows[s["row_id"]]["sample_rate"] == rate
            )
            for corpus in ("music_library", "speech_library")
            for rate in (44100, 48000)
        ]
    config["cache_path"] = str(root / "feature-cache")
    config["schedule_files"], config["schedule_hashes"] = {}, {}
    report = {"split_groups": audit_groups(rows), "training": {}, "review": {}}
    for corpus in ("music_library", "speech_library"):
        report["review"][corpus] = review_inventory(
            [s for s in config["specs"] if rows[s["row_id"]]["corpus"] == corpus], rows
        )
    for seed in config["seeds"]:
        original = Path(parent["schedule_files"][str(seed)])
        if sha256_file(original) != parent["input_hashes"][str(original)]:
            raise ValueError("parent schedule drift")
        schedule = read(original)
        if digest(schedule["draws"]) != parent["schedule_hashes"][str(seed)]:
            raise ValueError("parent draw sequence drift")
        if args.preflight:
            draws = schedule["draws"][: config["steps"]]
            indices = sorted({i for draw in draws for i in draw})
            remap = {old: new for new, old in enumerate(indices)}
            schedule = {
                "specs": [schedule["specs"][i] for i in indices],
                "draws": [[remap[i] for i in draw] for draw in draws],
            }
        validate_schedule(
            schedule, rows, parent["development"] + parent["shifted"], config["steps"]
        )
        if set(range(len(schedule["specs"]))) != {i for draw in schedule["draws"] for i in draw}:
            raise ValueError("coverage would include unsampled specs")
        path = root / f"draws-{seed}.json"
        registered(schedule, path)
        inputs[str(original)] = sha256_file(original)
        inputs[str(path)] = sha256_file(path)
        config["schedule_files"][str(seed)] = str(path)
        config["schedule_hashes"][str(seed)] = digest(schedule["draws"])
        report["training"][str(seed)] = {
            corpus: coverage(
                [s for s in schedule["specs"] if rows[s["row_id"]]["corpus"] == corpus], rows
            )
            for corpus in ("music_library", "speech_library")
        }
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
    for seed in config["seeds"]:
        initials = []
        for arm in ARMS:
            torch.manual_seed(seed)
            model = RestorationHead(**config["model_configs"][arm])
            if sum(p.numel() for p in model.parameters()) != config["parameters"][arm]:
                raise ValueError("architecture drift")
            initials.append(tensor_digest(model))
        if len(set(initials)) != 1:
            raise ValueError("initialization mismatch")
    config["input_hashes"] = inputs
    verify_inputs(config)
    for name in config["source_hashes"]:
        target = root.parent / "source/training" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    registered(config, destination)
    return config


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args, root)
        if args.prepare_only:
            return
        paths = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            Path(config["cache_path"]),
        )
        for seed in config["seeds"]:
            schedule = read(Path(config["schedule_files"][str(seed)]))
            batches = ScheduledBatches(LazyPool(schedule["specs"], clips), schedule["draws"])
            for arm in ARMS:
                progress(root, stage="training_start", arm=arm, seed=seed, end=config["steps"])
                train(arm, seed, root, config, batches, training_terms=training_terms)
        verify_inputs(config)
        endpoints = {}
        for seed in config["seeds"]:
            for arm in ARMS:
                path = root / f"{arm}-{seed}/step-{config['steps']}.pth"
                endpoints[f"{arm}-{config['steps']}-{seed}"] = {
                    "arm": arm,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
        registered(endpoints, root / "review-registration.json")
        progress(root, stage="frozen_validation_review", endpoints=len(endpoints))
        pools = {"development": LazyPool(config["development"], clips)}
        joint_review(
            "initial",
            None,
            root,
            pools,
            next(p for p in paths if p.name == "soundex-prediction-replay"),
            next(p for p in paths if p.name == "replay-contract.onnx"),
            observed_binary=next(p for p in paths if p.name == "soundex-analysis-trace"),
        )
        for label, value in endpoints.items():
            export_and_measure(Path(value["path"]), root / "exports" / label, iterations=100)
        write_json_report(
            {
                "status": "complete",
                "scope": "training complete; native/MR/benchmark stages separate",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
            },
            root / "summary.json",
        )
        progress(root, stage="full_native_validation_review")
        audit_native(
            SimpleNamespace(study=root, output=root / "native-audit", binary=args.native_binary)
        )
        progress(root, stage="multires_native_validation_review")
        spectrum_review(root, args.native_binary)
        if not args.preflight:
            progress(root, stage="native_compute_benchmark")
            bench_native(
                SimpleNamespace(study=root, output=root / "native-cost", binary=args.native_binary)
            )
        verify_inputs(config)
        assess(root)
        progress(root, stage="complete", preflight=args.preflight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "native-binary"):
        parser.add_argument("--" + name, type=Path, required=True)
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
