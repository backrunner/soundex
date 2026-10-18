#!/usr/bin/env python3
"""P28 small-width single-intervention diagnosis; no scaling or automatic promotion."""

import argparse
import fcntl
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, assert_isolated, protected_intervals
from data.causal_study import digest, load_sources
from data.weak_band_study import audit_groups
from evaluation.head_export import export_and_measure
from evaluation.head_study import review
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import RestorationHead, load_restoration_head
from scripts.analyze_head_study import run as analyze
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.audit_head_collapse import gradients, inspect
from scripts.bench_context_deployment import run as bench_native
from scripts.run_budget_study import freeze_endpoints
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_head_study import progress, train

ARMS = ["control", "decoder_norm", "budget_detach", "phase_input"]
STEPS = 2048


def protocol(parent, *, preflight=False):
    base = dict(
        arm="complex",
        width=96,
        residual_slope=1 / 64,
        generation_budget=True,
        budget_scope="residual",
    )
    steps, seeds = (4, parent["seeds"][:1]) if preflight else (STEPS, parent["seeds"])
    return {
        **{
            k: parent[k]
            for k in ("common", "weight", "rate", "development", "shifted", "tolerance")
        },
        "format": "soundex-small-mechanisms-v1",
        "arms": ARMS,
        "seeds": seeds,
        "steps": steps,
        "endpoints": [steps],
        "primary_endpoint": steps,
        "score_interval": steps,
        "score_entire_bank": False,
        "smoke": preflight,
        "scope": "discarded train-only integration"
        if preflight
        else "small-model optimization diagnosis; reused development review; not full convergence, capacity ranking or independent test",
        "model_configs": {
            arm: {
                **base,
                "decoder_normalization": arm == "decoder_norm",
                "phase_input": arm == "phase_input",
            }
            for arm in ARMS
        },
        "budget_weights": dict.fromkeys(ARMS, 0.1),
        "detached_budget_arms": ["budget_detach"],
        "paced_arms": ARMS,
        "schedule_groups": dict.fromkeys(ARMS, "broad"),
        "schedule_hashes": {},
        "schedule_files": {},
        "selection": "all fixed final endpoints, both seeds, no early selection or automatic scaling; shared initialization and draws; never promote from this diagnostic",
        "optimization": "Adam 1e-4, no decay, norm clip 5; unchanged historical main + fixed clean-complex consistency auxiliary + 0.1 budget loss; equal paired/identity risk",
        "factors": "decoder_norm adds LayerNorm before decoder GELU; budget_detach blocks ONLY budget auxiliary gradients into latent features; phase_input adds zero-initialized projection of observed current complex carrier before encoder LayerNorm",
        "interpretation": "four one-factor interventions, not a factorial interaction study; phase arm adds 258*96 parameters and cannot isolate information from added capacity",
        "sampling": f"same first {steps} broad P26 draws per seed; 4 distinct music recordings and 1 speech per step; original channel views and whole-window source isolation; zero gain augmentation",
        "stop_policy": "nonfinite, integrity drift or runtime error only; never truncate a comparison using intermediate loss",
    }


def register(parent_path, root, native_binary, *, preflight=False):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["smoke"] != preflight:
            raise ValueError("preflight/full protocol drift")
        verify_inputs(config)
        return config
    parent = read(parent_path / "registration.json")
    if read(parent_path / "summary.json")["status"] != "complete":
        raise ValueError("completed P26 parent required")
    inputs = dict(parent["input_hashes"])
    for path in (parent_path / "registration.json", parent_path / "summary.json", native_binary):
        inputs[str(path.resolve())] = sha256_file(path)
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    config = protocol(parent, preflight=preflight)
    protected = protected_intervals(parent["development"] + parent["shifted"], rows)
    assert_isolated(config["common"], rows, protected)
    if preflight:
        music = [s for s in config["common"] if rows[s["row_id"]]["corpus"] == "music_library"]
        speech = [s for s in config["common"] if rows[s["row_id"]]["corpus"] == "speech_library"]
        config.update(
            common=music[:4] + speech[:1],
            development=music[:1] + speech[:1],
            shifted=music[1:2] + speech[1:2],
        )
    config["native_count"] = (1 + len(ARMS) * len(config["seeds"])) * len(config["development"]) * 2
    config["joint_native_count"] = (
        (2 + len(ARMS) * len(config["seeds"])) * len(config["development"]) * 2
    )
    config["cache_path"] = str(root / "feature-cache")
    report = {"split_groups": audit_groups(rows), "schedules": {}}
    for seed in config["seeds"]:
        key = f"broad-{seed}"
        path = Path(parent["schedule_files"][key])
        schedule = read(path)
        if digest(schedule["draws"]) != parent["schedule_hashes"][key]:
            raise ValueError("parent schedule drift")
        draws = schedule["draws"][: config["steps"]]
        count = max(i for draw in draws for i in draw) + 1
        specs = schedule["specs"][:count]
        assert_isolated(specs, rows, protected)
        dest = root / f"draws-{key}.json"
        registered({"specs": specs, "draws": draws}, dest)
        config["schedule_files"][key], config["schedule_hashes"][key] = str(dest), digest(draws)
        inputs[str(dest)], inputs[str(path)] = sha256_file(dest), sha256_file(path)
        report["schedules"][key] = {
            name: coverage([s for s in specs if rows[s["row_id"]]["corpus"] == name], rows)
            for name in ("music_library", "speech_library")
        }
    registered(report, root / "coverage.json")
    inputs[str(root / "coverage.json")] = sha256_file(root / "coverage.json")
    config["parameters"] = {
        arm: sum(p.numel() for p in RestorationHead(**value).parameters())
        for arm, value in config["model_configs"].items()
    }
    if max(config["parameters"].values()) > 500000:
        raise ValueError("this diagnostic is limited to small models")
    config["mechanism_draws"] = sorted(
        {0, (config["steps"] - 1) // 3, 2 * (config["steps"] - 1) // 3, config["steps"] - 1}
    )
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


def mechanism_audit(root, config, clips, checkpoints):
    schedule = read(Path(config["schedule_files"][f"broad-{config['seeds'][0]}"]))
    pool = LazyPool(schedule["specs"], clips)
    batches = [pool.select(schedule["draws"][d]) for d in config["mechanism_draws"]]
    output = root / "mechanism"
    output.mkdir(exist_ok=True)
    for label, value in checkpoints.items():
        path = output / f"{label}.json"
        if path.exists():
            if read(path)["checkpoint_sha256"] != value["sha256"]:
                raise ValueError("mechanism checkpoint drift")
            continue
        model = load_restoration_head(
            torch.load(value["path"], map_location="cpu", weights_only=False)
        ).eval()
        write_json_report(
            {
                "scope": "fixed training-only crops; no generalization claim",
                "checkpoint_sha256": value["sha256"],
                "draw_indices": config["mechanism_draws"],
                "batches": [inspect(model, *b) for b in batches],
                "gradients": gradients(
                    model,
                    *batches[0],
                    config["weight"],
                    detach_auxiliary=value["arm"] in config["detached_budget_arms"],
                ),
            },
            path,
        )


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(
            args.parent.resolve(), root, args.native_binary.resolve(), preflight=args.preflight
        )
        if args.prepare_only:
            return
        paths = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        detector = next(p for p in paths if p.name == "soundex-gain-data")
        clips = BoundedEnvelopeClips(rows, detector, Path(config["cache_path"]))
        common = LazyPool(config["common"], clips)
        for seed in config["seeds"]:
            pool = LazyPool(read(Path(config["schedule_files"][f"broad-{seed}"]))["specs"], clips)
            for arm in ARMS:
                progress(root, stage="training_start", arm=arm, seed=seed)
                train(arm, seed, root, config, pool, common)
        verify_inputs(config)
        checkpoints = freeze_endpoints(root, config)
        mechanism_audit(root, config, clips, checkpoints)
        pools = {name: LazyPool(config[name], clips) for name in ("development", "shifted")}
        binary = next(p for p in paths if p.name == "soundex-prediction-replay")
        observer = next(p for p in paths if p.name == "soundex-analysis-trace")
        onnx = next(p for p in paths if p.name == "replay-contract.onnx")
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
                progress(root, stage="export", case=label)
                export_and_measure(
                    checkpoint,
                    root / "exports" / label,
                    iterations=100 if args.preflight else 10000,
                )
        verify_inputs(config)
        registered(
            {
                "status": "complete",
                "scope": "training and replay only; full frontend and timing separately",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )
        analyze(
            root,
            comparisons=[("control", arm) for arm in ARMS[1:]],
            source=Path(__file__).resolve().parents[1],
        )
        progress(root, stage="native_final_review")
        audit_native(
            SimpleNamespace(study=root, output=root / "native-audit", binary=args.native_binary)
        )
        if not args.preflight:
            progress(root, stage="native_compute_benchmark")
            bench_native(
                SimpleNamespace(study=root, output=root / "native-cost", binary=args.native_binary)
            )
        progress(root, stage="complete", preflight=args.preflight)


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
