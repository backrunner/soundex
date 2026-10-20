#!/usr/bin/env python3
"""P33 matched broad-coverage control versus added multi-resolution supervision."""

import argparse
import fcntl
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, assert_isolated, protected_intervals
from data.causal_study import digest, load_sources
from data.weak_band_study import audit_groups
from evaluation.head_export import export_and_measure
from evaluation.head_study import review
from evaluation.joint_review import joint_review
from evaluation.multires_review import review as spectrum_review
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import RestorationHead
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.bench_context_deployment import run as bench_native
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_direct_spectrum_study import MODEL, train
from scripts.run_head_study import progress
from scripts.run_music_coverage_study import tensor_digest
from scripts.run_spectrum_priority_study import training_terms

ARMS = ("control", "mr_add")
STEPS = 16192


class ScheduledBatches:
    """Materialize exactly the registered update, including after resume."""

    def __init__(self, pool, draws):
        self.pool, self.draws = pool, draws

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        return self.pool.select(self.draws[index])


def protocol(parent, broad, *, preflight=False):
    steps = 4 if preflight else STEPS
    development = broad["development"]
    return {
        "format": "soundex-broad-spectrum-v1",
        "scope": "broad training, recording-isolated REUSED validation review; no independent test or promotion",
        "review_scope": "all frozen final endpoints on recording-isolated reused validation; spectral primary, PCM auxiliary; no independent-test claim",
        "arms": list(ARMS),
        "seeds": parent["seeds"][:1] if preflight else parent["seeds"],
        "steps": steps,
        "primary_endpoint": steps,
        "smoke": preflight,
        "model_configs": {arm: dict(MODEL) for arm in ARMS},
        "parameters": dict.fromkeys(ARMS, 431015),
        "loss_routes": {arm: parent["loss_routes"][arm] for arm in ARMS},
        "direct_loss": parent["direct_loss"],
        "multires_loss": parent["multires_loss"],
        "weight": parent["weight"],
        "rate": parent["rate"],
        # These are REVIEW specs, never the training schedule.
        "specs": development,
        "development": development,
        "shifted": [],
        "paced_arms": list(ARMS),
        "optimization": "fresh identical initialization per seed; Adam 1e-4, no decay, clip 5; same broad draws per seed in both arms, 4 distinct music + 1 speech, equal paired/identity, 80 no-grad warmup frames",
        "selection": "all four final endpoints frozen before any validation score; no early selection, unequal extension or automatic promotion; common MR v1 by recording/corpus/role and all components, auxiliary PCM/identity/continuity",
        "stop_policy": "nonfinite, integrity or runtime failure only; completing budget is not convergence",
        "sampling": "reuse full P26 broad schedule, recording-balanced cycles and manifest-weighted codec/view rows; 20% speech experimental ratio; no new source or gain augmentation",
        "evaluation_limits": "reused validation, no fresh test; mono channel views do not prove simultaneous stereo fidelity; no blind listening or device certification",
    }


def validate_schedule(schedule, rows, probes, steps):
    if len(schedule["draws"]) != steps:
        raise ValueError("schedule length does not match fixed budget")
    assert_isolated(schedule["specs"], rows, protected_intervals(probes, rows))
    for draw in schedule["draws"]:
        if (
            len(draw) != 5
            or len(set(draw)) != 5
            or any(type(i) is not int or not 0 <= i < len(schedule["specs"]) for i in draw)
        ):
            raise ValueError("invalid paired sampling count")
        selected = [rows[schedule["specs"][i]["row_id"]] for i in draw]
        if [r["corpus"] for r in selected] != ["music_library"] * 4 + ["speech_library"]:
            raise ValueError("corpus ratio drift")
        if len({r["track_id"] for r in selected[:4]}) != 4:
            raise ValueError("music recordings must differ within a batch")


def review_inventory(specs, rows):
    selected = [rows[s["row_id"]] for s in specs]
    if not selected or any(r["split"] != "validation" for r in selected):
        raise ValueError("validation inventory requires validation-only rows")
    tracks = {r["track_id"]: r for r in selected}
    return {
        "clips": len(selected),
        "recordings": len(tracks),
        "scope": "recording-isolated reused validation; not independent test",
        "codecs": dict(Counter(r["codec_id"] for r in selected)),
        "views": dict(Counter(r["channel_role"] for r in selected)),
        "rates": dict(Counter(str(r["sample_rate"]) for r in selected)),
        "recording_genres": dict(
            Counter(r.get("source_metadata", {}).get("genre") or "unknown" for r in tracks.values())
        ),
        "recording_regions": dict(
            Counter(
                r.get("source_metadata", {}).get("music_region") or "unknown"
                for r in tracks.values()
            )
        ),
    }


def register(args, root):
    destination = root / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["smoke"] != args.preflight:
            raise ValueError("preflight/full drift")
        verify_inputs(config)
        return config
    parent_path, broad_path = args.parent.resolve(), args.coverage_parent.resolve()
    parent, broad = (read(p / "registration.json") for p in (parent_path, broad_path))
    if (
        parent["format"] != "soundex-spectrum-priority-fit-v1"
        or broad["format"] != "soundex-broad-budget-v1"
    ):
        raise ValueError("unexpected parent protocols")
    for p, config in ((parent_path, parent), (broad_path, broad)):
        if read(p / "progress.json")["stage"] != "complete":
            raise ValueError("completed parent required")
        for name, expected in config["source_hashes"].items():
            if sha256_file(p.parent / "source/training" / name) != expected:
                raise ValueError("parent frozen source drift")
    required = {
        "manifest.jsonl",
        "soundex-gain-data",
        "soundex-prediction-replay",
        "soundex-analysis-trace",
        "replay-contract.onnx",
    }
    inputs = {p: h for p, h in parent["input_hashes"].items() if Path(p).name in required}
    for p in (
        parent_path / "registration.json",
        parent_path / "progress.json",
        broad_path / "registration.json",
        broad_path / "progress.json",
        args.native_binary.resolve(),
    ):
        inputs[str(p)] = sha256_file(p)
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    group_audit = audit_groups(rows)
    config = protocol(parent, broad, preflight=args.preflight)
    if args.preflight:
        # Exercise both corpora and both declared rates without scoring full validation.
        dev = broad["development"]
        config["development"] = config["specs"] = [
            next(
                s
                for s in dev
                if rows[s["row_id"]]["corpus"] == corpus
                and rows[s["row_id"]]["sample_rate"] == rate
            )
            for corpus in ("music_library", "speech_library")
            for rate in (44100, 48000)
        ]
    if any(rows[s["row_id"]]["split"] != "validation" for s in config["development"]):
        raise ValueError("review requires validation recordings")
    config["cache_path"] = str(root / "feature-cache")
    config["schedule_files"], config["schedule_hashes"] = {}, {}
    report = {"split_groups": group_audit, "training": {}, "review": {}}
    for corpus in ("music_library", "speech_library"):
        report["review"][corpus] = review_inventory(
            [s for s in config["specs"] if rows[s["row_id"]]["corpus"] == corpus], rows
        )
    for seed in config["seeds"]:
        original = Path(broad["schedule_files"][f"broad-{seed}"])
        if sha256_file(original) != broad["input_hashes"][str(original)]:
            raise ValueError("parent draw file drift")
        schedule = read(original)
        if digest(schedule["draws"]) != broad["schedule_hashes"][f"broad-{seed}"]:
            raise ValueError("parent draw sequence drift")
        if args.preflight:
            schedule = {"specs": schedule["specs"][:20], "draws": schedule["draws"][:4]}
        validate_schedule(schedule, rows, broad["development"] + broad["shifted"], config["steps"])
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
        initial = []
        for arm in ARMS:
            torch.manual_seed(seed)
            model = RestorationHead(**config["model_configs"][arm])
            assert sum(p.numel() for p in model.parameters()) == config["parameters"][arm]
            initial.append(tensor_digest(model))
        if len(set(initial)) != 1:
            raise ValueError("unmatched initialization")
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
        binary = next(p for p in paths if p.name == "soundex-prediction-replay")
        observer = next(p for p in paths if p.name == "soundex-analysis-trace")
        onnx = next(p for p in paths if p.name == "replay-contract.onnx")
        for label, value in {"initial": None, **endpoints}.items():
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
                export_and_measure(checkpoint, root / "exports" / label, iterations=100)
        write_json_report(
            {
                "status": "complete",
                "scope": "training and frozen validation proxy review; native and MR stages separate",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
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
        progress(root, stage="complete", preflight=args.preflight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "coverage-parent", "output", "native-binary"):
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
