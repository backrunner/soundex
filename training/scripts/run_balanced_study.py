"""P4: matched smooth-DSP continuation with peak and bidirectional-band ablations.

Requires completed P3; starts from each seed's control checkpoint, not a winner
chosen on held-out audio. All arms use identical 12.5% identity-mixed draws.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback
from functools import partial
from pathlib import Path

import torch

from data.causal_study import digest, load_sources, make_schedule
from data.weak_band_study import (
    IdentityClips,
    add_identity_controls,
    audit_groups,
    identity_specs,
    review_specs,
    training_probe_specs,
)
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from models.balanced_objective import ARMS, LIMITER_MODE, VERSION, objective
from models.causal_objective import WEIGHTS
from models.weak_band_objective import (
    ABSOLUTE_POWER_FLOOR,
    BAND_EDGES_HZ,
    EXCESS_WEIGHT,
    RELATIVE_POWER_FLOOR,
)
from scripts.run_causal_study import actual_score, run_arm
from scripts.run_weak_band_study import common_validation, freeze_binary


def main(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = args.output.resolve()
    parent_root = args.parent.resolve()
    source = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in sorted(source.rglob("*.py"))
        if not any(
            s in {".venv", "tests", "outputs", "__pycache__"} for s in p.relative_to(source).parts
        )
    }
    parent_config = json.loads((parent_root / "config.json").read_text())
    summary = json.loads((parent_root / "summary.json").read_text())
    if parent_config["format"] != "soundex-p3-study-v1":
        raise ValueError("requires a registered P3 parent")
    if summary["status"] != "complete":
        raise ValueError("P3 must complete before P4 starts")
    previous = (
        json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    )
    stream_binary = freeze_binary(
        args.stream_binary,
        root / "binaries/soundex-stream-eval",
        previous.get("stream_binary_sha256") or sha256_file(args.stream_binary),
    )
    detector = freeze_binary(
        args.detector, root / "binaries/soundex-gain-data", parent_config["detector_sha256"]
    )
    rows, manifests = load_sources([Path(p) for p in parent_config["manifests"]])
    if manifests != parent_config["manifests"]:
        raise ValueError("parent data manifests changed")
    coverage = audit_groups(rows)
    heldout = review_specs(rows)
    train_specs = training_probe_specs(rows)
    if args.smoke:
        heldout = heldout[:1]
        train_specs = [
            s
            for corpus in ("music_library", "speech_library")
            for s in [v for v in train_specs if rows[v["row_id"]]["corpus"] == corpus][:2]
        ]
    specs = parent_config["validation_specs"]
    if args.smoke:
        specs = [
            s
            for corpus in ("music_library", "speech_library")
            for s in [v for v in specs if rows[v["row_id"]]["corpus"] == corpus][:2]
        ]
    parents = {}
    for seed in args.seeds:
        path = parent_root / f"control-{seed}" / "best.pth"
        receipt = json.loads((path.parent / "complete.json").read_text())
        if receipt["best_checkpoint_sha256"] != sha256_file(path):
            raise ValueError("parent receipt mismatch")
        parents[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
    config = {
        "format": "soundex-p4-study-v1",
        "training_source_sha256": digest(source_hashes),
        "parent_config_sha256": sha256_file(parent_root / "config.json"),
        "parents": parents,
        "steps": args.steps,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "validation_interval": args.validation_interval,
        "seeds": args.seeds,
        "arms": list(ARMS),
        "evaluation_limiter_mode": LIMITER_MODE,
        "base_weights": WEIGHTS,
        "parent_arm": "control",
        "device": args.device,
        "torch_version": str(torch.__version__),
        "python_version": sys.version,
        "manifests": manifests,
        "validation_specs": specs,
        "smoke": args.smoke,
        "selection_objective": VERSION,
        "identity_selection_weight": 0.125,
        "excess_weight": EXCESS_WEIGHT,
        "band_edges_hz": list(BAND_EDGES_HZ),
        "relative_power_floor": RELATIVE_POWER_FLOOR,
        "absolute_power_floor": ABSOLUTE_POWER_FLOOR,
        "stream_binary": str(stream_binary),
        "stream_binary_sha256": sha256_file(stream_binary),
        "detector_sha256": sha256_file(detector),
        "coverage": coverage,
        "training_probe_specs": train_specs,
        "heldout_review_specs": heldout,
        "heldout_used_for_selection": False,
        "independent_test": False,
    }
    if (root / "config.json").exists() and json.loads((root / "config.json").read_text()) != config:
        raise ValueError("registered study changed")
    write_json_report(config, root / "config.json")
    write_json_report(source_hashes, root / "source-files.json")
    provenance = {
        "data_recipe_sha256": next(iter(rows.values()))["recipe_hash"],
        "manifest_set_sha256": digest(manifests),
    }
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if (root / "source-git.json").exists():
        git_sha = json.loads((root / "source-git.json").read_text())["sha256"]
    else:
        write_json_report({"sha256": git_sha}, root / "source-git.json")
    provenance["source_git_sha"] = git_sha
    clips = IdentityClips(rows, detector, root / "cache")
    validator = partial(
        common_validation, train_specs=train_specs, objective_fn=objective, version=VERSION
    )
    completed = []
    for seed in args.seeds:
        schedule, sampling = make_schedule(rows, seed + 20000, args.steps * 4)
        write_json_report(
            {"draws": schedule, "sampling": sampling, "sha256": digest(schedule)},
            root / f"draws-{seed}.json",
        )
        parent_receipt = parents[str(seed)]
        for arm in config["arms"]:
            arm_root = root / f"{arm}-{seed}"
            arm_root.mkdir(exist_ok=True)
            draws = add_identity_controls(schedule, 4)
            arm_config = {
                **config,
                "arm": arm,
                "parent_checkpoint_sha256": parent_receipt["sha256"],
                "objective_version": VERSION + ":" + arm,
            }
            arm_provenance = {**provenance, "schedule_sha256": digest(draws)}
            # Parent is scored before ANY update with the common objective.
            baseline = arm_root / "parent-validation.json"
            if not baseline.exists():
                from export_study import make_model

                model = make_model("tcn").to(args.device)
                model.load_state_dict(
                    torch.load(parent_receipt["path"], map_location="cpu", weights_only=True)[
                        "model"
                    ]
                )
                write_json_report(validator(model, "tcn", clips, specs, args.device), baseline)
                del model
            result = run_arm(
                "tcn",
                seed,
                arm_config,
                arm_provenance,
                draws,
                clips,
                specs,
                arm_root,
                args.device,
                initial_checkpoint=Path(parent_receipt["path"]),
                objective_fn=partial(objective, arm=arm),
                validation_fn=validator,
            )
            identity_report = arm_root / "rust-identity.json"
            if not identity_report.exists():
                write_json_report(
                    actual_score(
                        arm_root / "best.onnx",
                        clips,
                        identity_specs(specs),
                        config["stream_binary"],
                        limiter_mode=LIMITER_MODE,
                    ),
                    identity_report,
                )
            heldout_path = arm_root / "rust-heldout-review.json"
            if not heldout_path.exists():
                report = actual_score(
                    arm_root / "best.onnx",
                    clips,
                    heldout,
                    config["stream_binary"],
                    sample_count=None,
                    limiter_mode=LIMITER_MODE,
                )
                report["used_for_selection"] = False
                report["scope"] = (
                    "full paired files, one per test recording; reused held-out review"
                )
                write_json_report(report, heldout_path)
            # Add genre/region strata without changing the fixed row selection.
            for name in ("rust-validation.json", "rust-identity.json", "rust-heldout-review.json"):
                report = json.loads((arm_root / name).read_text())
                for row in report["rows"]:
                    meta = clips.rows[row["row_id"]].get("source_metadata", {})
                    row["genre"] = meta.get("genre", "unlabeled")
                    row["region"] = meta.get("music_region", "unlabeled")
                report["aggregates"] = aggregate_metric_rows(
                    report["rows"],
                    samples=2000,
                    confidence=0.95,
                    seed=20261012,
                    cluster_field="track_id",
                    dimensions=(
                        "corpus",
                        "codec_id",
                        "sample_rate",
                        "channel_role",
                        "genre",
                        "region",
                    ),
                )
                write_json_report(report, arm_root / name)
            completed.append({**result, "arm": arm})
            write_json_report(
                {"status": "running", "completed": completed, "promoted": False},
                root / "summary.json",
            )
    write_json_report(
        {"status": "complete", "completed": completed, "promoted": False}, root / "summary.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261012, 20261013])
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.validation_interval) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid steps, interval or duplicate seed")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("study already running") from None
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    failed = args.output / "failure.json"
    if failed.exists():
        failed.rename(args.output / f"failure-{failed.stat().st_mtime_ns}.json")
    try:
        main(args)
    except Exception:
        write_json_report({"error": traceback.format_exc(), "pid": os.getpid()}, failed)
        raise
