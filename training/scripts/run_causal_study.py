"""P2: fresh legacy/GRU/TCN training with frozen paired draws and two seeds.

No automatic promotion. Best checkpoints use the same registered proxy; actual
Rust output is reported separately. Can resume only an identical registered run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.causal_study import (
    BURN,
    FRAMES,
    SAMPLES,
    PreparedClips,
    digest,
    load_sources,
    make_schedule,
)
from evaluation.compare_extension import selected_rows
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from evaluation.runner import _metric_row
from evaluation.rust_bridge import RustStreamEvaluator
from export_study import LEGACY_CONFIG, export_study, make_model
from models.causal_objective import WEIGHTS, objective


def prediction(model, kind, batch):
    x, rate = batch["x"], batch["rate"]
    if kind == "legacy":
        # No prefix forward: burn-in must not update BatchNorm statistics.
        pred = model(x[:, :, BURN:])
    else:
        with torch.no_grad():
            _, state = model(x[:, :, :BURN], model.initial_state(len(x), device=x.device), rate)
        pred, _ = model(x[:, :, BURN:], state.detach(), rate)
    return torch.cat((x[:, :, :BURN], pred), dim=2)


def atomic_save(value, path):
    staged = path.with_suffix(".partial.pth")
    try:
        torch.save(value, staged)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def validation(model, kind, clips, specs, device, objective_fn=objective):
    model.eval()
    rows = []
    with torch.no_grad():
        for spec in specs:
            batch = clips.batch([spec], device)
            values = {
                k: float(v) for k, v in objective_fn(prediction(model, kind, batch), batch).items()
            }
            if not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError("nonfinite validation")
            rows.append(
                {
                    "row_id": spec["row_id"],
                    "corpus": clips.rows[spec["row_id"]]["corpus"],
                    "metrics": values,
                }
            )
    groups = {
        name: {
            key: float(np.mean([r["metrics"][key] for r in rows if r["corpus"] == name]))
            for key in rows[0]["metrics"]
        }
        for name in sorted({r["corpus"] for r in rows})
    }
    score = sum(
        groups[n]["total"] * w
        for n, w in {"music_library": 253 / 254, "speech_library": 1 / 254}.items()
    )
    model.train()
    return {"score": score, "corpora": groups, "rows": rows}


def actual_score(model_path, clips, specs, binary, *, sample_count=SAMPLES, limiter_mode="legacy"):
    evaluator = RustStreamEvaluator(binary)
    metrics = []
    for spec in specs:
        row = clips.rows[spec["row_id"]]
        count = sample_count if sample_count is not None else row["length_samples"] - spec["start"]
        if count <= (BURN + 8) * 128 + 512:
            raise ValueError("scoring region too short")
        waves = []
        for role in ("clean", "degraded"):
            path = Path(row["_root"]) / row[f"{role}_path"]
            if sha256_file(path) != row[f"{role}_checksum"]:
                raise ValueError("scoring PCM changed")
            wave, rate = sf.read(path, dtype="float32")
            wave = wave[spec["start"] : spec["start"] + count]
            waves.append(np.pad(wave, (0, count - len(wave))))
        clean, degraded = waves
        result = evaluator.process(
            model_path,
            degraded,
            rate,
            mode="offline",
            high_band_gain="model",
            limiter_mode=limiter_mode,
        )
        full = evaluate_signal_triplet(
            clean,
            degraded,
            result.audio[:, 0],
            sample_rate=rate,
            cutoff_hz=row["measured_cutoff_hz"],
        )
        # Same fully observed region for the legacy C1 and new 64-hop warmup models.
        start, stop = (BURN + 8) * 128, count - 256
        steady = evaluate_signal_triplet(
            clean[start:stop],
            degraded[start:stop],
            result.audio[start:stop, 0],
            sample_rate=rate,
            cutoff_hz=row["measured_cutoff_hz"],
        )
        full.update({"steady_" + k: v for k, v in steady.items()})
        full["output_peak"] = float(np.abs(result.audio).max())
        metrics.append(_metric_row(row, full))
    return {
        "model_sha256": sha256_file(model_path),
        "limiter_mode": limiter_mode,
        "independent_test": False,
        "listening_performed": False,
        "rows": metrics,
        "aggregates": aggregate_metric_rows(
            metrics, samples=1000, seed=20261012, confidence=0.95, cluster_field="track_id"
        ),
    }


def run_arm(
    kind,
    seed,
    config,
    provenance,
    schedule,
    clips,
    val_specs,
    root,
    device,
    *,
    objective_fn=objective,
    validation_fn=validation,
    initial_checkpoint=None,
):
    root.mkdir(parents=True, exist_ok=True)
    last = root / "last.pth"
    if (root / "complete.json").exists():
        receipt = json.loads((root / "complete.json").read_text())
        if (
            receipt["best_checkpoint_sha256"] != sha256_file(root / "best.pth")
            or receipt["onnx_sha256"] != sha256_file(root / "best.onnx")
            or not (root / "rust-validation.json").is_file()
            or receipt["config_sha256"] != digest(config)
        ):
            raise ValueError("completed arm receipt mismatch")
        return receipt
    torch.manual_seed(seed)
    model = make_model(kind).to(device)
    if initial_checkpoint is not None:
        if sha256_file(initial_checkpoint) != config["parent_checkpoint_sha256"]:
            raise ValueError("parent checkpoint changed")
        parent = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if parent["kind"] != kind or parent["seed"] != seed:
            raise ValueError("parent kind/seed mismatch")
        model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    step, best, seconds = 0, float("inf"), 0.0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if (
            state["config"] != config
            or state["provenance"] != provenance
            or state["kind"] != kind
            or state["seed"] != seed
        ):
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, best, seconds = state["step"], state["best_score"], state["training_seconds"]
        torch.set_rng_state(state["rng_state"])
        if device == "mps":
            torch.mps.set_rng_state(state["mps_rng_state"])
    started = time.monotonic()
    initial_step = step
    model.train()
    for step in range(initial_step + 1, config["steps"] + 1):
        at = (step - 1) * config["batch_size"]
        specs = schedule[at : at + config["batch_size"]]
        batch = clips.batch(specs, device)
        lr = config["learning_rate"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / config["steps"]))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        values = objective_fn(prediction(model, kind, batch), batch)
        if not torch.isfinite(values["total"]):
            raise RuntimeError(f"nonfinite train loss at {kind}/{seed}/{step}")
        values["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 32 == 0 or step == 1:
            record = {
                "arm": config.get("arm", kind),
                "step": step,
                "loss": float(values["total"].detach()),
                "gradient_norm": float(norm),
                "lr": lr,
                "seconds_since_resume": time.monotonic() - started,
                "updates_since_resume": step - initial_step,
            }
            with (root / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({"kind": kind, "seed": seed, **record}), flush=True)
            write_json_report({"kind": kind, "seed": seed, **record}, root.parent / "progress.json")
        if step % config["validation_interval"] == 0 or step == config["steps"]:
            report = validation_fn(model, kind, clips, val_specs, device)
            improved = report["score"] < best
            best = min(best, report["score"])
            state = {
                "format": "soundex-p2-v1",
                "kind": kind,
                "seed": seed,
                "width": getattr(model, "width", None),
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "mps_rng_state": torch.mps.get_rng_state() if device == "mps" else None,
                "config": config,
                "provenance": provenance,
                "step": step,
                "best_score": best,
                "validation": report,
                "training_seconds": seconds + time.monotonic() - started,
            }
            atomic_save(state, last)
            if improved:
                atomic_save(state, root / "best.pth")
            write_json_report(report, root / f"validation-{step:05d}.json")
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "seed": seed,
                        "step": step,
                        "validation": report["score"],
                        "best": best,
                    }
                ),
                flush=True,
            )
    best_path = root / "best.pth"
    # Export and actual CPU Rust scoring are required even after a resumed final update.
    export_study(best_path, root / "best.onnx")
    actual = actual_score(
        root / "best.onnx",
        clips,
        val_specs,
        config["stream_binary"],
        limiter_mode=config.get("evaluation_limiter_mode", "legacy"),
    )
    write_json_report(actual, root / "rust-validation.json")
    state = torch.load(best_path, map_location="cpu", weights_only=True)
    receipt = {
        "kind": kind,
        "seed": seed,
        "steps": config["steps"],
        "best_step": state["step"],
        "best_score": state["best_score"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": torch.load(last, map_location="cpu", weights_only=True)[
            "training_seconds"
        ],
        "onnx_bytes": (root / "best.onnx").stat().st_size,
        "onnx_sha256": sha256_file(root / "best.onnx"),
        "best_checkpoint_sha256": sha256_file(best_path),
        "config_sha256": digest(config),
        "promoted": False,
    }
    write_json_report(receipt, root / "complete.json")
    return receipt


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(source_root)): sha256_file(p)
        for p in sorted(source_root.rglob("*.py"))
        if not any(
            part in {".venv", "tests", "outputs", "__pycache__"}
            for part in p.relative_to(source_root).parts
        )
    }
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "failure.json").exists():
        failed = root / "failure.json"
        failed.rename(root / f"failure-{failed.stat().st_mtime_ns}.json")
    rows, hashes = load_sources(args.manifest)
    validation_rows = selected_rows(args.manifest, args.selection)
    if args.smoke:
        validation_rows = [
            r
            for n in ("music_library", "speech_library")
            for r in [v for v in validation_rows if v["corpus"] == n][:2]
        ]
    specs = [
        {
            "row_id": r["row_id"],
            "start": max(0, (r["length_samples"] - SAMPLES) // 2),
            "gain_db": 0.0,
        }
        for r in validation_rows
    ]
    config = {
        "training_source_sha256": digest(source_hashes),
        "objective_version": "causal-reconstruction-v1",
        "objective_weights": WEIGHTS,
        "legacy_model": LEGACY_CONFIG,
        "frames": FRAMES,
        "burn_in_hops": BURN,
        "sample_rates": [44100, 48000],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_interval": args.validation_interval,
        "device": args.device,
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "threads": args.threads,
        "seeds": args.seeds,
        "kinds": args.kinds,
        "high_band_gain": "model",
        "phase_policy": "model",
        "warmup_scope": "per-crop cold start, past only",
        "smoke": args.smoke,
        "manifests": hashes,
        "validation_specs": specs,
        "stream_binary": str(args.stream_binary.resolve()),
        "stream_binary_sha256": sha256_file(args.stream_binary),
        "detector_sha256": sha256_file(args.detector),
        "selection_sha256": sha256_file(args.selection),
        "selection_rule": "minimum fixed validation proxy, corpus weights 253/254 and 1/254; no auto promotion",
        "bn_policy": "legacy supervised region training stats; eval frozen; burn-in never updates BN",
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("registered study differs; refuse mutation")
    write_json_report(config, config_path)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()
    provenance = {
        "data_recipe_sha256": next(iter(rows.values()))["recipe_hash"],
        "manifest_set_sha256": digest(hashes),
        "source_git_sha": source,
        "validation_sha256": digest(specs),
    }
    if (root / "provenance.json").exists():
        recorded = json.loads((root / "provenance.json").read_text())
        if recorded["training_source_sha256"] != config["training_source_sha256"]:
            raise ValueError("training source changed")
        provenance["source_git_sha"] = recorded["source_git_sha"]
    else:
        write_json_report(
            {
                **provenance,
                "training_source_sha256": config["training_source_sha256"],
                "source_files": source_hashes,
            },
            root / "provenance.json",
        )
    clips = PreparedClips(rows, args.detector, root / "cache")
    summary = []
    for seed in args.seeds:
        schedule, draw_report = make_schedule(rows, seed, args.steps * args.batch_size)
        write_json_report(
            {"schedule_sha256": digest(schedule), "draws": schedule, "sampling": draw_report},
            root / f"draws-{seed}.json",
        )
        for kind in args.kinds:
            arm_provenance = {**provenance, "schedule_sha256": digest(schedule)}
            receipt = run_arm(
                kind,
                seed,
                config,
                arm_provenance,
                schedule,
                clips,
                specs,
                root / f"{kind}-{seed}",
                args.device,
            )
            summary.append(receipt)
            write_json_report(
                {"status": "running", "completed": summary, "promoted": False},
                root / "summary.json",
            )
    write_json_report(
        {"status": "complete", "completed": summary, "promoted": False}, root / "summary.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-interval", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261012, 20261013])
    parser.add_argument(
        "--kinds", choices=["legacy", "gru", "tcn"], nargs="+", default=["legacy", "gru", "tcn"]
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.validation_interval, args.threads) < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.kinds)) != len(args.kinds)
    ):
        parser.error("invalid budget/seed/learning rate")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another process owns this study") from None
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json_report(
            {"status": "failed", "error": traceback.format_exc(), "pid": os.getpid()},
            args.output / "failure.json",
        )
        raise
