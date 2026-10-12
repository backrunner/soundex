#!/usr/bin/env python3
"""P5: matched fresh amplitude-only short/tiled/past representation experiment.

Run with PYTHONPATH=training. Artifacts are diagnostics, not production weights.
The P4 config supplies audited sources/splits only; no parent weights are loaded.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from data.causal_study import digest, load_sources, make_schedule
from data.envelope_study import EnvelopeClips
from data.weak_band_study import add_identity_controls, audit_groups, identity_specs
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_objective import BANDS, amplitude_metrics, envelope_objective
from models.envelope_probe import ARMS, EnvelopeProbe, context_features, predict_probe
from scripts.run_causal_study import atomic_save
from scripts.run_weak_band_study import freeze_binary


def score_objective(objective_fn, prediction, target, batch, *, burn):
    argument = batch if getattr(objective_fn, "uses_batch", False) else batch["rate"]
    return objective_fn(prediction, target, argument, burn=burn)


@torch.no_grad()
def evaluate(
    model, clips, specs, arm, device, detailed=False, *, objective_fn=envelope_objective, burn=64
):
    if model is not None:
        model.eval()
    records = []
    for spec in specs:
        batch = clips.batch([spec], device)
        context = context_features(batch["pcm"], arm).to(device)
        target = torch.pow(10.0, batch["y"][:, 0] / 20)
        pred = (
            predict_probe(model, batch, context, burn=burn)
            if model is not None
            else torch.pow(10.0, batch["x"][:, 0] / 20)
        )
        metrics = {
            k: float(v)
            for k, v in score_objective(objective_fn, pred, target, batch, burn=burn).items()
        }
        if detailed:
            metrics.update(
                amplitude_metrics(pred.cpu(), target.cpu(), batch["rate"].cpu(), burn=burn)
            )
        row = clips.rows[spec["row_id"]]
        meta = row.get("source_metadata", {})
        records.append(
            {
                "row_id": spec["row_id"],
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "codec_id": row["codec_id"],
                "sample_rate": row["sample_rate"],
                "channel_role": row["channel_role"],
                "genre": meta.get("genre", "unlabeled"),
                "region": meta.get("music_region", "unlabeled"),
                "metrics": metrics,
            }
        )
    by_corpus = {
        corpus: {
            key: float(np.mean([r["metrics"][key] for r in records if r["corpus"] == corpus]))
            for key in records[0]["metrics"]
        }
        for corpus in ("music_library", "speech_library")
    }
    weighted = (
        253 * by_corpus["music_library"]["total"] + by_corpus["speech_library"]["total"]
    ) / 254
    if model is not None:
        model.train()
    return {"score": weighted, "corpus": by_corpus, "rows": records}


def validate(model, clips, specs, arm, device, **kwargs):
    paired = evaluate(model, clips, specs, arm, device, **kwargs)
    identity = evaluate(model, clips, identity_specs(specs), arm, device, **kwargs)
    return {
        "score": 0.875 * paired["score"] + 0.125 * identity["score"],
        "paired": paired,
        "identity": identity,
    }


def run_arm(
    config,
    seed,
    arm,
    draws,
    clips,
    root,
    *,
    model_factory=EnvelopeProbe,
    training_objective=envelope_objective,
    selection_objective=envelope_objective,
    context_arm=None,
    burn=64,
    include_initial=False,
):
    directory = root / f"{arm}-{seed}"
    directory.mkdir(exist_ok=True)
    signature = digest({"config": config, "seed": seed, "arm": arm, "draws": draws})
    complete = directory / "complete.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if (
            receipt["signature"] != signature
            or receipt["checkpoint_sha256"] != sha256_file(directory / "best.pth")
            or receipt["evaluation_sha256"] != sha256_file(directory / "evaluation.json")
        ):
            raise ValueError("completed arm changed")
        return receipt
    torch.manual_seed(seed)
    model = model_factory().to(config["device"])
    context_arm = context_arm or arm
    scorer = partial(evaluate, objective_fn=selection_objective, burn=burn)
    validator = partial(
        validate,
        clips=clips,
        specs=config["validation_specs"],
        arm=context_arm,
        device=config["device"],
        objective_fn=selection_objective,
        burn=burn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    start, best = 0, float("inf")
    latest = directory / "latest.pth"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("resume configuration/source/draws changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        if config["device"] == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        start, best = checkpoint["step"], checkpoint["best_score"]
    initial = directory / "initial.json"
    if not initial.exists():
        write_json_report(validator(model), initial)

    def save_state(step, result, improved):
        checkpoint = {
            "format": config.get("checkpoint_format", "envelope-probe-v1"),
            "signature": signature,
            "step": step,
            "arm": arm,
            "seed": seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best,
            "validation": result,
            "config": config,
            "rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if config["device"] == "mps" else None,
        }
        # Publish best before latest so an interrupted improved checkpoint cannot
        # leave a latest checkpoint that refers to a missing best artifact.
        if improved:
            atomic_save(checkpoint, directory / "best.pth")
        atomic_save(checkpoint, latest)

    if include_initial and not latest.exists():
        result = json.loads(initial.read_text())
        best = result["score"]
        save_state(0, result, True)
    began = time.monotonic()
    for step in range(start + 1, config["steps"] + 1):
        model.train()
        batch = clips.batch(draws[(step - 1) * 4 : step * 4], config["device"])
        context = context_features(batch["pcm"], context_arm).to(config["device"])
        target = torch.pow(10.0, batch["y"][:, 0] / 20)
        optimizer.param_groups[0]["lr"] = config["learning_rate"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / config["steps"]))
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_probe(model, batch, context, burn=burn)
        loss = score_objective(training_objective, prediction, target, batch, burn=burn)["total"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step % 32 == 0 or step == 1:
            progress = {
                "arm": arm,
                "seed": seed,
                "step": step,
                "steps": config["steps"],
                "loss": float(loss.detach()),
                "segment_seconds": time.monotonic() - began,
            }
            write_json_report(progress, root / "progress.json")
            print(json.dumps(progress), flush=True)
        if step % config["validation_interval"] == 0 or step == config["steps"]:
            result = validator(model)
            improved = result["score"] < best
            best = min(best, result["score"])
            save_state(step, result, improved)
            write_json_report(result, directory / f"validation-{step}.json")
    checkpoint = torch.load(directory / "best.pth", map_location="cpu", weights_only=False)
    if checkpoint["signature"] != signature:
        raise ValueError("best checkpoint changed")
    model.load_state_dict(checkpoint["model"])
    reports = {}
    for name, specs in (
        ("validation", config["validation_specs"]),
        ("identity", identity_specs(config["validation_specs"])),
        ("train_probe", config["training_probe_specs"]),
    ):
        reports[name] = scorer(model, clips, specs, context_arm, config["device"], detailed=True)
    if arm in config.get("history_ablation_arms", ["past"]):
        for alternative in ("short", "tiled"):
            reports["ablation_" + alternative] = scorer(
                model,
                clips,
                config["validation_specs"],
                alternative,
                config["device"],
                detailed=True,
            )
    write_json_report(reports, directory / "evaluation.json")
    receipt = {
        "status": "complete",
        "signature": signature,
        "arm": arm,
        "seed": seed,
        "best_step": checkpoint["step"],
        "best_score": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256_file(directory / "best.pth"),
        "evaluation_sha256": sha256_file(directory / "evaluation.json"),
    }
    write_json_report(receipt, complete)
    return receipt


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = json.loads(args.parent.read_text())
    rows, manifests = load_sources([Path(p) for p in parent["manifests"]])
    if manifests != parent["manifests"]:
        raise ValueError("source manifests changed")
    detector = freeze_binary(
        args.parent.parent / "binaries/soundex-gain-data",
        root / "binaries/soundex-gain-data",
        parent["detector_sha256"],
    )
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in sorted(source.rglob("*.py"))
        if not any(
            x in {".venv", "tests", "outputs", "__pycache__"} for x in p.relative_to(source).parts
        )
    }
    specs = parent["validation_specs"]
    train_specs = parent["training_probe_specs"]
    if args.smoke:

        def subset(values):
            return [
                next(s for s in values if rows[s["row_id"]]["corpus"] == c)
                for c in ("music_library", "speech_library")
            ]

        specs, train_specs = subset(specs), subset(train_specs)
    config = {
        "format": "soundex-envelope-study-v1",
        "steps": args.steps,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "seeds": args.seeds,
        "arms": list(ARMS),
        "device": args.device,
        "smoke": args.smoke,
        "validation_interval": args.validation_interval,
        "validation_specs": specs,
        "training_probe_specs": train_specs,
        "manifests": manifests,
        "parent_config_sha256": sha256_file(args.parent),
        "parent_weights_used": False,
        "source_sha256": digest(hashes),
        "detector_sha256": sha256_file(detector),
        "coverage": audit_groups(rows),
        "bands_hz": list(map(list, BANDS)),
        "objective": "amplitude-log1p-v1",
        "burn_hops": 64,
        "context_stride": 4,
        "identity_selection_weight": 0.125,
        "torch_version": str(torch.__version__),
        "independent_test": False,
        "test_scored": False,
        "deployable_audio_model": False,
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("registered study changed")
    write_json_report(config, config_path)
    write_json_report(hashes, root / "source-files.json")
    clips = EnvelopeClips(rows, detector, root / "cache")
    if not (root / "compressed-reference.json").exists():
        write_json_report(
            evaluate(None, clips, specs, "short", args.device, detailed=True),
            root / "compressed-reference.json",
        )
    completed = []
    for seed in args.seeds:
        schedule, sampling = make_schedule(rows, seed + 30000, args.steps * 4)
        draws = add_identity_controls(schedule, 4)
        write_json_report(
            {"draws": draws, "sampling_before_identity": sampling, "sha256": digest(draws)},
            root / f"draws-{seed}.json",
        )
        for arm in ARMS:
            completed.append(run_arm(config, seed, arm, draws, clips, root))
    write_json_report(
        {
            "status": "complete",
            "arms": completed,
            "scope": "raw amplitude only; no audio quality or release qualification",
        },
        root / "summary.json",
    )
    lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, required=True, help="P4 config supplying sources and splits"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261012, 20261013])
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.steps < 1 or options.validation_interval < 1:
        parser.error("positive steps and interval required")
    run(options)
