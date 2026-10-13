#!/usr/bin/env python3
"""P19: 32 versus 253 music recordings, fixed speech, updates and music exposures."""

import argparse
import fcntl
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data.causal_study import digest, load_sources
from data.coverage_study import PoolView, banks, schedule
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from models.spectral_consistency import consistency_loss
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import complete, read, registered, review, verify_receipt
from scripts.run_consistency_lr_study import evaluate_bank, restore
from scripts.run_fit_diagnostic import metrics
from scripts.run_subset_study import Pool, aggregate, features


def tensor_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def full_bank_score(model, pool, weight):
    records, auxiliary = [], {"music_library": [], "speech_library": []}
    mode = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            for i, row in enumerate(pool.rows):
                objective, context = pool.select([i])
                prediction = features(model, objective, context)
                score = metrics(objective(prediction), objective.batch)
                auxiliary[row["corpus"]].append(
                    float(consistency_loss(prediction, objective.batch))
                )
                records.append(
                    {
                        "spec": pool.specs[i],
                        "track_id": row["track_id"],
                        "corpus": row["corpus"],
                        "metrics": score,
                    }
                )
    finally:
        model.train(mode)
    score = aggregate(records)
    aux = 0.8 * float(np.mean(auxiliary["music_library"])) + 0.2 * float(
        np.mean(auxiliary["speech_library"])
    )
    return {
        **score,
        "auxiliary": aux,
        "combined": score["weighted"]["total"] + weight * aux,
        "rows": records,
    }


def register(args, root):
    parent = read(args.p16 / "config.json")
    if sha256_file(args.parent) != parent["parent_sha256"]:
        raise ValueError("parent config drift")
    rows, manifests = load_sources([Path(p) for p in parent["manifests"]])
    if manifests != parent["manifests"]:
        raise ValueError("manifest drift")
    groups, selection = banks(parent["train_pools"]["40"], rows)
    if len(groups["wide"]["anchors"]) != 261 or len(groups["small"]["bank"]) != 320:
        raise ValueError("expected 253/32 music and the same 8 speech recordings")
    train_ids = {rows[s["row_id"]]["track_id"] for s in groups["wide"]["anchors"]}
    if len(parent["development_specs"]) != 89 or any(
        rows[s["row_id"]]["track_id"] in train_ids for s in parent["development_specs"]
    ):
        raise ValueError("development recording leakage")
    for spec in groups["wide"]["anchors"]:
        row = rows[spec["row_id"]]
        if row["corpus"] == "music_library":
            rights = row.get("source_metadata", {}).get("source_review", {})
            if (
                rights.get("status") != "accepted"
                or not rights.get("commercial_training")
                or not rights.get("apache_weight_distribution")
            ):
                raise ValueError("music recording lacks the existing accepted source review")
    source = Path(__file__).resolve().parents[1]
    prior = read(args.p16 / "source-files.json")
    if any(sha256_file(source / name) != expected for name, expected in prior.items()):
        raise ValueError("P16 model/loss dependencies changed")
    inputs = {str(p.resolve()): sha256_file(p) for p in (args.p16 / "config.json", args.parent)}
    for name, key in (
        ("binaries/soundex-gain-data", "detector_sha256"),
        ("binaries/soundex-prediction-replay", "replay_sha256"),
        ("replay-contract.onnx", "onnx_sha256"),
    ):
        path = args.p16 / name
        if sha256_file(path) != parent[key]:
            raise ValueError("binary/contract drift")
        inputs[str(path.resolve())] = sha256_file(path)
    inputs.update(manifests)
    exposure = {}
    schedules = {}
    for label, group in groups.items():
        end = 2048 if label == "small" else 16192
        for seed in parent["seeds"]:
            draws = schedule(group["anchors"], rows, end, seed)
            schedules[label, seed] = draws
            counts = {}
            for step in sorted({2048, end}):
                recording = np.bincount(
                    np.array(draws[:step]).reshape(-1) // 8, minlength=len(group["anchors"])
                )
                crop = np.bincount(np.array(draws[:step]).reshape(-1), minlength=len(group["bank"]))
                count = len(group["anchors"]) - 8
                counts[str(step)] = {
                    "music_recording_min": int(recording[:count].min()),
                    "music_recording_max": int(recording[:count].max()),
                    "speech_recording_min": int(recording[count:].min()),
                    "speech_recording_max": int(recording[count:].max()),
                    "music_crop_min": int(crop[: count * 8].min()),
                    "music_crop_max": int(crop[: count * 8].max()),
                    "speech_draws": step,
                    "music_draws": step * 4,
                }
            if (
                counts[str(end)]["music_recording_min"] != 256
                or counts[str(end)]["music_recording_max"] != 256
            ):
                raise ValueError("unequal registered music exposure")
            exposure[f"{label}-{seed}"] = counts
            registered(
                {"draws": draws, "sha256": digest(draws)}, root / f"draws-{label}-{seed}.json"
            )
    for seed in parent["seeds"]:
        for a, b in zip(schedules["small", seed], schedules["wide", seed][:2048], strict=True):
            if groups["small"]["bank"][a[-1]] != groups["wide"]["bank"][b[-1]]:
                raise ValueError("speech schedule differs at common updates")
    config = {
        "format": "soundex-music-coverage-v1",
        "scope": "P19 diagnostic music coverage; reused development; no promotion",
        "seeds": parent["seeds"],
        "rate": 0.0001,
        "weight": parent["consistency_weight"],
        "initialization": "fresh same seeded model for each arm; Adam no decay, clip=5; no P17/P18 checkpoint reuse",
        "budgets": {"small": 2048, "wide": 16192},
        "endpoints": {"small": [2048], "wide": [2048, 16192]},
        "schedule_hashes": {
            f"{label}-{seed}": digest(draws) for (label, seed), draws in schedules.items()
        },
        "exposure": exposure,
        "groups": groups,
        "selection": selection,
        "speech_caveat": "same eight speech recordings, 4 music+1 speech per batch; identical speech draws through 2048; wide equal-music-exposure endpoint repeats each speech recording 2024 times versus 256 for small",
        "development_specs": parent["development_specs"],
        "shifted_specs": parent["shifted_specs"],
        "support_rule": "compare wide-2048 vs small-2048 at common updates; wide-16192 vs small-2048 at equal music exposure, not equal speech exposure; both seeds must improve music restoration vs small and initial DSP under P18 support_checks; no dev-based selection",
        "native_count": 1246,
        "tolerance": 1e-6,
        "coverage": {
            label: {
                key: dict(Counter(str(rows[s["row_id"]].get(key)) for s in group["anchors"]))
                for key in ("corpus", "codec_id", "sample_rate", "channel_role")
            }
            for label, group in groups.items()
        },
        "genres": {
            label: dict(
                Counter(
                    rows[s["row_id"]].get("source_metadata", {}).get("genre", "unlabeled")
                    for s in group["anchors"]
                    if rows[s["row_id"]]["corpus"] == "music_library"
                )
            )
            for label, group in groups.items()
        },
        "input_hashes": inputs,
        "source_hashes": {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*.py")
            if not any(
                x in {"tests", ".venv", "outputs", "__pycache__"}
                for x in p.relative_to(source).parts
            )
        },
    }
    registered(config, root / "registration.json")
    return config, rows


def train(label, seed, root, config, pool, common, common_draws):
    directory = root / f"{label}-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return
    draws = read(root / f"draws-{label}-{seed}.json")["draws"]
    if digest(draws) != config["schedule_hashes"][f"{label}-{seed}"]:
        raise ValueError("registered draw schedule changed")
    signature = digest({"config": config, "label": label, "seed": seed})
    torch.manual_seed(seed)
    model = make_rendered_model("rendered_balanced")
    initial = tensor_digest(model)
    registered(
        {
            "seed": seed,
            "tensor_sha256": initial,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        root / f"initialization-{seed}.json",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["rate"])
    start, history = 0, []
    latest = directory / "latest.pth"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["signature"] != signature or state["initial_sha256"] != initial:
            raise ValueError("resume changed")
        optimizer = restore(model, state, config["rate"])
        start, history = state["step"], state["history"]
        if (
            not 0 <= start <= config["budgets"][label]
            or len(history) != start
            or any(int(v["step"]) != start for v in optimizer.state.values())
        ):
            raise ValueError("invalid restored training state")
    end = config["budgets"][label]
    for step in range(start, end + 1):
        if step > start:
            objective, context = pool.select(draws[step - 1])
            optimizer.zero_grad(set_to_none=True)
            prediction = features(model, objective, context)
            main = objective(prediction)["total"]
            aux = consistency_loss(prediction, objective.batch)
            loss = main + config["weight"] * aux
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "main": float(main.detach()),
                    "auxiliary": float(aux.detach()),
                    "preclip_gradient_norm": float(grad),
                }
            )
        score_step = step % 2048 == 0 or step == end
        if step % 64 == 0 or score_step:
            value = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "signature": signature,
                "initial_sha256": initial,
                "history": history,
            }
            atomic_save(value, latest)
            if score_step:
                checkpoint = directory / f"step-{step}.pth"
                if not checkpoint.exists():
                    atomic_save(value, checkpoint)
                registered(
                    {"step": step, "checkpoint_sha256": sha256_file(checkpoint)},
                    directory / f"registration-{step}.json",
                )
                target = directory / f"training-{step}.json"
                if not target.exists():
                    report = {
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "common": evaluate_bank(model, common, common_draws[:64], config["weight"]),
                    }
                    if step in config["endpoints"][label]:
                        report["arm_bank"] = (
                            report["common"]
                            if label == "small"
                            else full_bank_score(model, pool, config["weight"])
                        )
                    write_json_report(report, target)
                score = read(target)
                if score["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise ValueError("score checkpoint mismatch")
                info = {
                    "stage": "training_score",
                    "arm": label,
                    "seed": seed,
                    "step": step,
                    "end": end,
                    **score["common"]["weighted"],
                    "auxiliary": score["common"]["auxiliary"],
                }
            else:
                info = {
                    "stage": "training",
                    "arm": label,
                    "seed": seed,
                    "step": step,
                    "end": end,
                    **history[-1],
                }
            print(json.dumps(info), flush=True)
            write_json_report(info, root / "progress.json")
    write_json_report(history, directory / "updates.json")
    complete(directory)


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, rows = register(args, root)
        if args.prepare_only:
            print(
                json.dumps(
                    {"registered": True, "genres": config["genres"], "exposure": config["exposure"]}
                ),
                flush=True,
            )
            return
        clips = EnvelopeClips(
            rows, args.p16 / "binaries/soundex-gain-data", Path(read(args.parent)["cache_path"])
        )
        for label in ("small", "wide"):
            print(
                json.dumps(
                    {
                        "stage": "prepare_bank",
                        "arm": label,
                        "positions": len(config["groups"][label]["bank"]),
                    }
                ),
                flush=True,
            )
            pool = Pool(config["groups"][label]["bank"], clips)
            common = PoolView(pool, config["groups"]["small"]["bank"])
            for seed in config["seeds"]:
                common_draws = read(root / f"draws-small-{seed}.json")["draws"]
                train(label, seed, root, config, pool, common, common_draws)
            del common, pool
            gc.collect()
        checkpoints = {
            f"{label}-{step}-{seed}": root / f"{label}-{seed}/step-{step}.pth"
            for label in config["endpoints"]
            for step in config["endpoints"][label]
            for seed in config["seeds"]
        }
        registered(
            {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in checkpoints.items()
            },
            root / "review-registration.json",
        )
        pools = {
            "development": Pool(config["development_specs"], clips),
            "shifted": Pool(config["shifted_specs"], clips),
        }
        for label, checkpoint in {"initial": None, **checkpoints}.items():
            review(
                label,
                checkpoint,
                root,
                pools,
                args.p16 / "binaries/soundex-prediction-replay",
                args.p16 / "replay-contract.onnx",
            )
        for path, expected in config["input_hashes"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("input drift")
        for name, expected in config["schedule_hashes"].items():
            if digest(read(root / f"draws-{name}.json")["draws"]) != expected:
                raise ValueError("draw schedule drift")
        write_json_report(
            {
                "status": "complete",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
                "initialization_hashes": {
                    str(p.relative_to(root)): sha256_file(p)
                    for p in root.glob("initialization-*.json")
                },
            },
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p16", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
