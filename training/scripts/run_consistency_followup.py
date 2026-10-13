#!/usr/bin/env python3
"""P18: fixed low-rate continuation, followed by frozen cross-recording review."""

import argparse
import fcntl
import json
from pathlib import Path

import torch

from data.causal_study import digest, load_sources
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from models.spectral_consistency import consistency_loss
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_lr_study import evaluate_bank, restore
from scripts.run_subset_study import Pool, evaluate, features, native


def read(path):
    return json.loads(path.read_text())


def registered(value, path):
    """Idempotent registration must never silently change a running experiment."""
    value = json.loads(json.dumps(value))
    if path.exists() and read(path) != value:
        raise ValueError(f"registered artifact changed: {path}")
    if not path.exists():
        write_json_report(value, path)


def verify_receipt(directory, field):
    receipt = read(directory / "complete.json")
    for name, expected in receipt[field].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"completed artifact changed: {directory / name}")


def complete(directory):
    registered(
        {
            "files": {
                str(p.relative_to(directory)): sha256_file(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p != directory / "complete.json"
            }
        },
        directory / "complete.json",
    )


def prepare(args, root):
    config = read(args.p16 / "config.json")
    p17 = read(args.p17 / "registration.json")
    if config["seeds"] != [20261012, 20261013] or p17["end_step"] != 512:
        raise ValueError("requires the fixed two-seed P16/P17 studies")
    source = Path(__file__).resolve().parents[1]
    prior = read(args.p16 / "source-files.json")
    prior.update(p17["source_hashes"])
    if any(sha256_file(source / name) != expected for name, expected in prior.items()):
        raise ValueError("training dependency changed since P17")
    inputs = dict(p17["input_hashes"])
    inputs.update(
        {
            str(p.resolve()): sha256_file(p)
            for p in (
                args.p16 / "config.json",
                args.p17 / "registration.json",
                args.p17 / "summary.json",
                args.parent,
            )
        }
    )
    if sha256_file(args.parent) != config["parent_sha256"]:
        raise ValueError("parent config changed")
    for seed in config["seeds"]:
        for directory, field in (
            (args.p16 / f"size-40-{seed}", "hashes"),
            (args.p17 / f"seed-{seed}-lr-0.0001", "files"),
        ):
            verify_receipt(directory, field)
            for name, expected in read(directory / "complete.json")[field].items():
                inputs[str((directory / name).resolve())] = expected
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"frozen input changed: {path}")
    registration = {
        "format": "soundex-consistency-followup-v1",
        "scope": "P18 fixed-budget continuation; reused development review, not independent test; no promotion",
        "seeds": config["seeds"],
        "start_step": 512,
        "end_step": 2048,
        "rate": 0.0001,
        "weight": config["consistency_weight"],
        "training_checks": list(range(512, 2049, 256)),
        "policy": "restore full P17 low-rate state; continue original draws[512:2048]; save both final hashes before any development scoring; no early selection",
        "support_rule": "both seeds: music development main and paired improve versus start and initial DSP in clip and recording means; native paired high_sc/high_flux improve versus both, high_energy_error_db and low-band SC do not worsen versus start by >1e-6; identity does not worsen versus start; otherwise no evidence of consistent restoration",
        "tolerance": 1e-6,
        "native_policy": "all 89 development pairs and identity for initial DSP, both P17 starts, both P16 high-rate endpoints, both P18 endpoints: 1246 replays",
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
    registered(registration, root / "registration.json")
    return config, registration


def train(seed, args, root, config, registration, pool):
    directory = root / f"seed-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return directory / "step-2048.pth"
    draws = read(args.p16 / f"size-40-{seed}/draws.json")["crop_draws"]
    if len(draws) != 2048 or any(
        len(d) != 5 or sum(pool.rows[i]["corpus"] == "music_library" for i in d) != 4 for d in draws
    ):
        raise ValueError("requires registered 2048-step four-music/one-speech schedule")
    signature = digest({"registration": registration, "seed": seed})
    latest = directory / "latest.pth"
    resumed = latest.exists()
    state = torch.load(
        latest if resumed else args.p17 / f"seed-{seed}-lr-0.0001/step-512.pth",
        map_location="cpu",
        weights_only=False,
    )
    if resumed and state.get("signature") != signature:
        raise ValueError("resume signature changed")
    if (not resumed and state["step"] != 512) or not 512 <= state["step"] <= 2048:
        raise ValueError("invalid continuation step")
    if not state["optimizer"]["state"] or any(
        int(v["step"]) != state["step"] for v in state["optimizer"]["state"].values()
    ):
        raise ValueError("Adam counters differ from checkpoint step")
    if any(g["lr"] != registration["rate"] for g in state["optimizer"]["param_groups"]):
        raise ValueError("unexpected starting learning rate")
    model = make_rendered_model("rendered_balanced")
    optimizer = restore(model, state, registration["rate"])
    history = state.get("history", [])
    start = state["step"]
    weight = config["consistency_weight"]
    for step in range(start, 2049):
        if step > start:
            objective, context = pool.select(draws[step - 1])
            optimizer.zero_grad(set_to_none=True)
            prediction = features(model, objective, context)
            main = objective(prediction)["total"]
            aux = consistency_loss(prediction, objective.batch)
            loss = main + weight * aux
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "preupdate_main": float(main.detach()),
                    "preupdate_auxiliary": float(aux.detach()),
                    "preclip_gradient_norm": float(grad),
                }
            )
        if step % 64:
            continue
        value = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "signature": signature,
            "history": history,
        }
        atomic_save(value, latest)
        if step not in registration["training_checks"]:
            info = {
                "stage": "training",
                "seed": seed,
                "step": step,
                "end_step": 2048,
                **history[-1],
            }
        else:
            checkpoint = directory / f"step-{step}.pth"
            if not checkpoint.exists():
                atomic_save(value, checkpoint)
            checkpoint_hash = sha256_file(checkpoint)
            registered(
                {"step": step, "checkpoint_sha256": checkpoint_hash},
                directory / f"registration-{step}.json",
            )
            path = directory / f"training-{step}.json"
            if not path.exists():
                report = evaluate_bank(model, pool, draws[:64], weight)
                if step == 512:
                    expected = read(args.p17 / f"seed-{seed}-lr-0.0001/evaluation-512.json")
                    if (
                        any(
                            abs(report["weighted"][k] - v) > 1e-6
                            for k, v in expected["weighted"].items()
                        )
                        or abs(report["auxiliary"] - expected["auxiliary"]) > 1e-6
                    ):
                        raise ValueError("P17 start score changed")
                write_json_report(
                    {**report, "step": step, "checkpoint_sha256": checkpoint_hash}, path
                )
            report = read(path)
            if report["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("training evaluation checkpoint mismatch")
            info = {
                "stage": "full_training_bank",
                "seed": seed,
                "step": step,
                **report["weighted"],
                "auxiliary": report["auxiliary"],
                "combined": report["combined"],
            }
        print(json.dumps(info), flush=True)
        write_json_report(info, root / "progress.json")
    write_json_report(history, directory / "updates.json")
    complete(directory)
    return directory / "step-2048.pth"


def review(label, checkpoint, root, pools, binary, onnx):
    directory = root / "review" / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return
    model = None
    if checkpoint is not None:
        model = make_rendered_model("rendered_balanced")
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        )
        model.eval()
    checkpoint_hash = sha256_file(checkpoint) if checkpoint else None
    proxy_path = directory / "proxy.json"
    if not proxy_path.exists():
        write_json_report(
            {
                "checkpoint_sha256": checkpoint_hash,
                "groups": {name: evaluate(model, pool) for name, pool in pools.items()},
            },
            proxy_path,
        )
    if read(proxy_path)["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("review checkpoint changed")
    native_root = directory / "native"
    native_root.mkdir(exist_ok=True)
    pool = pools["development"]
    for i, spec in enumerate(pool.specs):
        path = native_root / f"{i:03d}.json"
        if path.exists():
            old = read(path)
            if old["spec"] != spec or old["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("native resume input changed")
            continue
        pair, identity = native(model, pool, [i], binary, onnx)
        row = pool.rows[i]
        write_json_report(
            {
                "spec": spec,
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "checkpoint_sha256": checkpoint_hash,
                "paired": pair,
                "identity": identity,
            },
            path,
        )
        if (i + 1) % 8 == 0 or i + 1 == len(pool.specs):
            info = {
                "stage": "native_review",
                "case": label,
                "pairs": i + 1,
                "total_pairs": len(pool.specs),
            }
            print(json.dumps(info), flush=True)
            write_json_report(info, root / "progress.json")
    complete(directory)


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, registration = prepare(args, root)
        rows, manifests = load_sources([Path(p) for p in config["manifests"]])
        if manifests != config["manifests"]:
            raise ValueError("manifest drift")
        bank = config["crop_sampling"]["bank"]
        if len(bank) != 320 or any(
            rows[s["row_id"]]["split"] != "train" or rows[s["row_id"]].get("held_out", False)
            for s in bank
        ):
            raise ValueError("invalid training bank")
        development = config["development_specs"]
        train_ids = {rows[s["row_id"]]["track_id"] for s in bank}
        if len(development) != 89 or any(
            rows[s["row_id"]]["track_id"] in train_ids for s in development
        ):
            raise ValueError("invalid cross-recording development set")
        clips = EnvelopeClips(
            rows, args.p16 / "binaries/soundex-gain-data", Path(read(args.parent)["cache_path"])
        )
        pool = Pool(bank, clips)
        checkpoints = {
            f"final-{seed}": train(seed, args, root, config, registration, pool)
            for seed in config["seeds"]
        }
        for seed in config["seeds"]:
            checkpoints[f"start-{seed}"] = args.p17 / f"seed-{seed}-lr-0.0001/step-512.pth"
            checkpoints[f"high-rate-{seed}"] = args.p16 / f"size-40-{seed}/step-2048.pth"
        # This barrier is reached only once both fixed endpoints have completed.
        registered(
            {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in checkpoints.items()
            },
            root / "review-registration.json",
        )
        del pool
        pools = {
            "shifted": Pool(config["shifted_specs"], clips),
            "development": Pool(development, clips),
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
        for path, expected in registration["input_hashes"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("input changed during study")
        write_json_report(
            {
                "status": "complete",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p16", "p17", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
