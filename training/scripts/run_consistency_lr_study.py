#!/usr/bin/env python3
"""P17: matched Adam continuation from P16 step 256; training-only diagnosis."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from models.spectral_consistency import consistency_loss
from scripts.run_causal_study import atomic_save
from scripts.run_subset_study import Pool, aggregate, features, native


def restore(model, checkpoint, rate):
    """Keep Adam moments/counters and RNG; change only the learning rate."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("positive finite learning rate required")
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = rate
    torch.set_rng_state(checkpoint["rng"])
    model.train()
    return optimizer


@torch.no_grad()
def evaluate_bank(model, pool, draws, weight):
    """Evaluate every crop once, with the same paired/corpus risk as training."""
    if sorted(i for draw in draws for i in draw) != list(range(len(pool.specs))):
        raise ValueError("evaluation must cover the complete bank exactly once")
    records, auxiliary = [], []
    mode = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            for draw in draws:
                objective, context = pool.select(draw)
                prediction = features(model, objective, context)
                score = objective(prediction)
                auxiliary.append(float(consistency_loss(prediction, objective.batch)))
                for j, index in enumerate(draw):
                    pair = float(score["restoration"][j])
                    identity = float(score["preservation"][j + len(draw)])
                    row = pool.rows[index]
                    records.append(
                        {
                            "spec": pool.specs[index],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "metrics": {
                                "total": 0.5 * (pair + identity),
                                "paired_restoration": pair,
                                "identity_preservation": identity,
                            },
                        }
                    )
    finally:
        model.train(mode)
    result = aggregate(records)
    aux = float(np.mean(auxiliary))
    return {
        **result,
        "auxiliary": aux,
        "combined": result["weighted"]["total"] + weight * aux,
        "rows": records,
    }


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "registration.json").exists():
        raise ValueError("preserve existing registered experiment; use a fresh output directory")
    config = json.loads((args.study / "config.json").read_text())
    if config["seeds"] != [20261012, 20261013] or config["consistency_weight"] <= 0:
        raise ValueError("requires registered P16 consistency arm")
    if sha256_file(args.parent) != config["parent_sha256"]:
        raise ValueError("parent config drift")
    source = Path(__file__).resolve().parents[1]
    prior_sources = json.loads((args.study / "source-files.json").read_text())
    if any(sha256_file(source / name) != value for name, value in prior_sources.items()):
        raise ValueError("P16 dependency source drift")
    inputs = {
        str(path.resolve()): sha256_file(path)
        for path in (args.parent, args.p16_log, args.study / "config.json")
    }
    schedules, checkpoints = {}, {}
    for seed in config["seeds"]:
        directory = args.study / f"size-40-{seed}"
        receipt = json.loads((directory / "complete.json").read_text())
        for name, expected in receipt["hashes"].items():
            path = directory / name
            if sha256_file(path) != expected:
                raise ValueError(f"P16 receipt drift: {path}")
            inputs[str(path.resolve())] = expected
        schedules[seed] = json.loads((directory / "draws.json").read_text())["crop_draws"]
        checkpoints[seed] = directory / "step-256.pth"
    binary = args.study / "binaries/soundex-prediction-replay"
    detector = args.study / "binaries/soundex-gain-data"
    onnx = args.study / "replay-contract.onnx"
    for path, key in (
        (binary, "replay_sha256"),
        (detector, "detector_sha256"),
        (onnx, "onnx_sha256"),
    ):
        if sha256_file(path) != config[key]:
            raise ValueError(f"binary/contract drift: {path}")
        inputs[str(path.resolve())] = sha256_file(path)
    rates, milestones = (0.001, 0.0001), (320, 384, 448, 512)
    registration = {
        "scope": "P17 training-only optimization diagnosis; no development selection or promotion",
        "start_step": 256,
        "end_step": 512,
        "rates": rates,
        "seeds": config["seeds"],
        "milestones": milestones,
        "weight": config["consistency_weight"],
        "sampling": "continue registered crop_draws[256:512]; evaluate first 64 draws covering all 320 positions",
        "decision": "report both seeds at all fixed points; assess original main, paired, identity, auxiliary and combined, never select the best checkpoint",
        "support_rule": "both seeds: final main and combined below start and original-rate endpoint by >1e-6; paired and identity <= start +1e-6; report intermediate worst score and counts above start separately",
        "input_hashes": inputs,
        "source_hashes": {
            **prior_sources,
            str(Path(__file__).relative_to(source)): sha256_file(Path(__file__)),
        },
        "native": "fixed P13 five anchor pairs plus identity, at common start and all four endpoints; 60 replays, no candidate ONNX inference",
    }
    write_json_report(registration, root / "registration.json")
    parent = json.loads(args.parent.read_text())
    rows, manifests = load_sources([Path(p) for p in config["manifests"]])
    if manifests != config["manifests"]:
        raise ValueError("manifest drift")
    bank = config["crop_sampling"]["bank"]
    if len(bank) != 320 or any(
        rows[s["row_id"]]["split"] != "train" or rows[s["row_id"]].get("held_out", False)
        for s in bank
    ):
        raise ValueError("requires 320 registered training crops")
    if any(
        len(draw) != 5
        or sum(rows[bank[i]["row_id"]]["corpus"] == "music_library" for i in draw) != 4
        for schedule in schedules.values()
        for draw in schedule[:512]
    ):
        raise ValueError("requires 4 music and 1 speech per batch")
    pool = Pool(bank, EnvelopeClips(rows, detector, Path(parent["cache_path"])))
    native_indices = [bank.index(s) for s in config["train_pools"]["5"]]
    old_logs = {}
    for line in args.p16_log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("preupdate_consistency_loss", 0) > 0:
            old_logs[(row["run"], row["step"])] = row
    results = []
    weight = config["consistency_weight"]
    for seed in config["seeds"]:
        state = torch.load(checkpoints[seed], map_location="cpu", weights_only=False)
        if state["step"] != 256 or any(
            int(value["step"]) != 256 for value in state["optimizer"]["state"].values()
        ):
            raise ValueError("requires complete Adam state at step 256")
        model = make_rendered_model("rendered_balanced")
        restore(model, state, rates[0])
        baseline = evaluate_bank(model, pool, schedules[seed][:64], weight)
        previous = json.loads((args.study / f"size-40-{seed}/evaluation-256.json").read_text())[
            "training_bank_proxy"
        ]["weighted"]
        if any(abs(baseline["weighted"][k] - v) > 1e-6 for k, v in previous.items()):
            raise ValueError("starting checkpoint score does not reproduce P16")
        baseline["native"] = native(model, pool, native_indices, binary, onnx)
        write_json_report(baseline, root / f"start-{seed}.json")
        for rate in rates:
            directory = root / f"seed-{seed}-lr-{rate}"
            directory.mkdir()
            optimizer = restore(model, state, rate)
            progress = []
            for step in range(257, 513):
                objective, context = pool.select(schedules[seed][step - 1])
                optimizer.zero_grad(set_to_none=True)
                prediction = features(model, objective, context)
                main = objective(prediction)["total"]
                aux = consistency_loss(prediction, objective.batch)
                loss = main + weight * aux
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
                batch_log = {
                    "step": step,
                    "preupdate_main": float(main.detach()),
                    "preupdate_auxiliary": float(aux.detach()),
                    "preclip_gradient_norm": float(grad),
                }
                if rate == 0.001 and step in milestones:
                    expected = old_logs[(f"size-40-{seed}", step)]
                    if (
                        abs(batch_log["preupdate_main"] - expected["preupdate_main_loss"]) > 1e-7
                        or abs(
                            batch_log["preupdate_auxiliary"]
                            - expected["preupdate_consistency_loss"]
                        )
                        > 1e-7
                    ):
                        raise ValueError("original-rate continuation does not reproduce P16")
                progress.append(batch_log)
                if step not in milestones:
                    continue
                checkpoint = directory / f"step-{step}.pth"
                atomic_save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                    },
                    checkpoint,
                )
                checkpoint_hash = sha256_file(checkpoint)
                write_json_report(
                    {"step": step, "checkpoint_sha256": checkpoint_hash},
                    directory / f"registration-{step}.json",
                )
                evaluation = evaluate_bank(model, pool, schedules[seed][:64], weight)
                evaluation.update(step=step, checkpoint_sha256=checkpoint_hash)
                if step == 512:
                    with torch.random.fork_rng(devices=[]):
                        evaluation["native"] = native(model, pool, native_indices, binary, onnx)
                write_json_report(evaluation, directory / f"evaluation-{step}.json")
                info = {
                    "seed": seed,
                    "rate": rate,
                    "step": step,
                    **evaluation["weighted"],
                    "auxiliary": evaluation["auxiliary"],
                    "combined": evaluation["combined"],
                }
                print(json.dumps(info), flush=True)
                write_json_report(info, root / "progress.json")
                results.append(info)
            write_json_report(progress, directory / "updates.json")
            write_json_report(
                {
                    "files": {
                        p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()
                    }
                },
                directory / "complete.json",
            )
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("frozen input changed during study")
    write_json_report(
        {
            "scope": registration["scope"],
            "registration_sha256": sha256_file(root / "registration.json"),
            "results": results,
        },
        root / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent", "p16-log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
