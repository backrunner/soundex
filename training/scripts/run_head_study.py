#!/usr/bin/env python3
"""P20: matched fresh anchored/amplitude/complex heads, frozen cross-recording review."""

import argparse
import fcntl
import gc
import hashlib
import math
from pathlib import Path

import torch

from data.causal_study import digest, load_sources
from data.coverage_study import PoolView
from data.envelope_study import EnvelopeClips
from evaluation.head_study import auxiliary, evaluate, predict, review
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import balanced_risk, budget_risk
from models.restoration_head import ARMS, RestorationHead, predict_budget, shared_state
from models.spectral_consistency import consistency_loss
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import complete, read, registered, verify_receipt
from scripts.run_consistency_lr_study import restore
from scripts.run_fit_diagnostic import select_specs
from scripts.run_music_coverage_study import tensor_digest
from scripts.run_subset_study import Pool


def progress(root, **info):
    write_json_report(info, root / "progress.json")
    print(info, flush=True)


def prepare(args, root):
    parent = read(args.p19 / "registration.json")
    inputs = dict(parent["input_hashes"])
    for p in [
        args.p19 / "registration.json",
        args.p19 / "summary.json",
        *args.p19.glob("draws-wide-*.json"),
    ]:
        inputs[str(p.resolve())] = sha256_file(p)
    for name, expected in inputs.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("parent input drift")
    source = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    rows, _ = load_sources([Path(p) for p in inputs if Path(p).name == "manifest.jsonl"])
    bank = parent["groups"]["wide"]["bank"]
    common = parent["groups"]["small"]["bank"]
    if args.smoke:
        bank = select_specs(parent["groups"]["wide"]["anchors"], rows)
        common = bank
    config = {
        "format": "soundex-head-study-v1",
        "arms": list(ARMS),
        "seeds": parent["seeds"],
        "scope": "training-only integration pilot"
        if args.smoke
        else "registered head comparison; reused development, not independent test; no promotion",
        "steps": 64 if args.smoke else 16192,
        "endpoints": [64] if args.smoke else [2048, 16192],
        "rate": parent["rate"],
        "weight": parent["weight"],
        "objective": "same rendered main; all heads use clean-complex reference consistency, unlike P19 input-phase reference",
        "risk": "4 music + 1 speech, paired/identity equal; Adam no decay; norm clip 5",
        "bank": bank,
        "common": common,
        "input_hashes": inputs,
        "source_hashes": sources,
        "development": bank if args.smoke else parent["development_specs"],
        "shifted": bank if args.smoke else parent["shifted_specs"],
        "selection": "none; all endpoints frozen before any held-out score",
        "smoke": args.smoke,
        "tolerance": 1e-6,
        "native_count": 70 if args.smoke else 2314,
        "screen": "both seeds: paired main, native HF SC/flux below matched anchored and identity DSP; identity, energy and low-band no worse than anchored; reused review only",
    }
    # Bind actual selected PCM, not just manifest text; no source/codec resampling.
    for row_id in sorted(
        {s["row_id"] for key in ("bank", "common", "development", "shifted") for s in config[key]}
    ):
        row = rows[row_id.removesuffix(":identity")]
        for role in ("clean", "degraded"):
            path = Path(row["_root"]) / row[f"{role}_path"]
            expected = row[f"{role}_checksum"]
            if sha256_file(path) != expected:
                raise ValueError("selected PCM checksum drift")
            inputs[str(path.resolve())] = expected
    for seed in config["seeds"]:
        draws = (
            [list(range(5))] * 64
            if args.smoke
            else read(args.p19 / f"draws-wide-{seed}.json")["draws"]
        )
        if len(draws) != config["steps"]:
            raise ValueError("schedule length mismatch")
        config.setdefault("schedule_hashes", {})[str(seed)] = digest(draws)
        registered({"draws": draws}, root / f"draws-{seed}.json")
    registered(config, root / "registration.json")
    return config, rows


def train(arm, seed, root, config, pool, common):
    directory = root / f"{arm}-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return
    schedule_key = config.get("schedule_groups", {}).get(arm)
    suffix = f"{schedule_key}-{seed}" if schedule_key else str(seed)
    draws = read(root / f"draws-{suffix}.json")["draws"]
    if digest(draws) != config["schedule_hashes"][suffix]:
        raise ValueError("draw schedule drift")
    torch.manual_seed(seed)
    model_config = config.get("model_configs", {}).get(arm)
    model = RestorationHead(**model_config) if model_config else RestorationHead(arm)
    rate = config.get("rates", {}).get(arm, config["rate"])
    shared = {}
    for name, tensor in shared_state(model).items():
        shared[name] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    width_key = f"-w{model_config.get('width', 144)}" if model_config else ""
    registered(shared, root / f"shared-initialization{width_key}-{seed}.json")
    initial = tensor_digest(model)
    signature = digest({"config": config, "arm": arm, "seed": seed})
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    start, history = 0, []
    if (directory / "latest.pth").exists():
        state = torch.load(directory / "latest.pth", map_location="cpu", weights_only=False)
        if state["signature"] != signature or state["initial_sha256"] != initial:
            raise ValueError("resume registration drift")
        optimizer = restore(model, state, rate)
        start, history = state["step"], state["history"]
        if (
            len(history) != start
            or not 0 <= start <= config["steps"]
            or any(int(v["step"]) != start for v in optimizer.state.values())
            or any(not math.isfinite(v) for row in history for v in row.values())
        ):
            raise ValueError("invalid resume step")
    for step in range(start, config["steps"] + 1):
        if step > start:
            model.train()
            objective, context = pool.select(draws[step - 1])
            optimizer.zero_grad(set_to_none=True)
            balanced = config.get("balanced_arms", [])
            budget_weight = config.get("budget_weights", {}).get(arm, 0.0)
            if budget_weight:
                prediction, budget = predict_budget(model, objective.batch, context)
                energy = budget_risk(budget, objective.batch)
            else:
                prediction = predict(model, objective, context)
                energy = prediction.new_zeros(())
            if arm in balanced:
                main = balanced_risk(objective, prediction)
                aux = consistency_loss(
                    prediction,
                    objective.batch,
                    phase_reference="clean",
                    normalization="global_band",
                )
            else:
                main, aux = objective(prediction)["total"], auxiliary(prediction, objective.batch)
            loss = main + config["weight"] * aux + budget_weight * energy
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training risk")
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
            if budget_weight or arm in balanced:
                history[-1].update(
                    budget=float(energy.detach()), optimized_total=float(loss.detach())
                )
        score_step = step % config.get("score_interval", 2048) == 0 or step in config["endpoints"]
        if step % 64 == 0 or score_step:
            state = {
                "step": step,
                "arm": arm,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "signature": signature,
                "initial_sha256": initial,
                "history": history,
            }
            if model_config:
                state["model_config"] = model_config
            atomic_save(state, directory / "latest.pth")
            if score_step:
                path = directory / f"step-{step}.pth"
                if not path.exists():
                    atomic_save(state, path)
                registered(
                    {"step": step, "checkpoint_sha256": sha256_file(path)},
                    directory / f"registration-{step}.json",
                )
                report = directory / f"training-{step}.json"
                if not report.exists():
                    measured = {
                        "checkpoint_sha256": sha256_file(path),
                        "common": evaluate(model, common, config["weight"]),
                    }
                    if step in config["endpoints"] and config.get("score_entire_bank", True):
                        measured["arm_bank"] = evaluate(model, pool, config["weight"])
                    write_json_report(measured, report)
                measured = read(report)
                if measured["checkpoint_sha256"] != sha256_file(path):
                    raise ValueError("training score checkpoint drift")
                progress(
                    root,
                    stage="training_score",
                    arm=arm,
                    seed=seed,
                    step=step,
                    end=config["steps"],
                    **measured["common"]["weighted"],
                )
            else:
                progress(
                    root, stage="training", arm=arm, seed=seed, end=config["steps"], **history[-1]
                )
    write_json_report(history, directory / "updates.json")
    complete(directory)


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, rows = prepare(args, root)
        if args.prepare_only:
            return
        inputs = [Path(p) for p in config["input_hashes"]]
        detector = next(p for p in inputs if p.name == "soundex-gain-data")
        binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
        onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
        parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
        clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
        progress(root, stage="prepare_training_bank", positions=len(config["bank"]))
        pool = Pool(config["bank"], clips)
        common = PoolView(pool, config["common"])
        for seed in config["seeds"]:
            for arm in config["arms"]:
                train(arm, seed, root, config, pool, common)
        del common, pool
        gc.collect()
        checkpoints = {
            f"{arm}-{step}-{seed}": {
                "arm": arm,
                "path": str(root / f"{arm}-{seed}/step-{step}.pth"),
                "sha256": sha256_file(root / f"{arm}-{seed}/step-{step}.pth"),
            }
            for arm in config["arms"]
            for seed in config["seeds"]
            for step in config["endpoints"]
        }
        registered(checkpoints, root / "review-registration.json")
        pools = {name: Pool(config[name], clips) for name in ("development", "shifted")}
        review("initial", None, None, root, pools, binary, onnx, config["weight"])
        for label, value in checkpoints.items():
            review(
                label,
                Path(value["path"]),
                value["arm"],
                root,
                pools,
                binary,
                onnx,
                config["weight"],
            )
        from evaluation.head_export import export_and_measure

        for label, value in checkpoints.items():
            progress(root, stage="export_and_measure", case=label)
            export_and_measure(Path(value["path"]), root / "exports" / label)
        for name, expected in config["source_hashes"].items():
            if sha256_file(Path(__file__).resolve().parents[1] / name) != expected:
                raise ValueError("final source drift")
        for seed, expected in config["schedule_hashes"].items():
            if digest(read(root / f"draws-{seed}.json")["draws"]) != expected:
                raise ValueError("final schedule drift")
        for name, expected in config["input_hashes"].items():
            if sha256_file(Path(name)) != expected:
                raise ValueError("final input drift")
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
        from scripts.analyze_head_study import run as analyze

        analyze(root)
        progress(root, stage="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p19", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
