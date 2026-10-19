#!/usr/bin/env python3
"""P29 training-only small-model fit: direct clean spectrum vs deployed supervision.

512 steps diagnose early learnability only, never convergence/capacity/generalization.
All endpoints are frozen before common scoring. No production model is replaced.
"""

import argparse
import fcntl
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, assert_isolated, protected_intervals
from data.causal_study import digest, load_sources
from evaluation.head_export import export_and_measure
from evaluation.head_study import auxiliary, review
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.complex_synthesis import complex_features
from models.direct_spectrum_objective import WEIGHTS, direct_spectrum_loss
from models.restoration_head import RestorationHead, load_restoration_head, predict_budget
from scripts.audit_context_deployment import run as audit_native
from scripts.audit_crop_coverage import coverage
from scripts.run_capacity_study import verify_inputs
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import complete, read, registered, verify_receipt
from scripts.run_head_study import progress
from scripts.run_music_coverage_study import tensor_digest

ARMS = ("deployed", "raw", "joint", "raw_unit_budget")
STEPS = 512
MODEL = dict(
    arm="complex", width=96, residual_slope=1 / 64, generation_budget=True, budget_scope="residual"
)


def make_model(seed, arm):
    torch.manual_seed(seed)
    model = RestorationHead(**MODEL)
    if arm == "raw_unit_budget":
        # Exact initial unit scale remains constant. Same inference graph/tensors.
        model.budget_head.requires_grad_(False)
    return model


def objective_terms(prediction, budget, objective, weight):
    old = objective(prediction)["total"] + weight * auxiliary(prediction, objective.batch)
    return {
        "deployed": old,
        "raw": direct_spectrum_loss(prediction, objective.batch)["total"],
        "budget": 0.1 * budget_risk(budget, objective.batch),
    }


def optimized_loss(arm, terms):
    if arm == "deployed":
        return terms["deployed"] + terms["budget"]
    if arm == "raw":
        return terms["raw"] + terms["budget"]
    if arm == "joint":
        return terms["raw"] + terms["deployed"] + terms["budget"]
    if arm == "raw_unit_budget":
        return terms["raw"]
    raise ValueError("unknown direct spectrum arm")


def register(args, root):
    path = root / "registration.json"
    if path.exists():
        config = read(path)
        if config["smoke"] != args.preflight:
            raise ValueError("preflight/full protocol drift")
        verify_inputs(config)
        return config
    parent_path = args.parent.resolve()
    parent = read(parent_path / "registration.json")
    if read(parent_path / "progress.json")["stage"] != "complete":
        raise ValueError("completed P28 parent required")
    paths = [Path(p) for p in parent["input_hashes"]]
    manifests = [p for p in paths if p.name == "manifest.jsonl"]
    rows, _ = load_sources(manifests)
    schedule_path = Path(parent["schedule_files"][f"broad-{parent['seeds'][0]}"])
    if sha256_file(schedule_path) != parent["input_hashes"][str(schedule_path)]:
        raise ValueError("parent schedule drift")
    schedule = read(schedule_path)
    draws = [0, 682, 1364, 2047]
    specs = [schedule["specs"][i] for d in draws for i in schedule["draws"][d]]
    assert_isolated(
        specs, rows, protected_intervals(parent["development"] + parent["shifted"], rows)
    )
    required = {
        "manifest.jsonl",
        "soundex-gain-data",
        "soundex-prediction-replay",
        "soundex-analysis-trace",
        "replay-contract.onnx",
    }
    inputs = {str(p): parent["input_hashes"][str(p)] for p in paths if p.name in required}
    for p in (
        parent_path / "registration.json",
        parent_path / "progress.json",
        schedule_path,
        args.native_binary.resolve(),
    ):
        inputs[str(p)] = sha256_file(p)
    for s in specs:
        row = rows[s["row_id"]]
        for role in ("clean", "degraded"):
            inputs[str(Path(row["_root"]) / row[f"{role}_path"])] = row[f"{role}_checksum"]
    source = Path(__file__).resolve().parents[1]
    config = {
        "format": "soundex-direct-spectrum-fit-v1",
        "scope": "fixed training crops only; early learnability diagnostic, not convergence, capacity or generalization; no promotion",
        "review_scope": "training-only fitted crops; all endpoints; NOT development or independent quality evaluation",
        "arms": list(ARMS),
        "seeds": parent["seeds"][:1] if args.preflight else parent["seeds"],
        "steps": 4 if args.preflight else STEPS,
        "primary_endpoint": 4 if args.preflight else STEPS,
        "smoke": args.preflight,
        "model_config": MODEL,
        "weight": parent["weight"],
        "rate": 1e-4,
        "draw_indices": draws,
        "specs": specs,
        # Legacy review API calls this slot development; the data are TRAINING only.
        "development": specs,
        "shifted": [],
        "cache_path": str(root / "feature-cache"),
        "direct_loss": {
            "version": 1,
            "weights": WEIGHTS,
            "burn": 89,
            "band_weights": [0.25, 1, 1, 1, 1],
            "relative_power_floor": 1e-4,
            "absolute_power_floor": 1e-14,
            "smooth_norm": 1e-3,
        },
        "optimization": "Adam 1e-4; no decay; grad norm clip 5; each step one fixed 4-music/1-speech batch plus identities; cycle four batches; same order and shared initial tensors for every arm; 80 no-grad warmup frames",
        "contrasts": "deployed vs raw replaces the reconstruction objective; raw vs joint adds deployed/consistency risk at weight 1; raw vs raw_unit_budget fixes budget at 1 and removes its auxiliary teacher. Not a full factorial or a pure gradient-path-only intervention.",
        "selection": "all final endpoints, both seeds; no best-checkpoint selection, no automatic extension or promotion; report common raw/deployed/native scores, never compare optimized totals across arms",
        "stop_policy": "nonfinite/integrity/runtime failure only; no early performance stopping",
        "input_hashes": inputs,
        "source_hashes": {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
            and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
        },
    }
    verify_inputs(config)
    registered(config, path)
    registered(
        {
            name: coverage([s for s in specs if rows[s["row_id"]]["corpus"] == name], rows)
            for name in ("music_library", "speech_library")
        },
        root / "coverage.json",
    )
    return config


def train(arm, seed, root, config, batches):
    directory = root / f"{arm}-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return
    model = make_model(seed, arm)
    initial = tensor_digest(model)
    registered({"initial_sha256": initial}, root / f"initialization-{seed}.json")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config["rate"]
    )
    signature = digest({"config": config, "arm": arm, "seed": seed})
    history, start = [], 0
    if (directory / "latest.pth").exists():
        state = torch.load(directory / "latest.pth", map_location="cpu", weights_only=False)
        if state["signature"] != signature or state["initial_sha256"] != initial:
            raise ValueError("resume signature drift")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["step"], state["history"]
        if len(history) != start or not 0 <= start <= config["steps"]:
            raise ValueError("invalid resume history")
    for step in range(start + 1, config["steps"] + 1):
        objective, context = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        prediction, budget = predict_budget(model, objective.batch, context)
        terms = objective_terms(prediction, budget, objective, config["weight"])
        loss = optimized_loss(arm, terms)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step,
                "optimized_total": float(loss.detach()),
                "gradient_norm": float(norm),
                **{k: float(v.detach()) for k, v in terms.items()},
            }
        )
        if step % 64 == 0 or step == config["steps"]:
            state = {
                "arm": arm,
                "step": step,
                "model_config": MODEL,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "signature": signature,
                "initial_sha256": initial,
            }
            atomic_save(state, directory / "latest.pth")
            progress(root, stage="training", arm=arm, seed=seed, end=config["steps"], **history[-1])
    atomic_save(state, directory / f"step-{config['steps']}.pth")
    write_json_report(history, directory / "updates.json")
    complete(directory)


@torch.no_grad()
def measure(prediction, objective):
    direct = direct_spectrum_loss(prediction, objective.batch)
    old = objective(prediction)
    b = objective.batch
    wave = objective.render(prediction)
    result = {}
    for role, selected in (("paired", ~b["identity"].bool()), ("identity", b["identity"].bool())):
        result[role] = {
            "raw_total": float(direct["per_example"][selected].mean()),
            "raw_components": {
                k: float(v[selected].mean()) for k, v in direct["components"].items()
            },
            "raw_bands": [
                {"hz": band["hz"], **{k: float(band[k][selected].mean()) for k in WEIGHTS}}
                for band in direct["bands"]
            ],
            "deployed_restoration": float(old["restoration"][selected].mean()),
            "deployed_preservation": float(old["preservation"][selected].mean()),
            "output_difference_rms": float(
                (wave - objective.baseline)[selected, objective.region].square().mean().sqrt()
            ),
            "raw_residual_rms": float(
                (complex_features(prediction) - complex_features(b["x"]))[selected, 89:]
                .abs()
                .square()
                .mean()
                .sqrt()
            ),
        }
    return result


def gradient_audit(model, objective, context, weight):
    prediction, budget = predict_budget(model, objective.batch, context)
    terms = objective_terms(prediction, budget, objective, weight)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    vectors, norms = {}, {}
    for name, loss in terms.items():
        gs = (
            torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
            if loss.requires_grad
            else [None] * len(params)
        )
        vector = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).flatten()
                for (_, p), g in zip(params, gs, strict=True)
            ]
        )
        vectors[name] = vector.detach()
        norms[name] = float(vector.norm())
    a, b = vectors["raw"], vectors["deployed"]
    denominator = a.norm() * b.norm()
    return {
        "weighted_losses": {k: float(v.detach()) for k, v in terms.items()},
        "gradient_norms": norms,
        "raw_deployed_cosine": float(torch.dot(a, b) / denominator) if denominator > 0 else None,
    }


@torch.no_grad()
def oracle_audit(batches):
    rows = []
    for objective, _ in batches:
        b = objective.batch
        original = complex_features(b["x"])
        target = complex_features(b["y"])
        records = []
        for amount in (0.0, 0.01, 0.1, 0.5, 1.0):
            z = original + amount * (target - original)
            features = torch.stack((20 * z.abs().clamp_min(1e-10).log10(), z.angle()), dim=1)
            features[:, :, :89] = b["x"][:, :, :89]
            if amount == 0:
                features = b["x"]
            records.append({"reference_fraction": amount, **measure(features, objective)})
        rows.append(records)
    return {
        "scope": "nondeployable clean-reference interpolation; fixed training crops only",
        "batches": rows,
    }


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
        detector = next(p for p in paths if p.name == "soundex-gain-data")
        clips = BoundedEnvelopeClips(rows, detector, Path(config["cache_path"]))
        pool = LazyPool(config["specs"], clips)
        batches = [pool.select(list(range(i, i + 5))) for i in range(0, len(pool.specs), 5)]
        write_json_report(oracle_audit(batches), root / "oracle.json")
        for seed in config["seeds"]:
            for arm in ARMS:
                train(arm, seed, root, config, batches)
        checkpoints = {}
        for seed in config["seeds"]:
            for arm in ARMS:
                path = root / f"{arm}-{seed}/step-{config['steps']}.pth"
                checkpoints[f"{arm}-{config['steps']}-{seed}"] = {
                    "arm": arm,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
        registered(checkpoints, root / "review-registration.json")
        progress(root, stage="frozen_endpoint_review")
        binary = next(p for p in paths if p.name == "soundex-prediction-replay")
        observer = next(p for p in paths if p.name == "soundex-analysis-trace")
        onnx = next(p for p in paths if p.name == "replay-contract.onnx")
        for label, value in {"initial": None, **checkpoints}.items():
            checkpoint = Path(value["path"]) if value else None
            model = (
                load_restoration_head(
                    torch.load(checkpoint, map_location="cpu", weights_only=False)
                ).eval()
                if value
                else None
            )
            if value and value["arm"] == "raw_unit_budget":
                model.budget_head.requires_grad_(False)
            direct = []
            for objective, context in batches:
                with torch.no_grad():
                    prediction = (
                        predict_budget(model, objective.batch, context)[0]
                        if model
                        else objective.batch["x"]
                    )
                    direct.append(measure(prediction, objective))
            write_json_report(
                {
                    "scope": config["scope"],
                    "checkpoint_sha256": value["sha256"] if value else None,
                    "batches": direct,
                    "gradients": [
                        gradient_audit(model, *batch, config["weight"]) for batch in batches
                    ]
                    if model
                    else None,
                },
                root / "direct-review" / f"{label}.json",
            )
            pools = {"development": pool}
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
            if value:
                export_and_measure(checkpoint, root / "exports" / label, iterations=100)
        verify_inputs(config)
        registered(
            {
                "status": "complete",
                "scope": "training-only fit and replay; full frontend audit separately",
                "registration_sha256": sha256_file(root / "registration.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )
        progress(root, stage="full_native_training_crop_audit")
        audit_native(
            SimpleNamespace(study=root, output=root / "native-audit", binary=args.native_binary)
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
