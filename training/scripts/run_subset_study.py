#!/usr/bin/env python3
"""P12 subset scaling / P13 crop coverage at fixed milestones, without dev selection."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from data.causal_study import digest, load_sources
from data.crop_study import crop_bank, crop_draws, protected_probes
from data.envelope_study import EnvelopeClips
from data.subset_study import make_draws, native_development_specs, nested_specs, shifted_specs
from data.weak_band_study import audit_groups, identity_specs
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_probe import context_features, predict_probe
from models.fit_diagnostic import BURN, FixedFitObjective
from models.rendered_objective import amplitude_features
from models.rendered_probe import make_rendered_model
from scripts.run_causal_study import atomic_save
from scripts.run_fit_diagnostic import metrics, replay
from scripts.run_weak_band_study import freeze_binary


class Pool:
    def __init__(self, specs, clips):
        self.specs = specs
        self.batch = clips.batch(specs + identity_specs(specs))
        self.context = context_features(self.batch["pcm"], "past")
        self.objective = FixedFitObjective(self.batch)
        self.rows = [clips.rows[s["row_id"]] for s in specs]

    def select(self, paired_indices):
        indices = paired_indices + [i + len(self.specs) for i in paired_indices]
        return self.objective.select(indices), self.context[indices]


def features(model, objective, context):
    if model is None:
        return objective.batch["x"]
    return amplitude_features(
        predict_probe(model, objective.batch, context, burn=BURN), objective.batch["x"]
    )


@torch.no_grad()
def evaluate(model, pool):
    # One pair at a time keeps memory bounded and sample clocks independent.
    records = []
    for i, row in enumerate(pool.rows):
        objective, context = pool.select([i])
        score = metrics(objective(features(model, objective, context)), objective.batch)
        records.append(
            {
                "spec": pool.specs[i],
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "genre": row.get("source_metadata", {}).get("genre"),
                "metrics": score,
            }
        )
    return {**aggregate(records), "rows": records}


def aggregate(records):
    if not records:
        return None
    corpus = {}
    for name in ("music_library", "speech_library"):
        group = [r["metrics"] for r in records if r["corpus"] == name]
        if not group:
            raise ValueError("both corpora required for common score")
        corpus[name] = {
            k: float(np.mean([r[k] for r in group]))
            for k in ("total", "paired_restoration", "identity_preservation")
        }
    common = {
        k: 0.8 * corpus["music_library"][k] + 0.2 * corpus["speech_library"][k]
        for k in corpus["music_library"]
    }
    return {"weighted": common, "corpus": corpus}


@torch.no_grad()
def native(model, pool, indices, binary, onnx):
    objective, context = pool.select(indices)
    return replay(features(model, objective, context), objective.batch, objective, binary, onnx)


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads(args.parent.read_text())
        rows, manifests = load_sources([Path(p) for p in parent["manifests"]])
        if manifests != parent["manifests"]:
            raise ValueError("manifest drift")
        pools = nested_specs(parent["training_probe_specs"], rows)
        shifted = (
            protected_probes(pools[40], rows) if args.crop_study else shifted_specs(pools[40], rows)
        )
        bank, crop_audit = (
            crop_bank(pools[40], shifted, rows, args.crop_positions)
            if args.crop_study
            else (pools[40], [])
        )
        development = parent["validation_specs"]
        development_native = native_development_specs(development, rows)
        source = Path(__file__).resolve().parents[1]
        hashes = {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*.py")
            if not any(
                x in {".venv", "tests", "outputs", "__pycache__"}
                for x in p.relative_to(source).parts
            )
        }
        config = {
            "format": "soundex-subset-study-v2",
            "crop_sampling": {
                "positions_per_recording": args.crop_positions,
                "bank": bank,
                "audit": crop_audit,
                "policy": "fixed anchor or balanced cycling of a finite protected crop bank; gain unchanged",
            },
            "sizes": args.sizes,
            "seeds": args.seeds,
            "common_steps": args.common_steps,
            "steps_by_size": {str(n): args.common_steps * n // 5 for n in args.sizes},
            "learning_rate": 0.001,
            "optimizer": "Adam, no decay, clip_grad_norm 5",
            "training_risk": "0.5 paired + 0.5 identity; fixed 4 music + 1 speech pairs per update",
            "report_risk": "0.8 music + 0.2 speech, within-corpus mean; same per-example P11 objective",
            "selection": "none: report step zero, fixed common-update and fixed equal-exposure checkpoints; never choose on development",
            "train_pools": pools,
            "shifted_specs": shifted,
            "development_specs": development,
            "native_development_specs": development_native,
            "manifests": manifests,
            "coverage": audit_groups(rows),
            "parent_sha256": sha256_file(args.parent),
            "source_sha256": digest(hashes),
            "detector_sha256": parent["detector_sha256"],
            "replay_sha256": sha256_file(args.replay_binary),
            "onnx_sha256": sha256_file(args.onnx),
            "scope": "diagnostic subset scaling; same-recording new scoring regions and reused development, not independent test; no promotion",
        }
        # JSON object keys are strings on reload.
        config = json.loads(json.dumps(config))
        path = root / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("registered study changed")
        write_json_report(config, path)
        write_json_report(hashes, root / "source-files.json")
        detector = freeze_binary(
            args.parent.parent / "binaries/soundex-gain-data",
            root / "binaries/soundex-gain-data",
            parent["detector_sha256"],
        )
        binary = freeze_binary(
            args.replay_binary, root / "binaries/soundex-prediction-replay", config["replay_sha256"]
        )
        onnx = root / "replay-contract.onnx"
        if not onnx.exists():
            onnx.write_bytes(args.onnx.read_bytes())
        if sha256_file(onnx) != config["onnx_sha256"]:
            raise ValueError("ONNX drift")
        clips = EnvelopeClips(rows, detector, Path(parent["cache_path"]))
        prepared = {
            "anchor": Pool(pools[40], clips),
            "shifted": Pool(shifted, clips),
            "development": Pool(development, clips),
        }
        training_pool = prepared["anchor"] if args.crop_positions == 1 else Pool(bank, clips)
        common_indices = [pools[40].index(s) for s in pools[5]]
        native_indices = {
            "anchor": common_indices,
            "shifted": common_indices,
            "development": [development.index(s) for s in development_native],
        }
        baseline_path = root / "baseline.json"
        if not baseline_path.exists():
            write_json_report(
                {
                    name: {
                        "proxy": evaluate(None, pool),
                        "native": native(None, pool, native_indices[name], binary, onnx),
                    }
                    for name, pool in prepared.items()
                },
                baseline_path,
            )
        completed = []
        for seed in args.seeds:
            for size in args.sizes:
                name = f"size-{size}-{seed}"
                directory = root / name
                directory.mkdir(exist_ok=True)
                steps = args.common_steps * size // 5
                draws = make_draws(size, steps, seed)
                sampled = crop_draws(draws, args.crop_positions, seed, size)
                signature = digest(
                    {
                        "config": config,
                        "seed": seed,
                        "size": size,
                        "draws": draws,
                        "crop_draws": sampled,
                    }
                )
                write_json_report(
                    {
                        "draws": draws,
                        "crop_draws": sampled,
                        "crop_exposures_final": np.bincount(
                            np.array(sampled).reshape(-1), minlength=size * args.crop_positions
                        ).tolist(),
                        "sha256": digest(draws),
                        "exposures_final": np.bincount(
                            np.array(draws).reshape(-1), minlength=size
                        ).tolist(),
                        "exposures_common": np.bincount(
                            np.array(draws[: args.common_steps]).reshape(-1), minlength=size
                        ).tolist(),
                    },
                    directory / "draws.json",
                )
                receipt_path = directory / "complete.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt["signature"] != signature or any(
                        sha256_file(directory / file) != sha
                        for file, sha in receipt["hashes"].items()
                    ):
                        raise ValueError("completed run changed")
                    completed.append(receipt)
                    continue
                torch.manual_seed(seed)
                model = make_rendered_model("rendered_balanced")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                start = 0
                if (directory / "latest.pth").exists():
                    state = torch.load(
                        directory / "latest.pth", map_location="cpu", weights_only=False
                    )
                    if state["signature"] != signature:
                        raise ValueError("resume changed")
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    torch.set_rng_state(state["rng"])
                    start = state["step"]
                anchor_indices = [pools[40].index(s) for s in pools[size]]
                milestones = sorted({args.common_steps, steps})

                def save(
                    step,
                    model=model,
                    optimizer=optimizer,
                    signature=signature,
                    milestones=milestones,
                    directory=directory,
                ):
                    value = {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                        "signature": signature,
                    }
                    if step in milestones:
                        atomic_save(value, directory / f"step-{step}.pth")
                    atomic_save(value, directory / "latest.pth")

                for step in range(start + 1, steps + 1):
                    chosen = [
                        anchor_indices[i // args.crop_positions] * args.crop_positions
                        + i % args.crop_positions
                        for i in sampled[step - 1]
                    ]
                    objective, context = training_pool.select(chosen)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(features(model, objective, context))["total"]
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    if step % 64 == 0 or step in milestones:
                        save(step)
                        progress = {
                            "run": name,
                            "step": step,
                            "steps": steps,
                            "preupdate_batch_loss": float(loss.detach()),
                        }
                        write_json_report(progress, root / "progress.json")
                        print(json.dumps(progress), flush=True)
                # Freeze both checkpoint hashes before scoring any held-out rows.
                registration = {
                    f"step-{step}.pth": sha256_file(directory / f"step-{step}.pth")
                    for step in milestones
                }
                write_json_report(registration, directory / "evaluation-registration.json")
                for step in milestones:
                    report_path = directory / f"evaluation-{step}.json"
                    if report_path.exists():
                        continue
                    state = torch.load(
                        directory / f"step-{step}.pth", map_location="cpu", weights_only=False
                    )
                    model.load_state_dict(state["model"])
                    report = {
                        "step": step,
                        "checkpoint_sha256": registration[f"step-{step}.pth"],
                        "groups": {},
                        "training_bank_proxy": evaluate(model, training_pool),
                    }
                    for group, pool in prepared.items():
                        result = evaluate(model, pool)
                        if group in ("anchor", "shifted"):
                            result["fitted_track_ids"] = [
                                rows[s["row_id"]]["track_id"] for s in pools[size]
                            ]
                            fitted = set(result["fitted_track_ids"])
                            result["fitted_tracks"] = aggregate(
                                [r for r in result["rows"] if r["track_id"] in fitted]
                            )
                            result["other_train_tracks"] = aggregate(
                                [r for r in result["rows"] if r["track_id"] not in fitted]
                            )
                        report["groups"][group] = {
                            "proxy": result,
                            "native": native(model, pool, native_indices[group], binary, onnx),
                        }
                    write_json_report(report, report_path)
                hashes_out = {
                    file: sha256_file(directory / file)
                    for file in [
                        *(f"step-{s}.pth" for s in milestones),
                        *(f"evaluation-{s}.json" for s in milestones),
                        "draws.json",
                        "evaluation-registration.json",
                    ]
                }
                receipt = {
                    "size": size,
                    "seed": seed,
                    "steps": steps,
                    "signature": signature,
                    "hashes": hashes_out,
                }
                write_json_report(receipt, receipt_path)
                completed.append(receipt)
        write_json_report(
            {"status": "complete", "scope": config["scope"], "runs": completed},
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--crop-study", action="store_true")
    parser.add_argument("--crop-positions", type=int, choices=(1, 8), default=1)
    parser.add_argument("--common-steps", type=int, default=256)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261012, 20261013])
    args = parser.parse_args()
    if (
        (args.crop_study and args.sizes != [40])
        or (args.crop_positions > 1 and not args.crop_study)
        or args.common_steps < 1
        or any(n not in (5, 10, 40) for n in args.sizes)
        or len(set(args.sizes)) != len(args.sizes)
    ):
        parser.error("invalid steps or sizes")
    run(args)
