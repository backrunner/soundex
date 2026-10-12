#!/usr/bin/env python3
"""P11 preregistered small-set fit: free amplitudes, complex spectra, shared network."""

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.causal_study import digest, load_sources
from data.envelope_study import EnvelopeClips
from data.weak_band_study import identity_specs
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.envelope_probe import context_features, predict_probe
from models.fit_diagnostic import BURN, FixedFitObjective, FreeSpectrum
from models.rendered_objective import amplitude_features
from models.rendered_probe import make_rendered_model
from scripts.replay_factorized_study import aligned_band_metrics
from scripts.run_causal_study import atomic_save
from scripts.run_weak_band_study import freeze_binary


def select_specs(specs, rows):
    selected, genres = [], set()
    for spec in sorted(specs, key=lambda s: s["row_id"]):
        row = rows[spec["row_id"]]
        if row["split"] != "train" or row["held_out"]:
            raise ValueError("training-only probe required")
        genre = row.get("source_metadata", {}).get("genre", "unlabeled")
        if row["corpus"] == "music_library" and genre not in genres and len(selected) < 4:
            selected.append(spec)
            genres.add(genre)
    if len(selected) != 4:
        raise ValueError("four distinct music genres required")
    selected.append(
        next(
            s
            for s in sorted(specs, key=lambda s: s["row_id"])
            if rows[s["row_id"]]["corpus"] == "speech_library"
        )
    )
    return selected


def metrics(result, batch):
    identity = batch["identity"].bool()
    return {
        "total": float(result["total"].detach()),
        "paired_restoration": float(result["restoration"][~identity].mean().detach()),
        "identity_preservation": float(result["preservation"][identity].mean().detach()),
        "rows": [
            {k: float(v[i].detach()) for k, v in result.items() if k != "total"}
            for i in range(len(identity))
        ],
    }


def replay(features, batch, objective, binary, onnx):
    records = []
    with torch.no_grad(), tempfile.TemporaryDirectory(prefix="soundex-p11-") as tmp:
        work = Path(tmp)
        proxy = objective.render(features)
        for i in range(len(features)):
            rate = int(batch["rate"][i, 0])
            write_sxa(work / "input.sxa", batch["pcm"][i].numpy(), rate)
            (work / "features.json").write_text(
                json.dumps(
                    features[i : i + 1].permute(2, 0, 1, 3).reshape(-1).tolist(), allow_nan=False
                )
            )
            subprocess.run(
                [
                    str(binary),
                    str(onnx),
                    str(work / "input.sxa"),
                    str(work / "features.json"),
                    str(work / "output.sxa"),
                ],
                check=True,
                capture_output=True,
                env={**os.environ, "SOUNDEX_EVAL_LIMITER_MODE": "smooth-knee"},
            )
            audio, actual_rate = read_sxa(work / "output.sxa")
            if audio.shape != (batch["pcm"].shape[-1], 1) or actual_rate != rate:
                raise ValueError("native output shape/rate mismatch")
            error = float(
                np.max(np.abs(audio[objective.region, 0] - proxy[i, objective.region].numpy()))
            )
            if not np.isfinite(error) or error > 2e-5:
                raise ValueError(f"native parity failure: {error}")
            raw, rendered = aligned_band_metrics(
                features[i : i + 1],
                batch["y"][i : i + 1],
                torch.from_numpy(audio[:, 0])[None],
                batch["rate"][i : i + 1],
            )
            records.append(
                {"proxy_max_error": error, "raw_band_metrics": raw, "output_band_metrics": rendered}
            )
    return records


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads(args.parent.read_text())
        rows, manifests = load_sources([Path(p) for p in parent["manifests"]])
        if manifests != parent["manifests"]:
            raise ValueError("source manifests changed")
        specs = select_specs(parent["training_probe_specs"], rows)
        specs += identity_specs(specs)
        source = Path(__file__).resolve().parents[1]
        hashes = {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*.py")
            if not any(
                x in {".venv", "tests", "__pycache__", "outputs"}
                for x in p.relative_to(source).parts
            )
        }
        runs = [
            {"arm": arm, "lr": lr, "seed": 0}
            for arm in ("amplitude", "complex")
            for lr in (0.0001, 0.001)
        ]
        runs += [
            {"arm": "network", "lr": lr, "seed": seed}
            for seed in (20261012, 20261013)
            for lr in (0.0001, 0.001)
        ]
        config = {
            "format": "soundex-fixed-fit-v1",
            "steps": args.steps,
            "runs": runs,
            "specs": specs,
            "scope": "target-fitted training-only diagnostic; no held-out quality or deployable free spectra",
            "selection": "minimum fixed-fit objective among step 0 and every 64 steps, including final",
            "risk": "equal paired/identity means; equal clip weights, no corpus reweighting",
            "optimizer": "Adam; fixed learning rate; no decay; clip_grad_norm 5; projected nonnegative amplitude",
            "warmup": BURN,
            "scored_start_hop": 89,
            "limiter": "smooth-knee",
            "manifests": manifests,
            "parent_sha256": sha256_file(args.parent),
            "source_sha256": digest(hashes),
            "detector_sha256": parent["detector_sha256"],
            "replay_sha256": sha256_file(args.replay_binary),
            "onnx_sha256": sha256_file(args.onnx),
            "metadata": [
                {k: clips_row.get(k) for k in ("track_id", "corpus", "codec_id", "source_metadata")}
                for clips_row in (rows[s["row_id"].removesuffix(":identity")] for s in specs)
            ],
        }
        path = root / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("registered experiment changed")
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
            raise ValueError("ONNX contract changed")
        clips = EnvelopeClips(rows, detector, Path(parent["cache_path"]))
        batch = clips.batch(specs)
        context = context_features(batch["pcm"], "past")
        objective = FixedFitObjective(batch)
        baseline = {
            "metrics": metrics(objective(batch["x"]), batch),
            "native": replay(batch["x"], batch, objective, binary, onnx),
        }
        write_json_report(baseline, root / "baseline.json")
        completed = []
        for entry in runs:
            arm, lr, seed = entry["arm"], entry["lr"], entry["seed"]
            name = f"{arm}-{lr:g}-{seed}"
            directory = root / name
            directory.mkdir(exist_ok=True)
            signature = digest({"config": config, "run": entry})
            receipt_path = directory / "complete.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (
                    receipt["signature"] != signature
                    or sha256_file(directory / "best.pth") != receipt["checkpoint_sha256"]
                ):
                    raise ValueError("completed run changed")
                completed.append(receipt)
                continue
            torch.manual_seed(seed)
            model = (
                make_rendered_model("rendered_balanced")
                if arm == "network"
                else FreeSpectrum(batch["x"], arm)
            )

            def prediction(model=model, arm=arm):
                return (
                    amplitude_features(predict_probe(model, batch, context, burn=BURN), batch["x"])
                    if arm == "network"
                    else model()
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best = float("inf")
            history = []
            for step in range(args.steps + 1):
                if step > 0:
                    optimizer.zero_grad(set_to_none=True)
                    result = objective(prediction())
                    if not torch.isfinite(result["total"]):
                        raise ValueError("nonfinite fit loss")
                    result["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    if arm == "amplitude":
                        with torch.no_grad():
                            model.delta.clamp_(
                                min=(1e-10 - model.magnitude[:, BURN:]) / model.scale[:, BURN:]
                            )
                if step % 64 == 0 or step == args.steps:
                    with torch.no_grad():
                        features = prediction()
                        measured = metrics(objective(features), batch)
                    history.append({"step": step, **measured})
                    if measured["total"] < best:
                        best = measured["total"]
                        atomic_save(
                            {
                                "step": step,
                                "features": features,
                                "model": model.state_dict(),
                                "signature": signature,
                            },
                            directory / "best.pth",
                        )
                    write_json_report(
                        {"run": name, "step": step, "steps": args.steps, "metrics": measured},
                        root / "progress.json",
                    )
                    print(
                        json.dumps(
                            {
                                "run": name,
                                "step": step,
                                "total": measured["total"],
                                "paired": measured["paired_restoration"],
                                "identity": measured["identity_preservation"],
                            }
                        ),
                        flush=True,
                    )
            write_json_report(history, directory / "history.json")
            checkpoint = torch.load(directory / "best.pth", map_location="cpu", weights_only=False)
            write_json_report(
                replay(checkpoint["features"], batch, objective, binary, onnx),
                directory / "native.json",
            )
            receipt = {
                **entry,
                "best_step": checkpoint["step"],
                "best_score": best,
                "signature": signature,
                "checkpoint_sha256": sha256_file(directory / "best.pth"),
                "native_sha256": sha256_file(directory / "native.json"),
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
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args)
