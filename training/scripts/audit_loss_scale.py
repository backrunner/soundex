#!/usr/bin/env python3
"""P14 read-only P13 loss, coordinate-gradient and parameter-gradient audit."""

import argparse
import fcntl
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.loss_scale import gradient_comparison, spectral_parts, summarize_parts
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_probe import predict_probe
from models.fit_diagnostic import BURN
from models.rendered_objective import amplitude_features, reanalyze, waveform_norm
from models.rendered_probe import make_rendered_model
from scripts.run_subset_study import Pool, native


def inspect(model, objective, context):
    batch = objective.batch
    prediction = predict_probe(model, batch, context, burn=BURN)
    features = amplitude_features(prediction, batch["x"])
    output = objective.render(features)
    rendered = reanalyze(output)
    roles, losses = [], []
    for i, role in enumerate(("paired", "identity")):
        rate = float(batch["rate"][i, 0])
        reference = (
            objective.target[i, : rendered.shape[1]] if i == 0 else objective.baseline_amplitude[i]
        )
        wave_reference = batch["clean_wave"][i] if i == 0 else objective.baseline[i]
        parts = spectral_parts(rendered[i], reference, rate)
        wave_loss = waveform_norm(
            output[i : i + 1, objective.region], wave_reference[None, objective.region]
        )[0]
        loss = sum(x["total"] for x in parts) + 0.25 * wave_loss
        losses.append(loss)
        # Independent rendered-amplitude coordinate: exposes denominator scaling.
        coordinate = rendered[i].detach().requires_grad_()
        records = summarize_parts(spectral_parts(coordinate, reference, rate), coordinate)
        raw_reference = objective.target[i] if i == 0 else torch.pow(10.0, batch["x"][i, 0] / 20)
        raw = summarize_parts(spectral_parts(prediction[i].detach(), raw_reference, rate))
        roles.append(
            {
                "role": role,
                "loss": float(loss.detach()),
                "waveform_loss": float(wave_loss.detach()),
                "rendered": records,
                "raw": raw,
            }
        )
    with torch.no_grad():
        expected = objective(features)["total"]
    torch.testing.assert_close(0.5 * sum(losses), expected, rtol=2e-5, atol=2e-6)
    parameters = tuple(model.parameters())
    grads = [
        torch.autograd.grad(loss, (prediction, *parameters), retain_graph=i == 0)
        for i, loss in enumerate(losses)
    ]
    if any(not torch.isfinite(g).all() for group in grads for g in group):
        raise ValueError("nonfinite audit gradient")
    scale = torch.pow(10.0, batch["x"][:, 0] / 20).square().mean(-1, keepdim=True).sqrt()
    for i, role in enumerate(roles):
        role["raw_amplitude_gradient_l2"] = float(grads[i][0][i].norm())
        role["input_frame_relative_gradient_l2"] = float((grads[i][0][i] * scale[i]).norm())
    return {"roles": roles, "parameter_gradients": gradient_comparison(grads[0][1:], grads[1][1:])}


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "audit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = args.study.resolve()
        config = json.loads((study / "config.json").read_text())
        parent = json.loads(args.parent.read_text())
        if sha256_file(args.parent) != config["parent_sha256"]:
            raise ValueError("parent config drift")
        summary = json.loads((study / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError("P13 must be complete")
        checkpoints = {}
        for receipt in summary["runs"]:
            directory = study / f"size-{receipt['size']}-{receipt['seed']}"
            for name, sha in receipt["hashes"].items():
                if sha256_file(directory / name) != sha:
                    raise ValueError("P13 completed artifact changed")
            checkpoints[f"final-{receipt['seed']}"] = directory / f"step-{receipt['steps']}.pth"
        rows, manifests = load_sources([Path(p) for p in config["manifests"]])
        if manifests != config["manifests"]:
            raise ValueError("manifest drift")
        detector = study / "binaries/soundex-gain-data"
        binary, onnx = study / "binaries/soundex-prediction-replay", study / "replay-contract.onnx"
        for path, key in (
            (detector, "detector_sha256"),
            (binary, "replay_sha256"),
            (onnx, "onnx_sha256"),
        ):
            if sha256_file(path) != config[key]:
                raise ValueError("native binary/contract drift")
        # Explicit post-hoc diagnostic selection, never a new acceptance subset.
        worst = set()
        for checkpoint in checkpoints.values():
            evaluation = json.loads(
                (checkpoint.parent / f"evaluation-{summary['runs'][0]['steps']}.json").read_text()
            )
            development = evaluation["groups"]["development"]["proxy"]["rows"]
            music = [r for r in development if r["corpus"] == "music_library"]
            worst.update(
                r["spec"]["row_id"]
                for r in sorted(music, key=lambda r: r["metrics"]["total"], reverse=True)[:3]
            )
        source = Path(__file__).resolve().parents[1]
        source_files = {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*.py")
            if not any(
                x in {".venv", "tests", "outputs", "__pycache__"}
                for x in p.relative_to(source).parts
            )
        }
        registration = {
            "scope": "post-hoc diagnostics; no training, no selection or promotion",
            "study_sha256": sha256_file(study / "config.json"),
            "checkpoints": {k: sha256_file(p) for k, p in checkpoints.items()},
            "native_row_ids": sorted(worst),
            "source_files": source_files,
            "groups": {
                "training_bank": len(config["crop_sampling"]["bank"]),
                "development": len(config["development_specs"]),
            },
        }
        path = root / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("audit registration changed")
        write_json_report(registration, path)
        clips = EnvelopeClips(rows, detector, Path(parent["cache_path"]))
        for group, specs in (
            ("training_bank", config["crop_sampling"]["bank"]),
            ("development", config["development_specs"]),
        ):
            pool = Pool(specs, clips)
            for label, checkpoint in {"initial": None, **checkpoints}.items():
                torch.manual_seed(20261012)
                model = make_rendered_model("rendered_balanced").eval()
                if checkpoint is not None:
                    model.load_state_dict(
                        torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
                    )
                records = []
                for i, row in enumerate(pool.rows):
                    objective, context = pool.select([i])
                    record = inspect(model, objective, context)
                    records.append(
                        {
                            "spec": specs[i],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "genre": row.get("source_metadata", {}).get("genre"),
                            **record,
                        }
                    )
                    if (i + 1) % 40 == 0:
                        progress = {
                            "group": group,
                            "model": label,
                            "cases": i + 1,
                            "total": len(specs),
                        }
                        write_json_report(progress, root / "progress.json")
                        print(json.dumps(progress), flush=True)
                report = {"group": group, "model": label, "records": records}
                if group == "development":
                    indices = [i for i, s in enumerate(specs) if s["row_id"] in worst]
                    report["native"] = native(model, pool, indices, binary, onnx)
                    report["native_specs"] = [specs[i] for i in indices]
                write_json_report(report, root / f"{group}-{label}.json")
        write_json_report(
            {
                "status": "complete",
                "reports": {
                    p.name: sha256_file(p)
                    for p in root.glob("*.json")
                    if p.name not in {"summary.json", "progress.json"}
                },
            },
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
