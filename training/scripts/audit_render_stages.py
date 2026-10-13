#!/usr/bin/env python3
"""Locate P14 error across blending, OLA, gating and limiting without altering inference."""

import argparse
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.loss_scale import spectral_parts, summarize_parts
from evaluation.reporting import sha256_file, write_json_report
from models.causal_objective import render
from models.losses import blend_deployment_features
from models.rendered_objective import amplitude_features, reanalyze
from models.rendered_probe import make_rendered_model
from scripts.run_subset_study import Pool, features


def stages(prediction, batch):
    output, prelimit = render(
        prediction,
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )
    blended = blend_deployment_features(
        batch["x"], prediction, batch["mask"] * batch["active"][:, None, :, None]
    )
    gated = batch["dry"] * (1 - batch["gate"]) + prelimit * batch["gate"]
    result = {
        "raw": torch.pow(10.0, prediction[:, 0] / 20),
        "blended": torch.pow(10.0, blended[:, 0] / 20),
        "ola": reanalyze(prelimit),
        "gated": reanalyze(gated),
        "limited": reanalyze(output),
    }
    # Reanalysis drops final unsupported frames; compare identical frame support.
    frames = min(value.shape[1] for value in result.values())
    return {name: value[:, :frames] for name, value in result.items()}, prelimit


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    config = json.loads((args.study / "config.json").read_text())
    parent = json.loads(args.parent.read_text())
    if sha256_file(args.parent) != config["parent_sha256"]:
        raise ValueError("parent drift")
    rows, manifests = load_sources([Path(p) for p in config["manifests"]])
    if manifests != config["manifests"]:
        raise ValueError("manifest drift")
    detector = args.study / "binaries/soundex-gain-data"
    if sha256_file(detector) != config["detector_sha256"]:
        raise ValueError("detector drift")
    receipt = json.loads((args.audit / "summary.json").read_text())
    for name, sha in receipt["reports"].items():
        if sha256_file(args.audit / name) != sha:
            raise ValueError("audit drift")
    registered = json.loads((args.audit / "registration.json").read_text())
    clips = EnvelopeClips(rows, detector, Path(parent["cache_path"]))
    results = []
    for group, specs in (
        ("training_bank", config["crop_sampling"]["bank"]),
        ("development", config["development_specs"]),
    ):
        pool = Pool(specs, clips)
        for label, sha in registered["checkpoints"].items():
            seed = label.split("-")[-1]
            checkpoint = args.study / f"size-40-{seed}" / "step-2048.pth"
            if sha256_file(checkpoint) != sha:
                raise ValueError("checkpoint drift")
            model = make_rendered_model("rendered_balanced").eval()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            )
            for i, row in enumerate(pool.rows):
                objective, context = pool.select([i])
                b = objective.batch
                initial = amplitude_features(torch.pow(10.0, b["x"][:, 0] / 20), b["x"])
                baseline, baseline_wave = stages(initial, b)
                candidate, wave = stages(features(model, objective, context), b)
                for index, role in enumerate(("paired", "identity")):
                    record = {
                        "group": group,
                        "model": label,
                        "spec": specs[i],
                        "corpus": row["corpus"],
                        "track_id": row["track_id"],
                        "role": role,
                        "stages": {},
                        "initial_stages": {},
                        "prelimit_changed_max": float(
                            (wave[index, objective.region] - baseline_wave[index, objective.region])
                            .abs()
                            .max()
                        ),
                        "prelimit_peak": float(wave[index, objective.region].abs().max()),
                        "baseline_prelimit_peak": float(
                            baseline_wave[index, objective.region].abs().max()
                        ),
                        "gate_min": float(b["gate"][index, objective.region].min()),
                    }
                    for name, spectrum in candidate.items():
                        reference = (
                            objective.target[index, : spectrum.shape[1]]
                            if index == 0
                            else baseline[name][index]
                        )
                        record["stages"][name] = summarize_parts(
                            spectral_parts(spectrum[index], reference, float(b["rate"][index, 0]))
                        )
                        record["initial_stages"][name] = summarize_parts(
                            spectral_parts(
                                baseline[name][index], reference, float(b["rate"][index, 0])
                            )
                        )
                    results.append(record)
            print(json.dumps({"group": group, "model": label, "pairs": len(specs)}), flush=True)
    write_json_report(
        {
            "scope": "post-hoc stage isolation, no runtime changes; identity uses same-stage initial reference",
            "audit_receipt_sha256": sha256_file(args.audit / "summary.json"),
            "source_files": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
                for p in [
                    Path(__file__),
                    Path(__file__).resolve().parents[1] / "evaluation/loss_scale.py",
                ]
            },
            "checkpoints": registered["checkpoints"],
            "records": results,
        },
        args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
