#!/usr/bin/env python3
"""P9 post-hoc diagnosis of applied/discarded corrections and identity conflict."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from data.weak_band_study import identity_specs
from evaluation.reporting import sha256_file, write_json_report
from models.causal_objective import render
from models.conditioned_shape import make_shape_model
from models.envelope_objective import BANDS
from models.envelope_probe import context_features, predict_probe
from models.factorized_envelope import ABSOLUTE_FLOOR, BURN, RELATIVE_FLOOR
from models.linear_objective import linear_objective
from scripts.replay_factorized_study import hybrid_features
from train import spectral_features


def spatial_error_delta(pred, dry, target, rate, effective):
    p, d, t, m = [v[:, BURN:] for v in (pred, dry, target, effective)]
    floor = (t.square().mean((1, 2), keepdim=True).sqrt() * RELATIVE_FLOOR).clamp_min(
        ABSOLUTE_FLOOR
    )
    frequency = torch.arange(129)[None, :] * rate / 256
    terms = {k: 0.0 for k in ("applied_raw", "discarded_raw", "blended")}
    for low, high in BANDS:
        mask = ((frequency >= low) & (frequency < high))[:, None, :]
        count = mask.sum() * t.shape[1]
        denom = (t.square() * mask).sum() + floor.square().squeeze() * count
        delta = (p - t).square() - (d - t).square()
        weight = (0.25 if low == 0 else 1) / 4.25
        terms["applied_raw"] += float((delta * m * mask).sum() / denom) * weight
        terms["discarded_raw"] += float((delta * (1 - m) * mask).sum() / denom) * weight
        blended = d + m * (p - d)
        terms["blended"] += (
            float((((blended - t).square() - (d - t).square()) * mask).sum() / denom) * weight
        )
    return terms


def loss_and_gradient(model, clips, spec):
    batch = clips.batch([spec])
    context = context_features(batch["pcm"], "past")
    p = predict_probe(model, batch, context, burn=BURN)
    target = torch.pow(10.0, batch["y"][:, 0] / 20)
    loss = linear_objective(p, target, batch["rate"])["total"]
    grads = torch.autograd.grad(loss, model.parameters())
    return float(loss.detach()), torch.cat([v.flatten() for v in grads])


@torch.no_grad()
def inspect(model, clips, spec):
    batch = clips.batch([spec])
    p = predict_probe(model, batch, context_features(batch["pcm"], "past"), burn=BURN)
    d, t = [torch.pow(10.0, batch[k][:, 0] / 20) for k in ("x", "y")]
    effective = batch["mask"][:, 0] * batch["active"][:, :, None]
    result = {
        "active_fraction": float(batch["active"][:, BURN:].float().mean()),
        "squared_error_delta": spatial_error_delta(p, d, t, batch["rate"], effective),
        "raw_loss_delta": float(
            linear_objective(p, t, batch["rate"])["total"]
            - linear_objective(d, t, batch["rate"])["total"]
        ),
    }
    variants = {
        "dry_dsp": batch["x"],
        "candidate_input_phase": hybrid_features(p, batch["x"][:, 1], batch["x"]),
        "oracle_amplitude_input_phase": hybrid_features(t, batch["x"][:, 1], batch["x"]),
        "oracle_complex": torch.cat((batch["x"][:, :, :BURN], batch["y"][:, :, BURN:]), dim=2),
    }
    outputs = {}
    for name, features in variants.items():
        wave, _ = render(
            features,
            batch["x"],
            batch["mask"],
            batch["active"],
            batch["gate"],
            batch["dry"],
            limiter_mode="smooth-knee",
        )
        output = spectral_features(wave[:, : 32768 - 128], 256, 128)
        amplitude = torch.pow(10.0, output[:, 0] / 20)
        loss = linear_objective(amplitude, t[:, : amplitude.shape[1]], batch["rate"], burn=BURN + 9)
        a, b = (
            wave[:, (BURN + 9) * 128 : 32768 - 128],
            batch["clean_wave"][:, (BURN + 9) * 128 : 32768 - 128],
        )
        outputs[name] = {
            "spectral": float(loss["total"]),
            "waveform_relative_rms": float(
                (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-7)
            ),
        }
    result["output"] = outputs
    return result


def run(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    config = json.loads((root / "config.json").read_text())
    rows, hashes = load_sources([Path(p) for p in config["manifests"]])
    if hashes != config["manifests"]:
        raise ValueError("source changed")
    detector = root / "binaries/soundex-gain-data"
    if sha256_file(detector) != config["detector_sha256"]:
        raise ValueError("detector changed")
    clips = EnvelopeClips(rows, detector, config["cache_path"])
    records, gradients, checkpoints = [], [], {}
    for seed in config["seeds"]:
        directory = root / f"anchored_band-{seed}"
        receipt = json.loads((directory / "complete.json").read_text())
        checkpoint = directory / "best.pth"
        if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        checkpoints[str(seed)] = sha256_file(checkpoint)
        model = make_shape_model("anchored_band").eval()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        )
        for split, specs in (
            ("train_probe", config["training_probe_specs"]),
            ("validation", config["validation_specs"]),
            ("identity", identity_specs(config["validation_specs"])),
        ):
            for spec in specs:
                row = clips.rows[spec["row_id"]]
                records.append(
                    {
                        "seed": seed,
                        "split": split,
                        "track_id": row["track_id"],
                        "corpus": row["corpus"],
                        "spec": spec,
                        **inspect(model, clips, spec),
                    }
                )
        for spec in config["training_probe_specs"]:
            row = rows[spec["row_id"]]
            a, ga = loss_and_gradient(model, clips, spec)
            b, gb = loss_and_gradient(model, clips, identity_specs([spec])[0])
            cosine = float((ga @ gb) / (ga.norm() * gb.norm()).clamp_min(1e-30))
            gradients.append(
                {
                    "seed": seed,
                    "spec": spec,
                    "track_id": row["track_id"],
                    "corpus": row["corpus"],
                    "paired_loss": a,
                    "identity_loss": b,
                    "cosine": cosine,
                    "paired_norm": float(ga.norm()),
                    "identity_norm": float(gb.norm()),
                }
            )
        print(
            json.dumps({"seed": seed, "records": len(records), "gradients": len(gradients)}),
            flush=True,
        )
    summary = {}
    for seed in config["seeds"]:
        groups = {}
        for split in ("train_probe", "validation", "identity"):
            for corpus in ("music_library", "speech_library"):
                by_track = defaultdict(list)
                for r in records:
                    if (r["seed"], r["split"], r["corpus"]) == (seed, split, corpus):
                        by_track[r["track_id"]].append(r)

                def mean(fn, by_track=by_track):
                    return float(
                        np.mean([np.mean([fn(r) for r in rr]) for rr in by_track.values()])
                    )

                groups[split + "_" + corpus] = {
                    "recordings": len(by_track),
                    "raw_loss_delta": mean(lambda r: r["raw_loss_delta"]),
                    "squared_error_delta": {
                        k: mean(lambda r, k=k: r["squared_error_delta"][k])
                        for k in ("applied_raw", "discarded_raw", "blended")
                    },
                    "output": {
                        v: {
                            k: mean(lambda r, v=v, k=k: r["output"][v][k])
                            for k in ("spectral", "waveform_relative_rms")
                        }
                        for v in records[0]["output"]
                    },
                }
        gs = [g for g in gradients if g["seed"] == seed]
        summary[str(seed)] = {
            "groups": groups,
            "gradients": {
                "samples": len(gs),
                "negative_cosines": sum(g["cosine"] < 0 for g in gs),
                "median_cosine": float(np.median([g["cosine"] for g in gs])),
            },
        }
    write_json_report(
        {
            "scope": "post-hoc development and training diagnosis; oracle targets never inference inputs",
            "checkpoints": checkpoints,
            "script_sha256": sha256_file(Path(__file__)),
            "summary": summary,
            "records": records,
            "gradients": gradients,
        },
        args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to replace report")
    run(args)
