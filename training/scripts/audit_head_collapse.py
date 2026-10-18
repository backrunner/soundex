#!/usr/bin/env python3
"""Training-only P26 mechanism audit; no fitting, selection or quality qualification."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, assert_isolated, protected_intervals
from data.causal_study import load_sources
from evaluation.head_study import auxiliary
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk, budget_target
from models.complex_synthesis import complex_features
from models.fit_diagnostic import BURN
from models.losses import blend_deployment_features
from models.restoration_head import load_restoration_head, predict_budget
from models.spectral_consistency import reference_features
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered


def stats(value):
    x = value.detach().double().flatten()
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite diagnostic")
    return {
        "mean": float(x.mean()),
        "rms": float(x.square().mean().sqrt()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def gradients(model, objective, context, weight, *, detach_auxiliary=False):
    prediction, budget = predict_budget(
        model, objective.batch, context, detach_auxiliary=detach_auxiliary
    )
    losses = objective(prediction)
    identity = objective.batch["identity"].bool()
    terms = {
        "paired_spectral": 0.5 * losses["spectral"][~identity].mean(),
        "paired_waveform": 0.125 * losses["waveform"][~identity].mean(),
        "identity": 0.5 * losses["preservation"][identity].mean(),
        "consistency": weight * auxiliary(prediction, objective.batch),
        "budget": 0.1 * budget_risk(budget, objective.batch),
    }
    named = list(model.named_parameters())
    groups = {
        "decoder_output": [
            i for i, (n, _) in enumerate(named) if n.startswith("backbone.decoder.2.")
        ],
        "budget": [i for i, (n, _) in enumerate(named) if n.startswith("budget_head.")],
        "shared": [
            i
            for i, (n, _) in enumerate(named)
            if not n.startswith(("backbone.decoder.2.", "budget_head."))
        ],
    }
    vectors, norms = {}, {}
    for name, term in terms.items():
        gs = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
        vectors[name] = {
            group: torch.cat(
                [
                    (gs[i] if gs[i] is not None else torch.zeros_like(named[i][1]))
                    .detach()
                    .flatten()
                    for i in indices
                ]
            )
            for group, indices in groups.items()
        }
        norms[name] = {group: float(v.norm()) for group, v in vectors[name].items()}
    cosines = {}
    for a, b in itertools.combinations(terms, 2):
        cosines[f"{a}:{b}"] = {}
        for group in groups:
            x, y = vectors[a][group], vectors[b][group]
            denominator = x.norm() * y.norm()
            cosines[f"{a}:{b}"][group] = (
                float(torch.dot(x, y) / denominator) if denominator > 0 else None
            )
    return {
        "weighted_losses": {k: float(v.detach()) for k, v in terms.items()},
        "gradient_norms": norms,
        "gradient_cosines": cosines,
    }


@torch.no_grad()
def inspect(model, objective, context):
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach()

        return hook

    def before_gelu(module, inputs):
        captured["decoder_pre_activation"] = inputs[0].detach()

    hooks = [
        model.backbone.decoder[1].register_forward_pre_hook(before_gelu),
        model.backbone.decoder[1].register_forward_hook(capture("decoder_hidden")),
        model.backbone.decoder[2].register_forward_hook(capture("decoder_logits")),
        model.budget_head.register_forward_hook(capture("budget_logits")),
    ]
    try:
        prediction, budget = predict_budget(model, objective.batch, context)
    finally:
        for hook in hooks:
            hook.remove()
    b = objective.batch
    identity = b["identity"].bool()
    targets = budget_target(b)
    original, predicted = complex_features(b["x"]), complex_features(prediction)
    effective = b["mask"] * b["active"][:, None, :, None]
    applied = complex_features(blend_deployment_features(b["x"], prediction, effective))
    actual_wave = objective.render(prediction)
    variants = {
        "identity_dsp": b["x"],
        "candidate": prediction,
        "oracle_amplitude_input_phase": reference_features(b, phase_reference="input"),
        "oracle_complex": reference_features(b, phase_reference="clean"),
    }
    scores = {name: objective(value) for name, value in variants.items()}
    result = {}
    for role, mask in (("paired", ~identity), ("identity", identity)):
        delta = predicted[mask, BURN + 9 :] - original[mask, BURN + 9 :]
        normalized = delta.abs() / original[mask, BURN + 9 :].abs().square().mean(
            -1, keepdim=True
        ).sqrt().clamp_min(1e-10)
        logits = captured["decoder_logits"][mask, 9:] * model.residual_slope
        result[role] = {
            "active_fraction": float(b["active"][mask, BURN + 9 :].float().mean()),
            "effective_mask_mean": float(
                (b["mask"][mask, :, BURN + 9 :] * b["active"][mask, None, BURN + 9 :, None]).mean()
            ),
            "decoder_hidden": stats(captured["decoder_hidden"][mask, 9:]),
            "decoder_pre_activation": stats(captured["decoder_pre_activation"][mask, 9:]),
            "hidden_near_zero_fraction": float(
                (captured["decoder_hidden"][mask, 9:].abs() < 1e-8).float().mean()
            ),
            "hidden_negative_saturation_fraction": float(
                (captured["decoder_pre_activation"][mask, 9:] < -6).float().mean()
            ),
            "scaled_decoder_logits": stats(logits),
            "tanh_saturated_fraction": float((logits.abs() > 3).float().mean()),
            "budget_logits": stats(captured["budget_logits"][mask, 9:]),
            "budget": stats(budget[mask, 9:]),
            "budget_target": stats(targets[mask, 9:]),
            "complex_residual_normalized": stats(normalized),
            "applied_complex_residual": stats(
                (applied[mask, BURN + 9 :] - original[mask, BURN + 9 :]).abs()
            ),
            "output_minus_identity_dsp": stats(
                (actual_wave - objective.baseline)[mask, objective.region]
            ),
            "scores": {
                name: {
                    k: float(value[k][mask].mean())
                    for k in ("restoration", "preservation", "spectral", "waveform")
                }
                for name, value in scores.items()
            },
        }
        per_band = []
        hz = torch.arange(129)[None, :] * b["rate"][mask] / 256
        for low, high in ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001)):
            band = ((hz >= low) & (hz < high))[:, None].expand_as(delta)
            per_band.append(
                {
                    "hz": [low, high],
                    "raw_residual": stats(delta.abs()[band]),
                    "applied_residual": stats(
                        (applied[mask, BURN + 9 :] - original[mask, BURN + 9 :]).abs()[band]
                    ),
                }
            )
        result[role]["bands"] = per_band
    return result


def run(args):
    torch.set_num_threads(1)
    root, output = args.study.resolve(), args.output.resolve()
    config = read(root / "registration.json")
    verify_inputs(config)
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    detector = next(p for p in paths if p.name == "soundex-gain-data")
    parent = next(p for p in paths if "rendered-p10" in str(p) and p.name == "config.json")
    schedule = read(Path(config["schedule_files"][f"broad-{config['seeds'][0]}"]))
    # Fixed geometry only. No score, genre, codec or validation outcome selects examples.
    draw_indices = [
        0,
        (config["steps"] - 1) // 3,
        2 * (config["steps"] - 1) // 3,
        config["steps"] - 1,
    ]
    specs = [schedule["specs"][i] for d in draw_indices for i in schedule["draws"][d]]
    assert_isolated(
        specs, rows, protected_intervals(config["development"] + config["shifted"], rows)
    )
    checkpoints = {
        f"{arm}-{seed}": root / f"{arm}-{seed}" / f"step-{config['primary_endpoint']}.pth"
        for arm in config["arms"]
        for seed in config["seeds"]
    }
    output.mkdir(parents=True, exist_ok=True)
    registered(
        {
            "scope": "training-only post-hoc mechanism diagnostic; not independent evaluation or fitted candidate",
            "study_sha256": sha256_file(root / "registration.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "source_hashes": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
                for p in Path(__file__).resolve().parents[1].rglob("*.py")
                if not set(p.parts) & {".venv", "__pycache__", "tests"}
            },
            "draw_indices": draw_indices,
            "specs": specs,
            "checkpoint_hashes": {k: sha256_file(p) for k, p in checkpoints.items()},
            "gradients": "first fixed 4-music/1-speech batch, all five weighted objective components, all final models",
            "oracle": "clean reference used only in nondeployable objective diagnostics",
        },
        output / "registration.json",
    )
    pool = LazyPool(specs, BoundedEnvelopeClips(rows, detector, Path(read(parent)["cache_path"])))
    batches = [pool.select(list(range(i, i + 5))) for i in range(0, len(specs), 5)]
    for label, checkpoint in checkpoints.items():
        model = load_restoration_head(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        ).eval()
        measured = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "batches": [inspect(model, *batch) for batch in batches],
            "gradients": gradients(model, *batches[0], config["weight"]),
        }
        write_json_report(measured, output / f"{label}.json")
        print(json.dumps({"complete": label}), flush=True)
    write_json_report(
        {
            "status": "complete",
            "models": len(checkpoints),
            "paired_training_crops": len(specs),
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
