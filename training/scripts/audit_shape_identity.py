#!/usr/bin/env python3
"""P9 first-update attribution: paired vs identity samples, same original P8 draws."""

import argparse
import copy
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.conditioned_shape import ARMS, make_shape_model
from models.linear_objective import linear_objective
from scripts.audit_linear_optimization import forward, measure, parameter_group, prepared


def flatten(grads):
    return torch.cat([g.detach().flatten() for g in grads])


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def run(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    config = json.loads((root / "config.json").read_text())
    rows, manifests = load_sources([Path(p) for p in config["manifests"]])
    if manifests != config["manifests"]:
        raise ValueError("source changed")
    detector = root / "binaries/soundex-gain-data"
    if sha256_file(detector) != config["detector_sha256"]:
        raise ValueError("detector changed")
    clips = EnvelopeClips(rows, detector, config["cache_path"])
    reports = []
    for seed in config["seeds"]:
        specs = json.loads((root / f"draws-{seed}.json").read_text())["draws"][:4]
        pair = prepared(clips, specs, "cpu")
        identity = [s["row_id"].endswith(":identity") for s in specs]
        for arm in ARMS:
            torch.manual_seed(seed)
            # Double precision excludes a float32-only roundoff explanation.
            model = make_shape_model(arm).double()
            typed = ({k: v.double() for k, v in pair[0].items()}, pair[1].double())
            before = copy.deepcopy(model.state_dict())
            initial = measure(model, typed)
            p, t, _ = forward(model, typed)
            named = list(model.named_parameters())
            samples, gradsets = [], []
            for i in range(len(specs)):
                loss = linear_objective(p[i : i + 1], t[i : i + 1], typed[0]["rate"][i : i + 1])[
                    "total"
                ]
                gradients = torch.autograd.grad(loss, [v for _, v in named], retain_graph=True)
                gradsets.append(gradients)
                samples.append(
                    {
                        "spec": specs[i],
                        "identity": identity[i],
                        "loss": float(loss.detach()),
                        "gradient_norms": {
                            group: float(
                                flatten(
                                    [
                                        g
                                        for (n, _), g in zip(named, gradients, strict=True)
                                        if parameter_group(n) == group
                                    ]
                                ).norm()
                            )
                            for group in ("energy", "shape", "backbone")
                        },
                    }
                )
            grouped = {}
            for name, selected in (
                ("all", list(range(len(specs)))),
                ("paired_only", [i for i, flag in enumerate(identity) if not flag]),
                ("identity_only", [i for i, flag in enumerate(identity) if flag]),
            ):
                grouped[name] = [
                    sum(gradsets[i][j] for i in selected) / len(specs) for j in range(len(named))
                ]
            directions = {k: flatten(v) for k, v in grouped.items()}
            updates = {}
            for name, gradients in grouped.items():
                model.load_state_dict(before)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                for (_, parameter), gradient in zip(named, gradients, strict=True):
                    parameter.grad = gradient.detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                updates[name] = measure(model, typed)
            reports.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "initial": initial,
                    "samples": samples,
                    "paired_identity_cosine": cosine(
                        directions["paired_only"], directions["identity_only"]
                    ),
                    "identity_to_paired_gradient_norm": float(
                        directions["identity_only"].norm() / directions["paired_only"].norm()
                    ),
                    "updates": updates,
                }
            )
    write_json_report(
        {
            "scope": "float64 CPU first-batch gradient attribution, not generalization",
            "parent_config_sha256": sha256_file(root / "config.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "reports": reports,
        },
        args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to replace existing report")
    run(args)
