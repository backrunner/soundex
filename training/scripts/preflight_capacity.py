#!/usr/bin/env python3
"""Train-only 16-update hardware and gradient smoke for the three P21 widths."""

import argparse
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.head_study import auxiliary, predict
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import RestorationHead
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read
from scripts.run_fit_diagnostic import replay
from scripts.run_subset_study import Pool


def run(root, output):
    torch.set_num_threads(1)
    config = read(root / "registration.json")
    inputs = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
    detector = next(p for p in inputs if p.name == "soundex-gain-data")
    parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
    onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
    seed = config["seeds"][0]
    draws = read(root / f"calibration/draws-{seed}.json")["draws"][:16]
    indices = sorted({i for draw in draws for i in draw})
    lookup = {index: i for i, index in enumerate(indices)}
    pool = Pool([config["bank"][i] for i in indices], clips)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("scaled96", "scaled144", "scaled224"):
        model_config = config["model_configs"][name]
        torch.manual_seed(seed)
        model = RestorationHead(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["rates"][name])
        gradients, losses = [], []
        for draw in draws:
            objective, context = pool.select([lookup[i] for i in draw])
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(model, objective, context)
            loss = objective(prediction)["total"] + config["weight"] * auxiliary(
                prediction, objective.batch
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite preflight loss")
            loss.backward()
            gradients.append(
                float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                )
            )
            losses.append(float(loss.detach()))
            optimizer.step()
        with torch.no_grad():
            native = replay(
                predict(model, objective, context), objective.batch, objective, binary, onnx
            )
        write_json_report(
            {
                "scope": "last of 16 registered training batches; paired and identity",
                "draw": draws[-1],
                "roles": native,
            },
            output / f"{name}-native.json",
        )
        checkpoint = output / f"{name}.pth"
        atomic_save(
            {"arm": name, "model_config": model_config, "model": model.state_dict()}, checkpoint
        )
        export_and_measure(checkpoint, output / name)
        results[name] = {
            "preclip_gradient_norms": gradients,
            "losses": losses,
            "median_gradient": float(np.median(gradients)),
            "clipped_updates": sum(v > 5 for v in gradients),
            "native_proxy_max_error": max(row["proxy_max_error"] for row in native),
            "checkpoint_sha256": sha256_file(checkpoint),
            "benchmark_sha256": sha256_file(output / name / "report.json"),
        }
        print(name, results[name]["median_gradient"], results[name]["clipped_updates"], flush=True)
    write_json_report(
        {
            "scope": "first 16 registered training batches, one seed, nonzero learned export and hardware integration only; no generalization or calibration pass claim",
            "registration_sha256": sha256_file(root / "registration.json"),
            "results": results,
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
