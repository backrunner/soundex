#!/usr/bin/env python3
"""Register one P16 coefficient from training-only initial gradient RMS; no dev access."""

import argparse
import json
import math
from pathlib import Path

import torch

from data.causal_study import digest, load_sources
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from models.spectral_consistency import consistency_loss
from scripts.run_subset_study import Pool, features


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.study / "config.json").read_text())
    parent = json.loads(args.parent.read_text())
    if sha256_file(args.parent) != config["parent_sha256"]:
        raise ValueError("parent drift")
    rows, manifests = load_sources([Path(p) for p in config["manifests"]])
    if manifests != config["manifests"]:
        raise ValueError("manifest drift")
    bank = config["crop_sampling"]["bank"]
    if len(bank) != 320 or any(
        rows[s["row_id"]]["split"] != "train" or rows[s["row_id"]].get("held_out", False)
        for s in bank
    ):
        raise ValueError("registered training-only crop bank required")
    detector = args.study / "binaries/soundex-gain-data"
    if sha256_file(detector) != config["detector_sha256"]:
        raise ValueError("detector drift")
    schedules = {}
    for seed in config["seeds"]:
        directory = args.study / f"size-40-{seed}"
        receipt = json.loads((directory / "complete.json").read_text())
        if sha256_file(directory / "draws.json") != receipt["hashes"]["draws.json"]:
            raise ValueError("schedule drift")
        draws = json.loads((directory / "draws.json").read_text())["crop_draws"][:64]
        if sorted(i for batch in draws for i in batch) != list(range(320)):
            raise ValueError("calibration must visit each training position exactly once per seed")
        schedules[str(seed)] = draws
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not any(
            x in {".venv", "tests", "outputs", "__pycache__"} for x in p.relative_to(source).parts
        )
    }
    registration = {
        "rule": "weight = 0.1 * sqrt(sum(main_parameter_gradient_squared) / sum(aux_parameter_gradient_squared)); initial model only; no updates",
        "seeds": config["seeds"],
        "bank_sha256": digest(bank),
        "source_sha256": digest(hashes),
        "source_files": hashes,
        "schedules": schedules,
        "study_sha256": sha256_file(args.study / "config.json"),
    }
    write_json_report(registration, root / "calibration-registration.json")
    pool = Pool(bank, EnvelopeClips(rows, detector, Path(parent["cache_path"])))
    results = []
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        model = make_rendered_model("rendered_balanced")
        parameters = tuple(model.parameters())
        for draw in schedules[str(seed)]:
            objective, context = pool.select(draw)
            prediction = features(model, objective, context)
            main = objective(prediction)["total"]
            auxiliary = consistency_loss(prediction, objective.batch)
            squared = []
            for loss, retain in ((main, True), (auxiliary, False)):
                gradients = torch.autograd.grad(loss, parameters, retain_graph=retain)
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError("nonfinite calibration gradient")
                squared.append(sum(float(g.double().square().sum()) for g in gradients))
            results.append(
                {
                    "seed": seed,
                    "indices": draw,
                    "main": float(main.detach()),
                    "auxiliary": float(auxiliary.detach()),
                    "main_gradient_squared": squared[0],
                    "auxiliary_gradient_squared": squared[1],
                }
            )
        print(json.dumps({"seed": seed, "calibration_batches": 64}), flush=True)
    a, b = (
        sum(r[key] for r in results)
        for key in ("main_gradient_squared", "auxiliary_gradient_squared")
    )
    if a <= 0 or b <= 0:
        raise ValueError("nonzero calibration gradients required")
    weight = 0.1 * math.sqrt(a / b)
    write_json_report(
        {
            "weight": weight,
            "seeds": config["seeds"],
            "bank_sha256": digest(bank),
            "source_sha256": digest(hashes),
            "registration_sha256": sha256_file(root / "calibration-registration.json"),
            "results": results,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "weight": weight,
                "main_gradient_rms": math.sqrt(a / len(results)),
                "auxiliary_gradient_rms": math.sqrt(b / len(results)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
