#!/usr/bin/env python3
"""Frozen-endpoint gradient decomposition on registered TRAINING draws only.

No update, candidate selection, DSP proxy change or new loss is performed.
"""

import argparse
import shutil
from itertools import combinations
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.direct_spectrum_objective import WEIGHTS as DIRECT_WEIGHTS
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import WEIGHTS as MR_WEIGHTS
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import load_restoration_head, predict_budget
from scripts.run_consistency_followup import read, registered


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    study, output = args.study.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh evidence directory")
    config, endpoints = read(study / "registration.json"), read(study / "review-registration.json")
    if config["format"] != "soundex-detector-condition-v1":
        raise ValueError("P38 compressed-risk control endpoints required")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("complete registered training before diagnosis")
    indices = [int(i * (config["steps"] - 1) / 15) for i in range(16)]
    hashes = dict(config["input_hashes"])
    for seed in config["seeds"]:
        hashes[config["schedule_files"][str(seed)]] = sha256_file(
            Path(config["schedule_files"][str(seed)])
        )
    hashes[str(study / "registration.json")] = sha256_file(study / "registration.json")
    hashes[str(study / "review-registration.json")] = sha256_file(
        study / "review-registration.json"
    )
    source = Path(__file__).resolve().parents[1]
    for name, digest in config["source_hashes"].items():
        if name.startswith(("models/", "data/")):
            hashes[str(source / name)] = digest
    hashes[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input/source drift: {path}")
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    registered(
        {
            "scope": "post-hoc fixed-endpoint gradients on 16 evenly spaced registered TRAINING draws per seed; no optimization, validation selection or generalization claim",
            "draw_indices": indices,
            "hashes": hashes,
            "endpoints": {k: v for k, v in endpoints.items() if k.startswith("control-")},
            "decomposition": "exact local log1p derivative via detached 1+loss denominator; paired component gradients plus identity and budget must reconstruct the actual gradient",
            "units": "parameter cosines compare directions; output magnitude uses dB coordinates and phase uses radians, so their norm ratio is not an invariant importance score",
        },
        output / "registration.json",
    )
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    reports = {}
    for seed in config["seeds"]:
        schedule = read(Path(config["schedule_files"][str(seed)]))
        pool = LazyPool(schedule["specs"], clips)
        label = f"control-{config['steps']}-{seed}"
        endpoint = endpoints[label]
        path = Path(endpoint["path"])
        if sha256_file(path) != endpoint["sha256"]:
            raise ValueError("checkpoint drift")
        model = load_restoration_head(
            torch.load(path, map_location="cpu", weights_only=False)
        ).eval()
        params = list(model.parameters())
        groups = {}
        offset = 0
        for name, parameter in model.named_parameters():
            group = "budget" if name.startswith("budget_head") else ".".join(name.split(".")[:2])
            groups.setdefault(group, []).append((offset, offset + parameter.numel()))
            offset += parameter.numel()
        cases = []
        for index in indices:
            objective, context = pool.select(schedule["draws"][index])
            batch = objective.batch
            prediction, budget = predict_budget(model, batch, context)
            direct = direct_spectrum_loss(prediction, batch)
            mr = multires_spectrum_loss(prediction, batch)
            identity = batch["identity"].bool()
            pair = ~identity
            denominators = (1 + direct["per_example"].detach(), 1 + mr["per_example"].detach())
            terms = {}
            for prefix, result, weights, denominator in zip(
                ("direct", "mr"),
                (direct, mr),
                (DIRECT_WEIGHTS, MR_WEIGHTS),
                denominators,
                strict=True,
            ):
                for name, weight in weights.items():
                    terms[f"paired_{prefix}_{name}"] = (
                        0.5 * (weight * result["components"][name] / denominator)[pair].mean()
                    )
            exact = direct["per_example"].log1p() + mr["per_example"].log1p()
            terms["identity"] = 0.5 * exact[identity].mean()
            terms["budget"] = 0.1 * budget_risk(budget, batch)
            vectors, feature_norms, phase_vectors, magnitude_vectors = {}, {}, {}, {}
            hz = torch.arange(prediction.shape[-1])[None, :] * batch["rate"].reshape(-1, 1) / 256
            high = (hz >= 8000)[pair, None, :]
            for name, term in terms.items():
                gradients = torch.autograd.grad(
                    term, [*params, prediction], retain_graph=True, allow_unused=True
                )
                vectors[name] = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).flatten()
                        for g, p in zip(gradients[:-1], params, strict=True)
                    ]
                )
                g = gradients[-1]
                if g is not None:
                    magnitude = g[pair, 0, 89:] * high
                    phase = g[pair, 1, 89:] * high
                    feature_norms[name] = {
                        "paired_hf_db_norm": float(magnitude.norm()),
                        "paired_hf_radian_norm": float(phase.norm()),
                    }
                    phase_vectors[name] = phase.flatten()
                    magnitude_vectors[name] = magnitude.flatten()
            actual = 0.5 * (exact[pair].mean() + exact[identity].mean()) + terms["budget"]
            actual_gradients = torch.autograd.grad(actual, params, allow_unused=True)
            total = torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(actual_gradients, params, strict=True)
                ]
            )
            torch.testing.assert_close(sum(vectors.values()), total, rtol=2e-4, atol=2e-7)
            cosines = {
                f"{a}/{b}": float(
                    torch.nn.functional.cosine_similarity(vectors[a], vectors[b], dim=0, eps=1e-20)
                )
                for a, b in combinations(vectors, 2)
            }
            phase_cosines = {
                f"{a}/{b}": float(
                    torch.nn.functional.cosine_similarity(
                        phase_vectors[a], phase_vectors[b], dim=0, eps=1e-20
                    )
                )
                for a, b in combinations(phase_vectors, 2)
            }
            magnitude_cosines = {
                f"{a}/{b}": float(
                    torch.nn.functional.cosine_similarity(
                        magnitude_vectors[a], magnitude_vectors[b], dim=0, eps=1e-20
                    )
                )
                for a, b in combinations(magnitude_vectors, 2)
            }
            parameter_groups = {}
            for group, slices in groups.items():
                parts = {
                    name: torch.cat([v[a:b] for a, b in slices]) for name, v in vectors.items()
                }
                parameter_groups[group] = {
                    "norms": {name: float(v.norm()) for name, v in parts.items()},
                    "cosines": {
                        f"{a}/{b}": float(
                            torch.nn.functional.cosine_similarity(
                                parts[a], parts[b], dim=0, eps=1e-20
                            )
                        )
                        for a, b in combinations(parts, 2)
                    },
                }
            cases.append(
                {
                    "draw_index": index,
                    "sample_specs": [schedule["specs"][j] for j in schedule["draws"][index]],
                    "parameter_gradient_norms": {k: float(v.norm()) for k, v in vectors.items()},
                    "parameter_cosines": cosines,
                    "parameter_groups": parameter_groups,
                    "output_hf_gradient_norms": feature_norms,
                    "output_hf_phase_cosines": phase_cosines,
                    "output_hf_db_cosines": magnitude_cosines,
                    "decomposition_max_error": float((sum(vectors.values()) - total).abs().max()),
                }
            )
            print(label, index, "complete", flush=True)
        reports[label] = cases
        write_json_report(cases, output / f"{label}.json")
        if sha256_file(path) != endpoint["sha256"]:
            raise ValueError("checkpoint changed during diagnosis")
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input/source drift after diagnosis: {path}")
    write_json_report(
        {
            "status": "complete",
            "cases": reports,
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
