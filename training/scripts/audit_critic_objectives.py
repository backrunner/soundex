"""P61 fixed-fit gradient audit; no critic or restoration training."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from evaluation.critic_objectives import (
    copied_adam,
    cosine,
    flattened_gradient,
    pcm_feature_probe,
    task_losses,
)
from evaluation.reporting import sha256_file, write_json_report
from evaluation.trainability_audit import assert_reserved_isolation
from scripts.calibrate_spectral_texture import metadata, sources
from scripts.calibrate_texture_context import load_critic
from scripts.preflight_spectral_texture import read, verify
from scripts.summarize_guarded_transfer import coverage


def audit(parent, output):
    torch.set_num_threads(1)
    if output.exists():
        raise ValueError("fresh audit output required")
    config = read(parent / "registration.json")
    verify(read(parent / "complete.json")["files"])
    verify(config["hashes"])
    critics = read(parent / "critics.json")
    inherited = config["parent_config"]
    indices = inherited["fit_review_indices"]
    selected = [
        sample for index in indices for sample in metadata(inherited, "optimization", index)
    ]
    catalogs = sorted(parent.parent.glob("*/evaluation-catalog.jsonl"))
    reserved = [json.loads(line) for p in catalogs for line in p.read_text().splitlines() if line]
    catalog = parent.parents[1] / "catalogs/current-approved.jsonl"
    universe = list(sources(inherited).values()) + [
        json.loads(s) for s in catalog.read_text().splitlines() if s
    ]
    assert_reserved_isolation([s["row"] for s in selected], universe, reserved)
    hashes = dict(config["hashes"])
    for entry in critics.values():
        for version in ("initial", "trained"):
            hashes[entry[version]["path"]] = entry[version]["sha256"]
    for p in (
        parent / "registration.json",
        parent / "critics.json",
        parent / "complete.json",
        catalog,
        *catalogs,
    ):
        hashes[str(p)] = sha256_file(p)
    verify(hashes)
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for name in (
        "evaluation/critic_objectives.py",
        "scripts/audit_critic_objectives.py",
        "tests/test_critic_objectives.py",
    ):
        source = root / name
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = sha256_file(source)
        hashes[str(target)] = sha256_file(target)
    registration = {
        "format": "soundex-critic-objectives-p61-v1",
        "hashes": hashes,
        "parent": str(parent),
        "fit_indices": indices,
        "fit_coverage": coverage([{"samples": {"fit": selected}}], "fit"),
        "scope": "Every P60 arm, both fixed generators, initial/trained critics, 16 original fit draws. No calibration or held-out scoring, selection, optimizer or parameter mutation. Local gradients are not a causal history decomposition or quality proof.",
        "decomposition": "Real-plus-fake tasks for prediction/phase/time. Use last-cycle phase seeds and shift fractions corresponding to each original fit draw. Reconstruct exact structured weights including roundtrip exclusions. Report unweighted task cosines and actual-arm total gradient per corpus/all. Copied fresh and saved-moment Adam at original lr1e-4 clip5, directional derivatives only.",
        "feature_probe": "Each paired predicted PCM example: frozen feature vs existing multires spectral objective gradients; fixed 1%-of-current-error RMS negative feature-gradient step. Raw unconstrained PCM, not a realizable generator update or complete direct+MR objective. Compare random and trained critics separately; no reward from classifier logits.",
    }
    write_json_report(registration, output / "registration.json")
    rows, pcm = [], []
    for label, entry in critics.items():
        for version in ("initial", "trained"):
            critic = load_critic(entry, version)
            checkpoint = torch.load(entry[version]["path"], weights_only=False)
            before = {k: v.clone() for k, v in critic.state_dict().items()}
            parameters = list(critic.parameters())
            flattened = torch.cat([p.detach().flatten() for p in parameters])
            for index in indices:
                values = torch.load(config["caches"][entry["generator"]][index], weights_only=False)
                step = 768 + index
                tasks, weights = task_losses(critic, *values, config["seed"] + step, step)
                meta = metadata(inherited, "optimization", index)
                speech = torch.tensor([s["row"]["corpus"] == "speech_library" for s in meta])
                for corpus, mask in (
                    ("all", torch.ones_like(speech)),
                    ("music", ~speech),
                    ("speech", speech),
                ):
                    gradients = [
                        flattened_gradient(tasks[mask, i].mean(), parameters) for i in range(3)
                    ]
                    active = ((tasks * weights).sum(-1) if entry["structured"] else tasks[:, 0])[
                        mask
                    ].mean()
                    total = flattened_gradient(active, parameters)
                    displacements = {
                        "fresh": copied_adam(flattened, total, checkpoint, persistent=False)
                    }
                    if version == "trained":
                        displacements["saved_moments"] = copied_adam(
                            flattened, total, checkpoint, persistent=True
                        )
                    rows.append(
                        {
                            "label": label,
                            "critic": version,
                            "draw": index,
                            "corpus": corpus,
                            "prediction_phase_cosine": cosine(gradients[0], gradients[1]),
                            "prediction_time_cosine": cosine(gradients[0], gradients[2]),
                            "task_losses": tasks[mask].detach().mean(0).tolist(),
                            "task_gradient_norms": [float(g.double().norm()) for g in gradients],
                            "negative_total_gradient_derivatives": [
                                -float(g.double().dot(total.double())) for g in gradients
                            ],
                            "adam_task_derivatives": {
                                name: [float(g.double().dot(delta)) for g in gradients]
                                for name, delta in displacements.items()
                            },
                        }
                    )
                for sample, result in zip(meta, pcm_feature_probe(critic, *values), strict=True):
                    pcm.append(
                        {
                            "label": label,
                            "critic": version,
                            "draw": index,
                            "source": sample["row"]["source_checksum"],
                            "corpus": sample["row"]["corpus"],
                            **result,
                        }
                    )
            if any(not torch.equal(v, before[k]) for k, v in critic.state_dict().items()):
                raise ValueError("critic mutated during audit")
            print(
                json.dumps(
                    {
                        "label": label,
                        "critic": version,
                        "gradient_rows": len(rows),
                        "pcm_rows": len(pcm),
                    }
                ),
                flush=True,
            )
    verify(hashes)
    write_json_report(
        {
            "gradient_rows": rows,
            "pcm_rows": pcm,
            "critic_updates": 0,
            "generator_updates": 0,
            "qualified": False,
        },
        output / "summary.json",
    )
    write_json_report(
        {
            "files": {
                str(output / name): sha256_file(output / name)
                for name in ("registration.json", "summary.json")
            }
        },
        output / "complete.json",
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "gradient_rows": len(rows),
                "pcm_rows": len(pcm),
                "generator_updates": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        audit(args.parent.resolve(), args.output.resolve())
    except Exception as error:
        if args.output.exists():
            write_json_report(
                {"error": repr(error), "qualified": False}, args.output / "failure.json"
            )
        raise
