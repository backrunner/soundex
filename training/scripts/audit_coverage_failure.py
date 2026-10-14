#!/usr/bin/env python3
"""Post-hoc P19 stage and frequency-transfer audit; no fitting or candidate selection."""

import argparse
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.band_transfer import band_masks, intervene, transfer
from evaluation.loss_scale import spectral_parts, summarize_parts
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from scripts.audit_render_stages import stages
from scripts.run_fit_diagnostic import metrics
from scripts.run_subset_study import Pool, features


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    config = read(args.study / "registration.json")
    checkpoints = read(args.study / "review-registration.json")
    for name, expected in config["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"input drift: {name}")
    if args.output.exists():
        raise ValueError("audit output already exists; choose a new directory")
    args.output.mkdir(parents=True)
    inputs = [Path(p) for p in config["input_hashes"]]
    manifests = [p for p in inputs if p.name == "manifest.jsonl"]
    detector = next(p for p in inputs if p.name == "soundex-gain-data")
    parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
    rows, _ = load_sources(manifests)
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    # Exactly one existing bank position per training recording, selected without scores.
    training = {}
    for spec in config["groups"]["wide"]["bank"]:
        training.setdefault(rows[spec["row_id"]]["track_id"], spec)
    source = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    registration = {
        "scope": "post-hoc diagnosis, one fixed crop per train recording and all reused development clips; not independent evaluation",
        "input_hashes": config["input_hashes"],
        "source_hashes": source_hashes,
        "checkpoints": checkpoints,
        "specs": {"training": list(training.values()), "development": config["development_specs"]},
        "selection": "stage/transfer covers every listed case; only/without interventions use three largest final music identity losses per seed",
    }
    write_json_report(registration, args.output / "registration.json")
    reports = {}
    for group, specs in registration["specs"].items():
        pool = Pool(specs, clips)
        labels = [
            f"wide-{step}-{seed}"
            for step in ([16192] if group == "training" else [2048, 16192])
            for seed in config["seeds"]
        ]
        for label in labels:
            binding = checkpoints[label]
            checkpoint = Path(binding["path"])
            if sha256_file(checkpoint) != binding["sha256"]:
                raise ValueError("checkpoint drift")
            model = make_rendered_model("rendered_balanced").eval()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            )
            proxy = read(args.study / "review" / label / "proxy.json")
            dev = proxy["groups"]["development"]["rows"]
            worst = {
                r["spec"]["row_id"]
                for r in sorted(
                    (r for r in dev if r["corpus"] == "music_library"),
                    key=lambda r: r["metrics"]["identity_preservation"],
                    reverse=True,
                )[:3]
            }
            records = []
            for i, row in enumerate(pool.rows):
                objective, context = pool.select([i])
                b = objective.batch
                predicted = features(model, objective, context)
                baseline, baseline_wave = stages(b["x"], b)
                candidate, wave = stages(predicted, b)
                score = metrics(objective(predicted), b)
                if group == "development":
                    expected = dev[i]
                    if expected["spec"] != specs[i]:
                        raise ValueError("case order drift")
                    if (
                        max(
                            abs(score[k] - expected["metrics"][k])
                            for k in ("total", "paired_restoration", "identity_preservation")
                        )
                        > 2e-6
                    ):
                        raise ValueError("frozen score mismatch")
                transmission = transfer(predicted, b["x"], b)
                intervention = {}
                if group == "development" and "16192" in label and specs[i]["row_id"] in worst:
                    masks = band_masks(b["rate"], dtype=predicted.dtype, device=predicted.device)
                    for source_band in range(5):
                        for remove in (False, True):
                            modified = intervene(
                                predicted, b["x"], masks, source_band, remove=remove
                            )
                            intervention[f"{'without' if remove else 'only'}-{source_band}"] = (
                                metrics(objective(modified), b)
                            )
                for index, role in enumerate(("paired", "identity")):
                    record = {
                        "spec": specs[i],
                        "track_id": row["track_id"],
                        "corpus": row["corpus"],
                        "genre": row.get("source_metadata", {}).get("genre", "unlabeled"),
                        "codec_id": row["codec_id"],
                        "rate": row["sample_rate"],
                        "channel_role": row["channel_role"],
                        "role": role,
                        "score": score["rows"][index],
                        "stages": {},
                        "initial_stages": {},
                        "active_fraction": float(b["active"][index, 89:].float().mean()),
                        "cutoff_mean": float(b["cutoff"][index, 89:].mean()),
                        "prelimit_peak": float(wave[index, objective.region].abs().max()),
                        "baseline_prelimit_peak": float(
                            baseline_wave[index, objective.region].abs().max()
                        ),
                        "gate_min": float(b["gate"][index, objective.region].min()),
                        "transfer": {
                            k: v[index].tolist() for k, v in transmission.items() if k != "masks"
                        },
                        "interventions": intervention,
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
                    records.append(record)
            name = f"{group}-{label}.json"
            write_json_report(
                {"checkpoint_sha256": binding["sha256"], "records": records}, args.output / name
            )
            reports[name] = sha256_file(args.output / name)
            print(json.dumps({"group": group, "model": label, "pairs": len(specs)}), flush=True)
        del pool
    if any(sha256_file(source / p) != expected for p, expected in source_hashes.items()):
        raise ValueError("audit source changed during execution")
    write_json_report(
        {"reports": reports, "registration_sha256": sha256_file(args.output / "registration.json")},
        args.output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
