#!/usr/bin/env python3
"""P30: all P29 endpoints, fixed training crops, HF amplitude/phase oracle swaps.

No fitting, candidate selection, threshold tuning, new inference path or promotion.
Offline features replay through actual Rust DSP; not an ORT performance benchmark.
"""

import argparse
import fcntl
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.absolute_audio import compare_pcm
from evaluation.phase_swap import VARIANTS, complex_error_parts, high_frequency_swaps
from evaluation.reporting import sha256_file, write_json_report
from models.causal_objective import render
from models.complex_synthesis import complex_features, synthesize
from models.restoration_head import load_restoration_head, predict_head
from scripts.audit_capacity_outliers import native_audio
from scripts.replay_factorized_study import aligned_band_metrics
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered


def register(study, output):
    parent = read(study / "registration.json")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("completed P29 required")
    for rel, expected in parent["source_hashes"].items():
        if sha256_file(study.parent / "source/training" / rel) != expected:
            raise ValueError("parent frozen source drift")
    endpoints = read(study / "review-registration.json")
    expected = {
        f"{arm}-{parent['primary_endpoint']}-{seed}"
        for arm in parent["arms"]
        for seed in parent["seeds"]
    }
    if set(endpoints) != expected:
        raise ValueError("all fixed final endpoints required")
    inputs = dict(parent["input_hashes"])
    for name in ("registration.json", "review-registration.json", "summary.json", "progress.json"):
        path = study / name
        inputs[str(path)] = sha256_file(path)
    for value in endpoints.values():
        inputs[value["path"]] = value["sha256"]
    source = Path(__file__).resolve().parents[1]
    config = {
        "format": "soundex-hf-phase-swap-v1",
        "scope": "post-hoc reference-assisted diagnosis on all 20 P29 fitted TRAINING crops; not independent evaluation, deployable predictions, listening or real-time qualification",
        "specs": parent["specs"],
        "checkpoints": endpoints,
        "variants": list(VARIANTS),
        "cutoff_hz": 4000,
        "replacement_start_hop": 80,
        "scored_start_hop": 89,
        "policy": "all final endpoints and both roles; fixed HF-only swaps; preserve candidate below 4 kHz and warmup; unchanged detector/gate/limiter; no selection or exclusions",
        "stage_policy": "raw OLA, detector-blended OLA, gated prelimit, actual Rust output; compare original clean PCM at common support, with calibrated absolute band errors",
        "native_roles": len(endpoints) * len(parent["specs"]) * len(VARIANTS) * 2
        + len(parent["specs"]) * 2,
        "cache_path": str(output / "feature-cache"),
        "input_hashes": inputs,
        "source_hashes": {
            str(p.relative_to(source)): sha256_file(p)
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
            and not set(p.relative_to(source).parts) & {".venv", "tests", "__pycache__", "outputs"}
        },
    }
    verify_inputs(config)
    registered(config, output / "registration.json")
    return config


@torch.no_grad()
def measure(features, objective, tools, row, spec, variant):
    b, region = objective.batch, objective.region
    waves, parity = native_audio(features, b, objective, *tools[:2], observed_binary=tools[2])
    _, prelimit = render(
        features, b["x"], b["mask"], b["active"], b["gate"], b["dry"], limiter_mode="smooth-knee"
    )
    stages = {
        "raw_ola": synthesize(complex_features(features)).numpy(),
        "blended_ola": prelimit.numpy(),
        "gated_prelimit": (b["dry"] * (1 - b["gate"]) + prelimit * b["gate"]).numpy(),
        "native_output": waves,
    }
    parts = complex_error_parts(features, b["y"], b["rate"])
    records = []
    for i, role in enumerate(("paired", "identity")):
        rate = int(b["rate"][i, 0])
        clean = b["clean_wave"][i, region].numpy()
        raw_metrics, output_metrics = aligned_band_metrics(
            features[i : i + 1],
            b["y"][i : i + 1],
            torch.from_numpy(waves[i : i + 1]),
            b["rate"][i : i + 1],
        )
        records.append(
            {
                "spec": spec,
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "role": role,
                "variant": variant,
                "parity": parity[i],
                "complex_parts": parts[i],
                "raw_band_metrics": raw_metrics,
                "output_band_metrics": output_metrics,
                "stages": {
                    name: compare_pcm(wave[i, region], clean, rate) for name, wave in stages.items()
                },
                "input_to_clean": compare_pcm(b["dry"][i, region].numpy(), clean, rate),
                "active_fraction": float(b["active"][i, 89:].float().mean()),
            }
        )
    return records


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["corpus"], row["role"])].append(row)
    result = {}
    for key, records in groups.items():
        values = {}
        for stage in ("raw_ola", "blended_ola", "gated_prelimit", "native_output"):
            per_track = defaultdict(list)
            for r in records:
                per_track[r["track_id"]].append(r["stages"][stage]["error"]["rms"])
            values[stage] = {
                "clip_mean_error_rms": float(
                    np.mean([r["stages"][stage]["error"]["rms"] for r in records])
                ),
                "recording_mean_error_rms": float(
                    np.mean([np.mean(v) for v in per_track.values()])
                ),
                "max_error_rms": max(r["stages"][stage]["error"]["rms"] for r in records),
                "band_mean_error_rms": {
                    band: float(
                        np.mean(
                            [r["stages"][stage]["bands"][band]["error"]["rms"] for r in records]
                        )
                    )
                    for band in records[0]["stages"][stage]["bands"]
                },
            }
        result["/".join(key)] = {
            "clips": len(records),
            "recordings": len({r["track_id"] for r in records}),
            "stages": values,
            "raw_band_metrics": {
                k: float(np.mean([r["raw_band_metrics"][k] for r in records]))
                for k in records[0]["raw_band_metrics"]
            },
            "output_band_metrics": {
                k: float(np.mean([r["output_band_metrics"][k] for r in records]))
                for k in records[0]["output_band_metrics"]
            },
            "hf_complex_parts": {
                k: float(np.mean([r["complex_parts"]["4000_24001"][k] for r in records]))
                for k in ("amplitude_power", "phase_power", "complex_power")
            },
        }
    return result


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = register(args.study.resolve(), output)
        if args.prepare_only:
            return
        paths = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        detector = next(p for p in paths if p.name == "soundex-gain-data")
        tools = [
            next(p for p in paths if p.name == n)
            for n in ("soundex-prediction-replay", "replay-contract.onnx", "soundex-analysis-trace")
        ]
        pool = LazyPool(
            config["specs"], BoundedEnvelopeClips(rows, detector, Path(config["cache_path"]))
        )
        batches = [pool.select([i]) for i in range(len(pool.specs))]
        all_summaries, reports, count = {}, {}, 0
        for label, entry in {"identity": None, **config["checkpoints"]}.items():
            path = output / "reports" / f"{label}.json"
            receipt_path = path.with_suffix(".receipt.json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["registration_sha256"] != sha256_file(
                    output / "registration.json"
                ) or receipt["sha256"] != sha256_file(path):
                    raise ValueError("completed report drift")
                report = read(path)
            else:
                model = (
                    load_restoration_head(
                        torch.load(entry["path"], map_location="cpu", weights_only=False)
                    ).eval()
                    if entry
                    else None
                )
                records = []
                with torch.no_grad():
                    for i, (objective, context) in enumerate(batches):
                        variants = (
                            high_frequency_swaps(
                                predict_head(model, objective.batch, context), objective.batch
                            )
                            if model
                            else {"identity": objective.batch["x"]}
                        )
                        for variant, features in variants.items():
                            records.extend(
                                measure(
                                    features, objective, tools, pool.rows[i], pool.specs[i], variant
                                )
                            )
                        if (i + 1) % 5 == 0:
                            state = {
                                "stage": "native_replay",
                                "case": label,
                                "pairs": i + 1,
                                "total_pairs": len(pool.specs),
                            }
                            write_json_report(state, output / "progress.json")
                            print(state, flush=True)
                report = {
                    "scope": config["scope"],
                    "checkpoint_sha256": entry["sha256"] if entry else None,
                    "rows": records,
                    "summary": summarize(records),
                }
                write_json_report(report, path)
                registered(
                    {
                        "registration_sha256": sha256_file(output / "registration.json"),
                        "sha256": sha256_file(path),
                    },
                    receipt_path,
                )
            all_summaries[label] = report["summary"]
            reports[str(path.relative_to(output))] = sha256_file(path)
            count += len(report["rows"])
        if count != config["native_roles"]:
            raise ValueError("incomplete native roles")
        verify_inputs(config)
        write_json_report(
            {
                "status": "complete",
                "scope": config["scope"],
                "native_roles": count,
                "registration_sha256": sha256_file(output / "registration.json"),
                "reports": reports,
                "results": all_summaries,
            },
            output / "summary.json",
        )
        write_json_report({"stage": "complete", "native_roles": count}, output / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
