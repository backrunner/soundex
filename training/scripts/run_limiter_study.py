"""Frozen post-P3 Rust limiter ablation. Historical review, never model promotion."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from data.causal_study import load_sources
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from evaluation.runner import _metric_row
from evaluation.rust_bridge import RustStreamEvaluator
from scripts.run_weak_band_study import freeze_binary


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    p3 = args.study.resolve()
    cfg = json.loads((p3 / "config.json").read_text())
    complete = json.loads((p3 / "summary.json").read_text())
    if complete["status"] != "complete":
        raise ValueError("P3 must be complete")
    rows, manifests = load_sources([Path(p) for p in cfg["manifests"]])
    models = {
        "identity": (args.identity.resolve(), "model"),
        "old-match-edge": (args.baseline.resolve(), "match-edge"),
        "old-model": (args.baseline.resolve(), "model"),
    }
    for arm in ("control", "excess_identity"):
        for seed in cfg["seeds"]:
            name = f"{arm}-{seed}"
            path = p3 / name / "best.onnx"
            receipt = next(
                r for r in complete["completed"] if r["arm"] == arm and r["seed"] == seed
            )
            if sha256_file(path) != receipt["onnx_sha256"]:
                raise ValueError("P3 receipt mismatch")
            models[name] = (path, "model")
    groups = {
        "development": [{**s, "length": 32768} for s in cfg["validation_specs"]],
        "heldout_review": [{**s, "start": 0, "length": None} for s in cfg["heldout_review_specs"]],
    }
    binary = freeze_binary(args.binary, root / "soundex-stream-eval", sha256_file(args.binary))
    registration = {
        "models": {
            k: {"path": str(p), "sha256": sha256_file(p), "gain": g} for k, (p, g) in models.items()
        },
        "limiter_modes": ["legacy", "smooth-knee"],
        "binary_sha256": sha256_file(binary),
        "study_config_sha256": sha256_file(p3 / "config.json"),
        "manifests": manifests,
        "groups": groups,
        "scored_region": "start 72*128, trim final 256; full prepared files are not full songs",
        "script_sha256": sha256_file(__file__),
        "bridge_sha256": sha256_file(Path(__file__).parents[1] / "evaluation/rust_bridge.py"),
        "metrics_sha256": sha256_file(Path(__file__).parents[1] / "evaluation/metrics.py"),
        "independent_test": False,
        "listening_performed": False,
        "promoted": False,
    }
    registered = root / "registration.json"
    if registered.exists() and json.loads(registered.read_text()) != registration:
        raise ValueError("registered ablation changed")
    write_json_report(registration, registered)
    cache = {}
    for group, specs in groups.items():
        for spec in specs:
            row = rows[spec["row_id"]]
            if group == "heldout_review" and row["split"] != "test":
                raise ValueError("review recording is not held out")
            waves = []
            for role in ("clean", "degraded"):
                path = Path(row["_root"]) / row[role + "_path"]
                if sha256_file(path) != row[role + "_checksum"]:
                    raise ValueError("audio checksum changed")
                wave, rate = sf.read(path, dtype="float32")
                end = None if spec["length"] is None else spec["start"] + spec["length"]
                waves.append(wave[spec["start"] : end])
            cache[group, spec["row_id"]] = (*waves, rate)

    def score(task):
        name, mode, group = task
        path = root / f"{name}-{mode}-{group}.json"
        if path.exists():
            return str(path)
        model, gain = models[name]
        ev = RustStreamEvaluator(binary)
        scored = []
        maximum_peak = 0.0
        for spec in groups[group]:
            row = rows[spec["row_id"]]
            clean, degraded, rate = cache[group, spec["row_id"]]
            output = ev.process(
                model, degraded, rate, mode="offline", high_band_gain=gain, limiter_mode=mode
            ).audio[:, 0]
            maximum_peak = max(maximum_peak, float(np.abs(output).max()))
            if not np.isfinite(output).all() or maximum_peak > 0.950001:
                raise ValueError("output safety failed")
            region = slice(72 * 128, len(clean) - 256)
            metrics = evaluate_signal_triplet(
                clean[region],
                degraded[region],
                output[region],
                sample_rate=rate,
                cutoff_hz=row["measured_cutoff_hz"],
            )
            record = _metric_row(row, metrics)
            record["genre"] = row.get("source_metadata", {}).get("genre", "unlabeled")
            scored.append(record)
        report = {
            "model_sha256": sha256_file(model),
            "limiter_mode": mode,
            "gain": gain,
            "rows": scored,
            "peak": maximum_peak,
            "aggregates": aggregate_metric_rows(
                scored,
                samples=2000,
                confidence=0.95,
                seed=20261012,
                cluster_field="track_id",
                dimensions=("corpus", "codec_id", "sample_rate", "channel_role", "genre"),
            ),
            "independent_test": False,
            "listening_performed": False,
            "promoted": False,
        }
        write_json_report(report, path)
        print(f"scored {name} {mode} {group}", flush=True)
        return str(path)

    tasks = [(n, m, g) for n in models for m in registration["limiter_modes"] for g in groups]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(score, tasks))
    write_json_report(
        {"status": "complete", "reports": reports, "promoted": False}, root / "summary.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "binary", "baseline", "identity"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
