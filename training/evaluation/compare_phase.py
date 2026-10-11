"""Same-row phase-only ablations through the complete Rust production stream."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluation.compare_extension import selected_rows
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from evaluation.runner import _load_unit_audio, _metric_row
from evaluation.rust_bridge import RustStreamEvaluator

POLICIES = ("model", "observed", "surviving-power")


def compare(model: Path, rows: list[dict], evaluator: RustStreamEvaluator, workers: int) -> dict:
    """Fix ONNX, gains and detector; vary only within-frame phase selection."""

    def score(row: dict) -> dict:
        clean, degraded = _load_unit_audio([row])
        rate = int(row["sample_rate"])
        result = {}
        for mode in POLICIES:
            offline = evaluator.process(model, degraded, rate, mode="offline", phase_policy=mode)
            metrics = evaluate_signal_triplet(
                clean[:, 0],
                degraded[:, 0],
                offline.audio[:, 0],
                sample_rate=rate,
                cutoff_hz=float(row["measured_cutoff_hz"]),
            )
            # Startup/tail exclusion is reported separately, not used to hide failures.
            start, end = int(rate * 0.25), len(clean) - 256
            if end - start < 256:
                raise ValueError("phase comparison needs 0.25 seconds warmup plus 512 samples")
            steady = evaluate_signal_triplet(
                clean[start:end, 0],
                degraded[start:end, 0],
                offline.audio[start:end, 0],
                sample_rate=rate,
                cutoff_hz=float(row["measured_cutoff_hz"]),
            )
            metrics.update({"steady_" + key: value for key, value in steady.items()})
            errors = []
            for pattern in ((128,), (31, 512, 997, 7)):
                chunked = evaluator.process(
                    model,
                    degraded,
                    rate,
                    mode="chunked",
                    chunk_frames=pattern,
                    phase_policy=mode,
                )
                errors.append(float(np.max(np.abs(chunked.audio - offline.audio))))
            metrics.update(
                chunk_max_abs_error=max(errors),
                nonfinite_output=float(not np.isfinite(offline.audio).all()),
                output_peak=float(np.max(np.abs(offline.audio))),
            )
            result[mode] = {
                **_metric_row(row, metrics),
                "genre": row.get("source_metadata", {}).get(
                    "genre", "speech" if row["corpus"] == "speech_library" else "unlabeled"
                ),
            }
        return result

    metric_rows = {mode: [] for mode in POLICIES}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(score, rows), 1):
            for mode in POLICIES:
                metric_rows[mode].append(result[mode])
            if number % 10 == 0:
                print(f"Completed {number}/{len(rows)}", flush=True)
    options = dict(
        samples=2000,
        confidence=0.95,
        seed=1592594996,
        cluster_field="track_id",
        dimensions=("corpus", "codec_id", "sample_rate", "channel_role", "genre"),
    )
    base = {row["row_id"]: row for row in metric_rows["model"]}
    paired = {}
    for mode in POLICIES[1:]:
        changes = [
            {
                **row,
                "metrics": {
                    key: value - base[row["row_id"]]["metrics"][key]
                    for key, value in row["metrics"].items()
                },
            }
            for row in metric_rows[mode]
        ]
        paired[mode] = aggregate_metric_rows(changes, **options)
    return {
        "diagnostic": True,
        "formal_release_evaluation": False,
        "listening_performed": False,
        "independent_test": False,
        "same_row_ids": True,
        "phase_policies": list(POLICIES),
        "enhancement_mode": "neural",
        "high_band_gain": "match-edge",
        "gain_calibration": None,
        "model_sha256": sha256_file(model),
        "rust_binary_sha256": sha256_file(evaluator.binary),
        "recipe_sha256": rows[0]["recipe_hash"],
        "rows": metric_rows,
        "aggregates": {
            mode: aggregate_metric_rows(values, **options) for mode, values in metric_rows.items()
        },
        "paired_changes_vs_model": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    rows = selected_rows(args.manifest, args.selection)
    evaluator = RustStreamEvaluator(args.binary)
    evaluator.ensure_available()
    report = compare(args.model, rows, evaluator, args.workers)
    report.update(
        selection_sha256=sha256_file(args.selection),
        selection=json.loads(args.selection.read_text()),
        manifests={str(path): sha256_file(path) for path in args.manifest},
    )
    write_json_report(report, args.output)


if __name__ == "__main__":
    main()
