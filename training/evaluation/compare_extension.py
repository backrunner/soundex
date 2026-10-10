"""Same-row validation diagnostic for neural, DSP and hybrid production paths."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data.protocol import load_manifest, validate_manifest_rows
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from evaluation.runner import _load_unit_audio, _metric_row
from evaluation.rust_bridge import RustStreamEvaluator

MODES = ("neural", "spectral", "hybrid")


def selected_rows(manifests: list[Path], selection: Path) -> list[dict]:
    """Require an explicit, unchanged validation selection; audit only its PCM."""
    requested = json.loads(selection.read_text(encoding="utf-8"))
    ids = requested.get("row_ids", [])
    if requested.get("split") != "validation" or not ids or len(ids) != len(set(ids)):
        raise ValueError("selection must contain unique validation row_ids")
    by_id: dict[str, dict] = {}
    training_tracks: set[str] = set()
    for path in manifests:
        rows = load_manifest(path)
        validate_manifest_rows(rows, path.parent, audit_files=False)
        training_tracks.update(str(row["track_id"]) for row in rows if row["split"] == "train")
        chosen = [row for row in rows if row["row_id"] in ids]
        if chosen:
            validate_manifest_rows(chosen, path.parent, audit_files=True)
        for row in chosen:
            if row["row_id"] in by_id or row["split"] != "validation":
                raise ValueError("selection contains duplicate or non-validation membership")
            by_id[row["row_id"]] = {**row, "_root": str(path.parent.resolve())}
    if set(by_id) != set(ids):
        raise ValueError("not all selected rows are present in the manifests")
    rows = [by_id[row_id] for row_id in ids]
    if any(str(row["track_id"]) in training_tracks for row in rows):
        raise ValueError("validation recording overlaps training")
    if len({row["recipe_hash"] for row in rows}) != 1:
        raise ValueError("selected rows must share one preprocessing recipe")
    return rows


def compare(model: Path, rows: list[dict], evaluator: RustStreamEvaluator, workers: int) -> dict:
    """Compare the same ONNX with each candidate path, never claim release acceptance."""

    def score(row: dict) -> dict:
        clean, degraded = _load_unit_audio([row])
        rate = int(row["sample_rate"])
        result = {}
        for mode in MODES:
            offline = evaluator.process(
                model, degraded, rate, mode="offline", enhancement_mode=mode
            )
            metrics = evaluate_signal_triplet(
                clean[:, 0],
                degraded[:, 0],
                offline.audio[:, 0],
                sample_rate=rate,
                cutoff_hz=float(row["measured_cutoff_hz"]),
            )
            errors = []
            for pattern in ((128,), (31, 512, 997, 7)):
                chunked = evaluator.process(
                    model,
                    degraded,
                    rate,
                    mode="chunked",
                    chunk_frames=pattern,
                    enhancement_mode=mode,
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

    metric_rows = {mode: [] for mode in MODES}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(score, rows), 1):
            for mode in MODES:
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
    base = {row["row_id"]: row for row in metric_rows["neural"]}
    paired = {}
    for mode in MODES[1:]:
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
        "enhancement_modes": list(MODES),
        "model_sha256": sha256_file(model),
        "rust_binary_sha256": sha256_file(evaluator.binary),
        "recipe_sha256": rows[0]["recipe_hash"],
        "rows": metric_rows,
        "aggregates": {
            mode: aggregate_metric_rows(values, **options) for mode, values in metric_rows.items()
        },
        "paired_changes_vs_neural": paired,
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
