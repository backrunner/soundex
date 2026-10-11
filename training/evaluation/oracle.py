"""Clean-reference upper-bound diagnostics through the production Rust DSP."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluation.compare_extension import selected_rows
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import aggregate_metric_rows, sha256_file, write_json_report
from evaluation.runner import _load_unit_audio, _metric_row
from evaluation.rust_bridge import read_sxa, write_sxa


def evaluate_oracles(model: Path, binary: Path, rows: list[dict], workers: int) -> dict:
    """Evaluate both gain policies without selecting or training on oracle targets."""
    arms = [
        (mode, gain)
        for mode in ("observed", "magnitude", "phase", "complex")
        for gain in ("match-edge", "model")
    ]

    def score(row: dict) -> dict:
        clean, degraded = _load_unit_audio([row])
        rate = int(row["sample_rate"])
        scores = {}
        with tempfile.TemporaryDirectory(prefix="soundex-oracle-") as temporary:
            root = Path(temporary)
            write_sxa(root / "input.sxa", degraded, rate)
            write_sxa(root / "clean.sxa", clean, rate)
            for mode, gain in arms:
                subprocess.run(
                    [
                        str(binary),
                        str(model),
                        str(root / "input.sxa"),
                        str(root / "clean.sxa"),
                        str(root / "output.sxa"),
                        str(root / "trace.json"),
                        mode,
                        gain,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                audio, output_rate = read_sxa(root / "output.sxa")
                if output_rate != rate or audio.shape != clean.shape:
                    raise RuntimeError("oracle output alignment mismatch")
                trace = json.loads((root / "trace.json").read_text())
                if (
                    not trace.get("diagnostic_only")
                    or trace["oracle"] != mode
                    or trace["high_band_gain"] != gain
                ):
                    raise RuntimeError("oracle report policy mismatch")
                metrics = evaluate_signal_triplet(
                    clean[:, 0],
                    degraded[:, 0],
                    audio[:, 0],
                    sample_rate=rate,
                    cutoff_hz=float(row["measured_cutoff_hz"]),
                )
                start, stop = int(rate * 0.25), len(clean) - 256
                steady = evaluate_signal_triplet(
                    clean[start:stop, 0],
                    degraded[start:stop, 0],
                    audio[start:stop, 0],
                    sample_rate=rate,
                    cutoff_hz=float(row["measured_cutoff_hz"]),
                )
                metrics.update({"steady_" + key: value for key, value in steady.items()})
                active = [h for h in trace["hops"] if not h["bypassed"]]
                metrics.update(
                    oracle_active_ratio=len(active) / max(len(trace["hops"]), 1),
                    oracle_mean_gain_db=float(np.mean([h["gain_db"] for h in active]))
                    if active
                    else 0.0,
                    output_peak=float(np.abs(audio).max()),
                    max_change_from_input=float(np.abs(audio - degraded).max()),
                )
                scores[f"{mode}/{gain}"] = _metric_row(row, metrics)
        return scores

    result = {f"{mode}/{gain}": [] for mode, gain in arms}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, scores in enumerate(pool.map(score, rows), 1):
            for arm, value in scores.items():
                result[arm].append(value)
            if index % 10 == 0:
                print(f"Oracle {index}/{len(rows)}", flush=True)
    options = dict(
        samples=2000,
        confidence=0.95,
        seed=1592594996,
        cluster_field="track_id",
        dimensions=("corpus", "codec_id"),
    )
    deltas = [
        {
            **left,
            "metrics": {
                key: value - right["metrics"][key] for key, value in left["metrics"].items()
            },
        }
        for left, right in zip(result["complex/model"], result["complex/match-edge"], strict=True)
    ]
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "independent_test": False,
        "listening_performed": False,
        "model_sha256": sha256_file(model),
        "rust_binary_sha256": sha256_file(binary),
        "rows": result,
        "aggregates": {
            name: aggregate_metric_rows(values, **options) for name, values in result.items()
        },
        "complex_model_minus_match_edge": aggregate_metric_rows(deltas, **options),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = selected_rows(args.manifest, args.selection)
    report = evaluate_oracles(args.model.resolve(), args.binary.resolve(), rows, args.workers)
    report["selection_sha256"] = sha256_file(args.selection)
    report["manifests"] = [
        {"path": str(p.resolve()), "sha256": sha256_file(p)} for p in args.manifest
    ]
    write_json_report(report, args.output)


if __name__ == "__main__":
    main()
