#!/usr/bin/env python3
"""Aggregate registered read-only P14 diagnostics; retain every corpus and case."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite audit result")
    return {
        "count": len(values),
        "mean": float(array.mean()),
        **{
            k: float(np.quantile(array, q))
            for k, q in (("min", 0), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1))
        },
    }


def run(root):
    receipt = json.loads((root / "summary.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("audit incomplete")
    for name, sha in receipt["reports"].items():
        if sha256_file(root / name) != sha:
            raise ValueError("registered audit changed")
    results = []
    for name in receipt["reports"]:
        if name == "registration.json":
            continue
        report = json.loads((root / name).read_text())
        for corpus in ("music_library", "speech_library"):
            rows = [r for r in report["records"] if r["corpus"] == corpus]
            records = {
                "model": report["model"],
                "group": report["group"],
                "corpus": corpus,
                "pairs": len(rows),
                "recordings": len({r["track_id"] for r in rows}),
                "parameter_gradients": {
                    key: distribution([r["parameter_gradients"][key] for r in rows])
                    for key in rows[0]["parameter_gradients"]
                },
                "roles": {},
            }
            for i, role in enumerate(("paired", "identity")):
                examples = [r["roles"][i] for r in rows]
                record = {
                    key: distribution([r[key] for r in examples])
                    for key in (
                        "loss",
                        "waveform_loss",
                        "raw_amplitude_gradient_l2",
                        "input_frame_relative_gradient_l2",
                    )
                }
                record["bands"] = {}
                for index in range(5):
                    bands = [r["rendered"][index] for r in examples]
                    record["bands"][bands[0]["band"]] = {
                        key: distribution([r[key] for r in bands])
                        for key in (
                            "reference_rms",
                            "output_rms",
                            "error_rms",
                            "reference_to_global",
                            "error_to_global",
                            "floor_dominated",
                            "amplitude_loss",
                            "dynamics_loss",
                            "total",
                            "amplitude_coordinate_gradient_l2",
                            "clip_relative_coordinate_gradient_l2",
                        )
                    }
                records["roles"][role] = record
            results.append(records)
    write_json_report(
        {
            "scope": "post-hoc diagnostics, no training or new acceptance score",
            "audit_receipt_sha256": sha256_file(root / "summary.json"),
            "analysis_sha256": sha256_file(Path(__file__)),
            "results": results,
        },
        root / "analysis.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    run(parser.parse_args().audit)
