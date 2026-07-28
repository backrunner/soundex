"""Deterministic bootstrap aggregation and evaluation report publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

AGGREGATE_DIMENSIONS = ("corpus", "codec_id", "quality", "sample_rate", "channel_role")


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the arithmetic mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap samples/confidence are invalid")
    if len(array) == 1:
        value = float(array[0])
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    confidence: float,
    seed: int,
    dimensions: Sequence[str] = AGGREGATE_DIMENSIONS,
    cluster_field: str | None = None,
) -> dict[str, Any]:
    """Aggregate overall and per-stratum metrics with deterministic CIs."""
    if not rows:
        raise ValueError("evaluation requires at least one metric row")
    ordered = sorted(rows, key=lambda row: str(row["row_id"]))
    row_ids = [str(row["row_id"]) for row in ordered]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("evaluation row IDs must be unique")
    overall = _aggregate_group(
        ordered,
        samples=samples,
        confidence=confidence,
        seed=seed,
        label="overall",
        cluster_field=cluster_field,
    )
    strata: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in ordered:
            if dimension not in row:
                raise ValueError(f"evaluation row {row['row_id']} lacks {dimension}")
            grouped.setdefault(str(row[dimension]), []).append(row)
        strata[dimension] = {
            name: _aggregate_group(
                group_rows,
                samples=samples,
                confidence=confidence,
                seed=seed,
                label=f"{dimension}:{name}",
                cluster_field=cluster_field,
            )
            for name, group_rows in sorted(grouped.items())
        }
    return {
        "row_count": len(ordered),
        "row_ids": row_ids,
        "bootstrap_unit": cluster_field or "row",
        "overall": overall,
        "strata": strata,
    }


def write_json_report(report: dict[str, Any], destination: str | Path) -> None:
    """Atomically write strict JSON without NaN or Infinity extensions."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(staging_name, path)
    finally:
        Path(staging_name).unlink(missing_ok=True)


def write_markdown_report(report: dict[str, Any], destination: str | Path) -> None:
    """Write a concise human-readable summary beside the machine report."""
    gates = report["release_gates"]
    overall = report["aggregates"]["overall"]["metrics"]
    lines = [
        "# SoundEx Deployment Evaluation",
        "",
        f"- Artifact SHA-256: `{report['artifact']['sha256']}`",
        f"- Test rows: {report['aggregates']['row_count']}",
        f"- Release gates: **{'PASS' if gates['passed'] else 'FAIL'}**",
        f"- ViSQOLAudio: `{report['perceptual']['status']}`",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Mean | 95% CI | Count |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "baseline_high_lsd_db",
        "enhanced_high_lsd_db",
        "delta_high_lsd_db",
        "low_band_preservation_lsd_db",
        "baseline_visqol",
        "enhanced_visqol",
        "delta_visqol",
        "chunk_max_abs_error",
    ):
        if name not in overall:
            continue
        metric = overall[name]
        lines.append(
            f"| `{name}` | {metric['mean']:.6g} | "
            f"[{metric['ci95'][0]:.6g}, {metric['ci95'][1]:.6g}] | {metric['count']} |"
        )
    lines.extend(["", "## Release Gates", "", "| Gate | Status | Detail |", "|---|---|---|"])
    for gate in gates["results"]:
        detail = str(gate["detail"]).replace("|", "\\|")
        lines.append(f"| `{gate['name']}` | {'PASS' if gate['passed'] else 'FAIL'} | {detail} |")
    lines.extend(
        [
            "",
            "## Listening Protocol",
            "",
            str(report["listening_protocol"]["summary"]),
            "",
        ]
    )
    Path(destination).write_text("\n".join(lines), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    """Hash a report or model artifact incrementally."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_group(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    confidence: float,
    seed: int,
    label: str,
    cluster_field: str | None,
) -> dict[str, Any]:
    if cluster_field is not None:
        missing = [str(row["row_id"]) for row in rows if cluster_field not in row]
        if missing:
            raise ValueError(
                f"evaluation row {missing[0]} lacks bootstrap cluster field {cluster_field}"
            )
    metric_names = sorted({str(name) for row in rows for name in row["metrics"]})
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [float(row["metrics"][name]) for row in rows if name in row["metrics"]]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"metric {name} contains non-finite values")
        metric_seed = _derived_seed(seed, f"{label}:{name}")
        if cluster_field is None:
            low, high = bootstrap_mean_ci(
                values,
                samples=samples,
                confidence=confidence,
                seed=metric_seed,
            )
            cluster_count = len(values)
        else:
            low, high, cluster_count = _cluster_bootstrap_mean_ci(
                rows,
                metric_name=name,
                cluster_field=cluster_field,
                samples=samples,
                confidence=confidence,
                seed=metric_seed,
            )
        metrics[name] = {
            "count": len(values),
            "cluster_count": cluster_count,
            "mean": math.fsum(values) / len(values),
            "ci95": [low, high],
        }
    group_cluster_count = (
        len(rows) if cluster_field is None else len({str(row[cluster_field]) for row in rows})
    )
    return {
        "row_count": len(rows),
        "cluster_count": group_cluster_count,
        "metrics": metrics,
    }


def _cluster_bootstrap_mean_ci(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    cluster_field: str,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, int]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        if metric_name in row["metrics"]:
            clusters.setdefault(str(row[cluster_field]), []).append(
                float(row["metrics"][metric_name])
            )
    ordered = [clusters[name] for name in sorted(clusters)]
    if not ordered:
        raise ValueError(f"metric {metric_name} has no bootstrap clusters")
    sums = np.asarray([math.fsum(values) for values in ordered], dtype=np.float64)
    counts = np.asarray([len(values) for values in ordered], dtype=np.int64)
    if len(ordered) == 1:
        mean = float(sums[0] / counts[0])
        return mean, mean, 1
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(ordered), size=(samples, len(ordered)))
    means = np.sum(sums[indices], axis=1) / np.sum(counts[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high), len(ordered)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
