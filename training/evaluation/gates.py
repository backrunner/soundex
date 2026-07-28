"""Versioned, machine-checkable SoundEx release quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

GATE_SCHEMA_VERSION = 1
_REQUIRED_STEREO_METRICS = {
    "baseline_correlation_error",
    "enhanced_correlation_error",
    "baseline_image_width_error_db",
    "enhanced_image_width_error_db",
    "baseline_interchannel_phase_error_rad",
    "enhanced_interchannel_phase_error_rad",
}


def load_gate_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the versioned gate configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != GATE_SCHEMA_VERSION:
        raise ValueError("unsupported release-gate schema")
    required = {
        "bootstrap",
        "thresholds",
        "required_stratum_dimensions",
        "required_evidence",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"release-gate config is missing {missing[0]}")
    return config


def evaluate_release_gates(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all gates and return named results without hiding failures."""
    if int(report.get("schema_version", 0)) != 2:
        raise ValueError("unsupported evaluation report schema")
    thresholds = config["thresholds"]
    overall = report["aggregates"]["overall"]["metrics"]
    rows = report["rows"]
    results: list[dict[str, Any]] = []

    high_delta = _metric(overall, "delta_high_lsd_db")
    _add(
        results,
        "mean_high_band_improvement",
        high_delta["mean"] < float(thresholds["mean_high_band_delta_db_max"]),
        f"mean enhanced-baseline high-band LSD = {high_delta['mean']:.6g} dB",
    )
    _add(
        results,
        "high_band_improvement_ci",
        high_delta["ci95"][1] < float(thresholds["high_band_delta_ci_upper_db_max"]),
        f"95% CI upper bound = {high_delta['ci95'][1]:.6g} dB",
    )

    regressions: list[str] = []
    for dimension in config["required_stratum_dimensions"]:
        groups = report["aggregates"]["strata"].get(str(dimension), {})
        if not groups:
            regressions.append(f"{dimension}:missing")
            continue
        for name, aggregate in groups.items():
            metric = aggregate["metrics"].get("delta_high_lsd_db")
            if metric is None or metric["ci95"][0] > 0.0:
                regressions.append(f"{dimension}:{name}")
    _add(
        results,
        "required_strata_no_supported_regression",
        not regressions,
        "none" if not regressions else ", ".join(regressions),
    )

    low_preservation = _metric(overall, "low_band_preservation_lsd_db")
    _add(
        results,
        "low_band_preservation",
        low_preservation["mean"] <= float(thresholds["low_band_preservation_mean_db_max"]),
        f"mean low-band enhanced-vs-degraded LSD = {low_preservation['mean']:.6g} dB",
    )
    chunk_maximum = max(float(row["metrics"]["chunk_max_abs_error"]) for row in rows)
    _add(
        results,
        "stream_chunk_equivalence",
        chunk_maximum <= float(thresholds["stream_equivalence_max_abs"]),
        f"maximum offline-vs-chunked difference = {chunk_maximum:.6g}",
    )
    clipped = sum(int(row["metrics"]["clipped_samples"]) for row in rows)
    nonfinite = sum(int(row["metrics"]["nonfinite_output"]) for row in rows)
    _add(
        results,
        "finite_unclipped_output",
        clipped <= int(thresholds["clipped_samples_max"])
        and nonfinite <= int(thresholds["nonfinite_rows_max"]),
        f"clipped samples = {clipped}, non-finite rows = {nonfinite}",
    )
    false_bypass_rate = math.fsum(float(row["metrics"]["gate_false_bypass"]) for row in rows) / len(
        rows
    )
    _add(
        results,
        "gate_false_bypass",
        false_bypass_rate <= float(thresholds["gate_false_bypass_rate_max"]),
        f"known-degradation false-bypass rate = {false_bypass_rate:.6g}",
    )

    parity = report.get("parity_evidence", {})
    parity_required = bool(config["required_evidence"].get("pytorch_ort_rust_parity", True))
    parity_matches = (
        int(parity.get("schema_version", 0)) == 1
        and parity.get("suite") == "pytorch-ort-rust-v1"
        and bool(parity.get("passed", False))
        and parity.get("artifact_sha256") == report["artifact"]["sha256"]
    )
    _add(
        results,
        "pytorch_ort_rust_parity",
        not parity_required or parity_matches,
        "matching pass evidence"
        if parity_matches
        else "missing, failed, or artifact hash mismatch",
    )

    visqol_required = bool(config["required_evidence"].get("visqol_audio", True))
    perceptual = report.get("perceptual", {})
    visqol_metric = overall.get("delta_visqol")
    visqol_passed = (
        perceptual.get("status") == "available"
        and visqol_metric is not None
        and int(visqol_metric["count"]) == int(perceptual.get("required_rows", -1))
        and visqol_metric["mean"] >= float(thresholds["visqol_mean_delta_min"])
    )
    _add(
        results,
        "visqol_audio",
        not visqol_required or visqol_passed,
        (
            f"status={perceptual.get('status', 'missing')}, "
            f"scored={0 if visqol_metric is None else visqol_metric['count']}, "
            f"required={perceptual.get('required_rows', 0)}"
        ),
    )
    audit_required = bool(config["required_evidence"].get("test_manifest_file_audit", True))
    files_audited = bool(report.get("protocol", {}).get("audio_files_audited", False))
    _add(
        results,
        "test_manifest_file_audit",
        not audit_required or files_audited,
        "checksums and audio headers audited" if files_audited else "file audit was skipped",
    )
    stereo_required = bool(config["required_evidence"].get("stereo_metrics", True))
    stereo_rows = report.get("stereo_rows", [])
    stereo_count = len(stereo_rows) if isinstance(stereo_rows, list) else 0
    complete_stereo_rows = 0
    if isinstance(stereo_rows, list):
        for row in stereo_rows:
            metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
            if (
                isinstance(metrics, dict)
                and set(metrics) >= _REQUIRED_STEREO_METRICS
                and all(math.isfinite(float(metrics[name])) for name in _REQUIRED_STEREO_METRICS)
            ):
                complete_stereo_rows += 1
    _add(
        results,
        "stereo_metrics_present",
        not stereo_required or (stereo_count > 0 and complete_stereo_rows == stereo_count),
        f"complete paired stereo rows = {complete_stereo_rows}/{stereo_count}",
    )
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "passed": all(result["passed"] for result in results),
        "failed": [result["name"] for result in results if not result["passed"]],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a SoundEx evaluation report")
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("release_gates.v1.yaml"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    gates = evaluate_release_gates(report, load_gate_config(args.config))
    for result in gates["results"]:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}: {result['detail']}")
    raise SystemExit(0 if gates["passed"] else 2)


def _metric(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in metrics:
        raise ValueError(f"evaluation aggregate is missing metric {name}")
    return metrics[name]


def _add(results: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    results.append({"name": name, "passed": bool(passed), "detail": detail})


if __name__ == "__main__":
    main()
