"""Validate artifact-bound, cross-hardware SoundEx performance evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import onnx

from artifact_contract import (
    DEFAULT_FFT_SIZE,
    DEFAULT_HOP_SIZE,
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    frame_contract_from_metadata,
)
from evaluation.reporting import sha256_file

_REQUIRED_CASES = {(44_100, 1), (44_100, 2), (48_000, 1), (48_000, 2)}
_MINIMUM_STRESS_SECONDS = 30 * 60
_MAXIMUM_ALGORITHMIC_LATENCY_SAMPLES = 512
_MAXIMUM_STEREO_P99_US = 5_000.0
_MINIMUM_PRODUCTION_MODEL_BYTES = 1_000_000
_MAXIMUM_PRODUCTION_MODEL_BYTES = 8 * 1024 * 1024
_MAXIMUM_PEAK_RSS_BYTES = 50 * 1024 * 1024


def check_performance_reports(
    *, artifact_path: str | Path, report_paths: Iterable[str | Path]
) -> dict[str, Any]:
    """Validate raw benchmark evidence on Apple Silicon and x86_64."""
    artifact = Path(artifact_path).resolve()
    artifact_sha256 = sha256_file(artifact)
    artifact_size = artifact.stat().st_size
    if artifact_size < _MINIMUM_PRODUCTION_MODEL_BYTES:
        raise ValueError("performance evidence requires a non-trivial production artifact")
    if artifact_size >= _MAXIMUM_PRODUCTION_MODEL_BYTES:
        raise ValueError("performance evidence requires an FP32 artifact smaller than 8 MiB")
    fft_size, hop_size = _load_artifact_frame_contract(artifact)
    if (fft_size, hop_size) != (DEFAULT_FFT_SIZE, DEFAULT_HOP_SIZE):
        raise ValueError(
            "release performance evidence requires the 1024/512 low-latency artifact contract"
        )
    paths = [Path(path).resolve() for path in report_paths]
    if len(paths) < 2:
        raise ValueError("performance evidence requires Apple Silicon and x86_64 reports")

    reports: list[dict[str, Any]] = []
    report_hashes: dict[str, str] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"{path.name}: performance report must be a JSON object")
        _check_report(
            report,
            artifact_sha256=artifact_sha256,
            artifact_size=artifact_size,
            fft_size=fft_size,
            hop_size=hop_size,
            label=path.name,
        )
        reports.append(report)
        report_hashes[str(path)] = sha256_file(path)

    hardware_labels = [str(report["hardware"]["label"]) for report in reports]
    if len(set(hardware_labels)) != len(hardware_labels):
        raise ValueError("performance reports must name distinct reference hardware")
    has_apple_silicon = any(
        report["hardware"]["operating_system"] == "macos"
        and report["hardware"]["architecture"] in {"aarch64", "arm64"}
        for report in reports
    )
    has_x86_64 = any(
        report["hardware"]["architecture"] in {"x86_64", "amd64"} for report in reports
    )
    if not has_apple_silicon:
        raise ValueError("performance evidence lacks an Apple Silicon report")
    if not has_x86_64:
        raise ValueError("performance evidence lacks an x86_64 report")

    return {
        "artifact_sha256": artifact_sha256,
        "performance_report_sha256": report_hashes,
        "hardware": hardware_labels,
        "protocol": {"fft_size": fft_size, "hop_size": hop_size},
    }


def _check_report(
    report: dict[str, Any],
    *,
    artifact_sha256: str,
    artifact_size: int,
    fft_size: int,
    hop_size: int,
    label: str,
) -> None:
    if report.get("schema_version") != 2 or report.get("report_type") != "soundex-performance":
        raise ValueError(f"{label}: unsupported performance report schema")
    if report.get("diagnostic") is not False:
        raise ValueError(f"{label}: diagnostic benchmark cannot be release evidence")
    artifact = report.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError(f"{label}: artifact evidence is malformed")
    if artifact.get("sha256") != artifact_sha256:
        raise ValueError(f"{label}: benchmark artifact hash does not match ONNX file")
    if _required_int(artifact, "size_bytes", label=f"{label}:artifact", minimum=1) != artifact_size:
        raise ValueError(f"{label}: benchmark artifact size does not match ONNX file")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{label}: benchmark frame protocol is missing")
    reported_fft_size = _required_int(protocol, "fft_size", label=label, minimum=2)
    reported_hop_size = _required_int(protocol, "hop_size", label=label, minimum=1)
    reported_bins = _required_int(protocol, "frequency_bins", label=label, minimum=2)
    if (reported_fft_size, reported_hop_size) != (fft_size, hop_size):
        raise ValueError(f"{label}: benchmark frame protocol does not match ONNX metadata")
    if reported_bins != fft_size // 2 + 1:
        raise ValueError(f"{label}: benchmark frequency-bin count does not match FFT size")
    gates = report.get("performance_gates", {})
    if not isinstance(gates, dict) or gates.get("passed") is not True:
        failed = gates.get("failed", []) if isinstance(gates, dict) else []
        detail = (
            ", ".join(failed)
            if isinstance(failed, list) and all(isinstance(item, str) for item in failed)
            else "malformed failure list"
        )
        raise ValueError(f"{label}: performance gates failed: {detail}")
    if gates.get("failed") != []:
        raise ValueError(f"{label}: passing performance gates contain failures")
    _check_thresholds(gates.get("thresholds"), label=label)

    hardware = report.get("hardware", {})
    if not isinstance(hardware, dict):
        raise ValueError(f"{label}: hardware evidence is malformed")
    for field in (
        "label",
        "cpu",
        "architecture",
        "operating_system",
        "host",
        "power_mode",
        "rustc",
    ):
        value = hardware.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip() in {"unknown", "not-recorded"}
        ):
            raise ValueError(f"{label}: hardware field {field!r} is not recorded")
    ort = report.get("ort", {})
    if not isinstance(ort, dict):
        raise ValueError(f"{label}: ORT evidence is malformed")
    _required_int(ort, "intra_threads", label=f"{label}:ORT", minimum=1)
    _required_int(ort, "inter_threads", label=f"{label}:ORT", minimum=1)
    if not isinstance(ort.get("parallel_execution"), bool):
        raise ValueError(f"{label}: ORT execution mode is not explicit")
    if ort.get("graph_optimization") != "level3" or ort.get("output_preallocation") is not True:
        raise ValueError(f"{label}: ORT optimization/preallocation settings are incomplete")
    _required_int(report, "generated_unix_seconds", label=label, minimum=1)
    _required_float(report, "cold_model_load_ms", label=label, positive=True)

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label}: benchmark cases are missing")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    case_rss_values: list[int] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{label}: benchmark case is malformed")
        key = (
            _required_int(case, "sample_rate_hz", label=label, minimum=1),
            _required_int(case, "channels", label=label, minimum=1),
        )
        if key in indexed:
            raise ValueError(f"{label}: duplicate benchmark case {key}")
        indexed[key] = case
        case_rss_values.append(
            _check_case(
                case,
                fft_size=fft_size,
                hop_size=hop_size,
                label=f"{label}:{key[0]}Hz/{key[1]}ch",
            )
        )
    if set(indexed) != _REQUIRED_CASES:
        raise ValueError(f"{label}: benchmark cases must be exactly {sorted(_REQUIRED_CASES)}")
    peak_rss = _required_int(report, "peak_rss_bytes", label=label, minimum=1)
    if peak_rss != max(case_rss_values):
        raise ValueError(f"{label}: peak RSS does not match case evidence")
    if peak_rss > _MAXIMUM_PEAK_RSS_BYTES:
        raise ValueError(f"{label}: peak RSS exceeds 50 MiB")


def _check_case(case: dict[str, Any], *, fft_size: int, hop_size: int, label: str) -> int:
    sample_rate = _required_int(case, "sample_rate_hz", label=label, minimum=1)
    channels = _required_int(case, "channels", label=label, minimum=1)
    configured_seconds = _required_int(
        case,
        "configured_measurement_seconds",
        label=label,
        minimum=1,
    )
    if configured_seconds < _MINIMUM_STRESS_SECONDS:
        raise ValueError(f"{label}: stress duration is shorter than 30 minutes")
    warmup_hops = _required_int(case, "warmup_hops", label=label, minimum=1)
    algorithmic_latency = _required_int(
        case,
        "algorithmic_latency_samples_per_channel",
        label=label,
        minimum=1,
    )
    if not 0 < algorithmic_latency <= _MAXIMUM_ALGORITHMIC_LATENCY_SAMPLES:
        raise ValueError(f"{label}: algorithmic latency exceeds 512 samples")
    if algorithmic_latency != fft_size - hop_size:
        raise ValueError(f"{label}: algorithmic latency does not match frame protocol")
    algorithmic_latency_ms = _required_float(
        case, "algorithmic_latency_ms", label=label, positive=True
    )
    _assert_close(
        algorithmic_latency_ms,
        algorithmic_latency / sample_rate * 1_000.0,
        label=f"{label}: algorithmic_latency_ms",
    )
    _required_float(case, "processor_construction_ms", label=label, positive=True)
    _required_float(case, "first_frame_us", label=label, positive=True)
    measurement_wall_seconds = _required_float(
        case, "measurement_wall_seconds", label=label, positive=True
    )
    if measurement_wall_seconds < configured_seconds:
        raise ValueError(f"{label}: measured wall duration is shorter than configured duration")
    if _required_int(case, "nonfinite_output_samples", label=label, minimum=0) != 0:
        raise ValueError(f"{label}: non-finite output was observed")
    rss = _required_int(case, "rss_bytes", label=label, minimum=1)
    if rss > _MAXIMUM_PEAK_RSS_BYTES:
        raise ValueError(f"{label}: RSS exceeds 50 MiB")
    if not isinstance(case.get("rss_measurement"), str) or not case["rss_measurement"].strip():
        raise ValueError(f"{label}: RSS measurement method was not recorded")

    samples = case.get("samples_ns")
    measured_hops = _required_int(case, "measured_hops", label=label, minimum=1)
    if (
        not isinstance(samples, list)
        or len(samples) != measured_hops
        or not samples
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in samples
        )
    ):
        raise ValueError(f"{label}: raw latency samples are incomplete")
    processing_seconds = _required_float(case, "processing_seconds", label=label, positive=True)
    recomputed_processing_seconds = sum(samples) / 1_000_000_000.0
    _assert_close(
        processing_seconds,
        recomputed_processing_seconds,
        label=f"{label}: processing_seconds",
    )
    if processing_seconds > measurement_wall_seconds + 1e-9:
        raise ValueError(f"{label}: summed processing time exceeds measured wall duration")

    audio_seconds = _required_float(case, "audio_seconds", label=label, positive=True)
    recomputed_audio_seconds = measured_hops * hop_size / sample_rate
    _assert_close(audio_seconds, recomputed_audio_seconds, label=f"{label}: audio_seconds")
    if audio_seconds < configured_seconds:
        raise ValueError(f"{label}: raw hop count covers less than the configured stress duration")

    rtf = _required_float(case, "real_time_factor", label=label, positive=True)
    recomputed_rtf = processing_seconds / audio_seconds
    _assert_close(rtf, recomputed_rtf, label=f"{label}: real_time_factor")
    if rtf >= 1.0:
        raise ValueError(f"{label}: real-time factor must be below one")
    throughput = _required_float(case, "throughput_x_realtime", label=label, positive=True)
    _assert_close(throughput, 1.0 / rtf, label=f"{label}: throughput_x_realtime")

    deadline_us = _required_float(case, "deadline_us", label=label, positive=True)
    _assert_close(
        deadline_us,
        hop_size / sample_rate * 1_000_000.0,
        label=f"{label}: deadline_us",
        absolute_tolerance=1e-3,
    )
    deadline_misses = _required_int(case, "deadline_misses", label=label, minimum=0)
    recomputed_deadline_misses = sum(value / 1_000.0 > deadline_us for value in samples)
    if deadline_misses != recomputed_deadline_misses:
        raise ValueError(f"{label}: deadline misses do not match raw samples")
    if deadline_misses != 0:
        raise ValueError(f"{label}: deadline misses are nonzero")

    session_runs = _required_int(case, "session_runs", label=label, minimum=0)
    expected_session_runs = _required_int(case, "expected_session_runs", label=label, minimum=0)
    derived_session_runs = 1 + warmup_hops + measured_hops
    if expected_session_runs != derived_session_runs or session_runs != expected_session_runs:
        raise ValueError(f"{label}: session run count does not match processed hops")

    sorted_samples = sorted(samples)
    recomputed = {
        "p50_us": _percentile(sorted_samples, 0.50) / 1_000.0,
        "p95_us": _percentile(sorted_samples, 0.95) / 1_000.0,
        "p99_us": _percentile(sorted_samples, 0.99) / 1_000.0,
        "max_us": sorted_samples[-1] / 1_000.0,
    }
    latency = case.get("latency", {})
    if not isinstance(latency, dict):
        raise ValueError(f"{label}: latency summary is malformed")
    for metric, expected in recomputed.items():
        actual = _required_float(latency, metric, label=label, minimum=0.0)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label}: {metric} does not match raw samples")
    if channels == 2 and recomputed["p99_us"] >= _MAXIMUM_STEREO_P99_US:
        raise ValueError(f"{label}: stereo p99 is not below 5 ms")
    return rss


def _load_artifact_frame_contract(artifact: Path) -> tuple[int, int]:
    try:
        graph = onnx.load(str(artifact), load_external_data=False)
    except Exception as error:
        raise ValueError("performance artifact is not a readable ONNX model") from error
    metadata = {item.key: item.value for item in graph.metadata_props}
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ValueError(f"performance artifact metadata is missing {missing[0]}")
    try:
        return frame_contract_from_metadata(metadata)
    except ExportValidationError as error:
        raise ValueError(f"performance artifact frame contract is invalid: {error}") from error


def _check_thresholds(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: performance gate thresholds are missing")
    expected = {
        "maximum_algorithmic_latency_samples": _MAXIMUM_ALGORITHMIC_LATENCY_SAMPLES,
        "maximum_stereo_p99_us": _MAXIMUM_STEREO_P99_US,
        "maximum_deadline_misses": 0,
        "maximum_real_time_factor": 1.0,
        "maximum_peak_rss_bytes": _MAXIMUM_PEAK_RSS_BYTES,
        "minimum_warmup_hops": 1,
        "minimum_measurement_seconds_per_case": _MINIMUM_STRESS_SECONDS,
    }
    if set(value) != set(expected):
        raise ValueError(f"{label}: performance gate threshold fields are incomplete")
    for field, expected_value in expected.items():
        actual = value[field]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{label}: performance gate threshold {field!r} is invalid")
        if float(actual) != float(expected_value):
            raise ValueError(f"{label}: performance gate threshold {field!r} has drifted")


def _required_int(value: dict[str, Any], field: str, *, label: str, minimum: int) -> int:
    actual = value.get(field)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < minimum:
        raise ValueError(f"{label}: integer field {field!r} is invalid or missing")
    return actual


def _required_float(
    value: dict[str, Any],
    field: str,
    *,
    label: str,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    actual = value.get(field)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label}: numeric field {field!r} is invalid or missing")
    number = float(actual)
    if not math.isfinite(number):
        raise ValueError(f"{label}: numeric field {field!r} is not finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label}: numeric field {field!r} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}: numeric field {field!r} is below its minimum")
    return number


def _assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
    absolute_tolerance: float = 1e-9,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=absolute_tolerance):
        raise ValueError(f"{label} does not match raw evidence")


def _percentile(sorted_values: list[int], percentile: float) -> int:
    rank = math.ceil(percentile * len(sorted_values))
    return sorted_values[max(0, min(rank - 1, len(sorted_values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("performance_reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        result = check_performance_reports(
            artifact_path=args.artifact,
            report_paths=args.performance_reports,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL performance-evidence: {error}")
        raise SystemExit(2) from error
    print(
        "PASS performance-evidence: "
        f"artifact={result['artifact_sha256']} hardware={', '.join(result['hardware'])}"
    )


if __name__ == "__main__":
    main()
