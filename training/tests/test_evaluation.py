"""Metrics, Rust bridge, bootstrap, release-gate, and model-card tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

import evaluation.performance as performance
from evaluation.compare_extension import selected_rows
from evaluation.gates import evaluate_release_gates, load_gate_config
from evaluation.metrics import (
    _flux_error,
    evaluate_signal_triplet,
    evaluate_stereo_image,
    scale_invariant_sdr,
)
from evaluation.model_card import (
    _REQUIRED_FIELDS,
    _validate_artifact_metadata_binding,
    _validate_evaluation_evidence,
    _validate_external_evidence,
    check_release_model_card,
)
from evaluation.performance import check_performance_reports
from evaluation.reporting import aggregate_metric_rows, bootstrap_mean_ci, sha256_file
from evaluation.runner import _artifact_frame_contract, _evaluation_units
from evaluation.rust_bridge import RustStreamEvaluator, read_sxa, write_sxa
from evaluation.visqol import _parse_score
from parity_metrics import PARITY_POLICY, POLICY_SHA256, REQUIRED_PARITY_CASES

REPOSITORY = Path(__file__).resolve().parents[2]
GATE_CONFIG = Path(__file__).resolve().parents[1] / "evaluation" / "release_gates.v1.yaml"


@pytest.mark.parametrize(
    "selection",
    [
        {"split": "train", "row_ids": ["a"]},
        {"split": "test", "row_ids": ["a"]},
        {"split": "validation", "row_ids": []},
        {"split": "validation", "row_ids": ["a", "a"]},
    ],
)
def test_extension_diagnostic_requires_fixed_unique_validation_selection(
    tmp_path: Path, selection: dict
) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="unique validation row_ids"):
        selected_rows([], path)


def test_high_flux_distinguishes_attacks_from_constant_high_band_energy() -> None:
    reference = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]])
    high = np.array([False, True])
    assert _flux_error(reference, reference, high) == 0.0
    assert _flux_error(np.ones_like(reference), reference, high) == pytest.approx(1.0)
    assert _flux_error(np.zeros_like(reference), np.zeros_like(reference), high) == 0.0
    assert _flux_error(reference, np.zeros_like(reference), high) == 1e6


@pytest.mark.parametrize("enhancement_mode", ["spectral", "hybrid"])
def test_rust_extension_bridge_binds_mode_and_keeps_chunks_aligned(enhancement_mode: str) -> None:
    audio = np.sin(np.arange(5003, dtype=np.float32) * 0.17)[:, None] * 0.1
    evaluator = RustStreamEvaluator()
    model = REPOSITORY / "tests/fixtures/low-latency-identity.onnx"
    offline = evaluator.process(
        model, audio, 48000, mode="offline", enhancement_mode=enhancement_mode
    )
    chunked = evaluator.process(
        model,
        audio,
        48000,
        mode="chunked",
        chunk_frames=(31, 128, 997),
        enhancement_mode=enhancement_mode,
    )
    np.testing.assert_array_equal(offline.audio, chunked.audio)
    assert offline.report["enhancement_mode"] == enhancement_mode
    assert offline.report["processor"]["latency_samples_per_channel"] == 128


def test_identical_and_silent_metrics_have_finite_expected_values() -> None:
    sample_rate = 44_100
    time = np.arange(512, dtype=np.float64) / sample_rate
    signal = 0.2 * np.sin(2.0 * np.pi * 1000.0 * time)

    metrics = evaluate_signal_triplet(
        signal,
        signal,
        signal,
        sample_rate=sample_rate,
        cutoff_hz=10_000.0,
        fft_size=64,
        hop_size=16,
    )
    assert metrics["baseline_full_lsd_db"] == 0.0
    assert metrics["enhanced_high_lsd_db"] == 0.0
    assert metrics["low_band_preservation_lsd_db"] == 0.0
    assert metrics["enhanced_spectral_convergence"] == 0.0
    assert metrics["enhanced_high_band_energy_error_db"] == 0.0
    assert metrics["enhanced_si_sdr_db"] == 120.0

    silence = np.zeros(512)
    silent_metrics = evaluate_signal_triplet(
        silence,
        silence,
        silence,
        sample_rate=sample_rate,
        cutoff_hz=0.0,
        fft_size=64,
        hop_size=16,
    )
    assert all(np.isfinite(value) for value in silent_metrics.values())
    assert scale_invariant_sdr(silence, silence) == 0.0


def test_metrics_reject_unaligned_inputs_and_detect_high_band_recovery() -> None:
    sample_rate = 44_100
    time = np.arange(2048, dtype=np.float64) / sample_rate
    low = 0.2 * np.sin(2.0 * np.pi * 1000.0 * time)
    high = 0.1 * np.sin(2.0 * np.pi * 12_000.0 * time)
    clean = low + high

    metrics = evaluate_signal_triplet(
        clean,
        low,
        clean,
        sample_rate=sample_rate,
        cutoff_hz=8000.0,
        fft_size=256,
        hop_size=64,
    )
    assert metrics["delta_high_lsd_db"] < 0.0
    with pytest.raises(ValueError, match="exactly equal lengths"):
        evaluate_signal_triplet(
            clean,
            low[:-1],
            clean,
            sample_rate=sample_rate,
            cutoff_hz=8000.0,
            fft_size=256,
            hop_size=64,
        )


def test_stereo_metrics_are_zero_error_for_identical_triplet() -> None:
    time = np.arange(512, dtype=np.float64) / 44_100.0
    stereo = np.column_stack(
        (
            np.sin(2.0 * np.pi * 440.0 * time),
            np.sin(2.0 * np.pi * 440.0 * time + 0.2),
        )
    )

    metrics = evaluate_stereo_image(stereo, stereo, stereo, fft_size=64, hop_size=16)

    assert metrics["enhanced_correlation_error"] == 0.0
    assert metrics["enhanced_image_width_error_db"] == 0.0
    assert metrics["enhanced_interchannel_phase_error_rad"] == 0.0


def test_left_right_manifest_rows_share_one_stereo_evaluation_unit() -> None:
    common = {
        "_manifest": "/fixture/manifest.jsonl",
        "source_id": "song",
        "start_sample": 0,
        "length_samples": 4096,
        "codec_id": "mp3-cbr-64",
        "sample_rate": 44_100,
    }
    rows = [
        {**common, "row_id": "right", "channel_role": "right"},
        {**common, "row_id": "mid", "channel_role": "mid"},
        {**common, "row_id": "left", "channel_role": "left"},
    ]

    units = _evaluation_units(rows)

    assert [[row["channel_role"] for row in unit] for unit in units] == [
        ["left", "right"],
        ["mid"],
    ]


def test_bootstrap_and_aggregation_are_deterministic() -> None:
    assert bootstrap_mean_ci([2.0], samples=10, seed=1) == (2.0, 2.0)
    rows = [_quality_row("b", -2.0), _quality_row("a", -1.0)]

    first = aggregate_metric_rows(rows, samples=100, confidence=0.95, seed=7)
    second = aggregate_metric_rows(list(reversed(rows)), samples=100, confidence=0.95, seed=7)

    assert first == second
    assert first["overall"]["metrics"]["delta_high_lsd_db"]["mean"] == -1.5


def test_release_aggregation_bootstraps_correlated_rows_by_track() -> None:
    rows = [{**_quality_row(f"track-a-{index}", 0.0), "track_id": "track-a"} for index in range(10)]
    rows.append({**_quality_row("track-b-0", 10.0), "track_id": "track-b"})

    aggregate = aggregate_metric_rows(
        rows,
        samples=2_000,
        confidence=0.95,
        seed=7,
        cluster_field="track_id",
    )

    metric = aggregate["overall"]["metrics"]["delta_high_lsd_db"]
    assert aggregate["bootstrap_unit"] == "track_id"
    assert metric["cluster_count"] == 2
    assert metric["mean"] == pytest.approx(10.0 / 11.0)
    assert metric["ci95"][1] > 5.0


def test_sxa_round_trip_and_real_rust_chunk_equivalence(tmp_path: Path) -> None:
    sample_rate = 44_100
    time = np.arange(5003, dtype=np.float32) / sample_rate
    audio = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 440.0 * time),
            0.1 * np.sin(2.0 * np.pi * 997.0 * time),
        )
    ).astype(np.float32)
    transport = tmp_path / "audio.sxa"
    write_sxa(transport, audio, sample_rate)
    decoded, decoded_rate = read_sxa(transport)
    assert decoded_rate == sample_rate
    np.testing.assert_array_equal(decoded, audio)

    evaluator = RustStreamEvaluator()
    model = REPOSITORY / "tests" / "fixtures" / "identity.onnx"
    offline = evaluator.process(
        model, audio, sample_rate, mode="offline", fft_size=1024, hop_size=512
    )
    chunk_a = evaluator.process(
        model,
        audio,
        sample_rate,
        mode="chunked",
        chunk_frames=(1, 257, 509),
        fft_size=1024,
        hop_size=512,
    )
    chunk_b = evaluator.process(
        model,
        audio,
        sample_rate,
        mode="chunked",
        chunk_frames=(31, 512, 997, 7),
        fft_size=1024,
        hop_size=512,
    )

    np.testing.assert_allclose(chunk_a.audio, offline.audio, rtol=0, atol=1e-5)
    np.testing.assert_allclose(chunk_b.audio, offline.audio, rtol=0, atol=1e-5)
    assert offline.report["processor"]["latency_samples_per_channel"] == 512


def test_evaluation_protocol_is_bound_to_1024_512_artifact_metadata() -> None:
    model = REPOSITORY / "tests" / "fixtures" / "identity.onnx"
    metadata = _read_artifact_metadata(model)
    assert _artifact_frame_contract(metadata) == (1024, 512, 1000.0)

    config = load_gate_config(GATE_CONFIG)
    report = _gate_report(_quality_row("contract-row", -1.0), artifact_hash="a" * 64)
    report["artifact"]["metadata"] = metadata
    report["release_gates"] = evaluate_release_gates(report, config)
    assert report["protocol"]["fft_size"] == 1024
    assert report["protocol"]["hop_size"] == 512
    _validate_evaluation_evidence(report)

    protocol_drift = json.loads(json.dumps(report))
    protocol_drift["protocol"]["fft_size"] = 2048
    with pytest.raises(ValueError, match="protocol FFT/hop"):
        _validate_evaluation_evidence(protocol_drift)

    metadata_tamper = json.loads(json.dumps(report))
    metadata_tamper["artifact"]["metadata"]["soundex.input_shape"] = "batch,2,1,1025"
    with pytest.raises(ValueError, match="frame metadata is malformed"):
        _validate_evaluation_evidence(metadata_tamper)


def test_known_good_and_bad_reports_have_clear_exit_codes(tmp_path: Path) -> None:
    config = load_gate_config(GATE_CONFIG)
    good = _gate_report(_quality_row("good", -1.0), artifact_hash="a" * 64)
    good["release_gates"] = evaluate_release_gates(good, config)
    assert good["release_gates"]["passed"]

    incomplete_stereo = json.loads(json.dumps(good))
    incomplete_stereo["stereo_rows"][0]["metrics"].pop("enhanced_correlation_error")
    incomplete_stereo["release_gates"] = evaluate_release_gates(incomplete_stereo, config)
    assert "stereo_metrics_present" in incomplete_stereo["release_gates"]["failed"]

    bad_row = _quality_row("bad", 1.0)
    bad_row["metrics"].update(
        {
            "low_band_preservation_lsd_db": 0.5,
            "chunk_max_abs_error": 0.1,
            "clipped_samples": 1.0,
            "gate_false_bypass": 1.0,
        }
    )
    bad = _gate_report(bad_row, artifact_hash="b" * 64)
    bad["parity_evidence"] = {"passed": False, "artifact_sha256": None}
    bad["perceptual"] = {"status": "unavailable", "required_rows": 1, "scored_rows": 0}
    bad["release_gates"] = evaluate_release_gates(bad, config)
    assert not bad["release_gates"]["passed"]
    assert "high_band_improvement_ci" in bad["release_gates"]["failed"]
    assert "visqol_audio" in bad["release_gates"]["failed"]

    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    good_process = subprocess.run(
        [sys.executable, "-m", "evaluation.gates", str(good_path), "--config", str(GATE_CONFIG)],
        cwd=REPOSITORY / "training",
        capture_output=True,
        text=True,
        check=False,
    )
    bad_process = subprocess.run(
        [sys.executable, "-m", "evaluation.gates", str(bad_path), "--config", str(GATE_CONFIG)],
        cwd=REPOSITORY / "training",
        capture_output=True,
        text=True,
        check=False,
    )
    assert good_process.returncode == 0
    assert bad_process.returncode == 2
    assert "FAIL high_band_improvement_ci" in bad_process.stdout


def test_visqol_parser_accepts_json_and_named_text() -> None:
    assert _parse_score('{"moslqo": 4.25}') == 4.25
    assert _parse_score("MOS-LQO: 3.75") == 3.75
    with pytest.raises(RuntimeError, match="could not parse"):
        _parse_score("no score here")


def test_model_card_checker_binds_artifact_and_report_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(performance, "_MINIMUM_STRESS_SECONDS", 1)
    # This synthetic test isolates existing quality/hash binding; source-rights
    # acceptance and rejection are exercised in test_weight_licensing.py.
    monkeypatch.setattr("evaluation.model_card.check_license_review", lambda *_args: "a" * 64)
    artifact = _write_production_sized_artifact(tmp_path / "model.onnx")
    artifact_hash = sha256_file(artifact)
    gate_config = load_gate_config(GATE_CONFIG)
    report = _gate_report(_quality_row("release-row", -1.0), artifact_hash=artifact_hash)
    report["artifact"]["metadata"] = _read_artifact_metadata(artifact)
    report["protocol"]["fft_size"] = 256
    report["protocol"]["hop_size"] = 128
    report["release_gates"] = evaluate_release_gates(report, gate_config)
    bound_files = _bind_external_evidence(report, tmp_path)
    _validate_artifact_metadata_binding(report, artifact)
    metadata_tamper = json.loads(json.dumps(report))
    metadata_tamper["artifact"]["metadata"]["soundex.model_architecture"] = "forged"
    with pytest.raises(ValueError, match="metadata does not match ONNX file"):
        _validate_artifact_metadata_binding(metadata_tamper, artifact)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_hash = sha256_file(report_path)
    apple_performance = _write_performance_report(
        tmp_path / "performance-apple.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="aarch64",
        operating_system="macos",
        hardware_label="Apple M4 reference",
    )
    x86_performance = _write_performance_report(
        tmp_path / "performance-x86.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="x86_64",
        operating_system="linux",
        hardware_label="x86_64 reference",
    )
    values = {field: "filled release value" for field in _REQUIRED_FIELDS}
    values["SHA-256"] = artifact_hash
    values["Evaluation report / SHA-256"] = f"evaluation-report.json {report_hash}"
    values["Real-model runtime evidence"] = " ".join(
        f"{path.name} {sha256_file(path)}" for path in (apple_performance, x86_performance)
    )
    card = tmp_path / "model-card.md"
    card.write_text(
        "\n".join(f"| **{field}** | {values[field]} |" for field in sorted(values)),
        encoding="utf-8",
    )

    result = check_release_model_card(
        artifact_path=artifact,
        model_card_path=card,
        report_path=report_path,
        performance_report_paths=[apple_performance, x86_performance],
    )
    assert result["artifact_sha256"] == artifact_hash

    parity_contents = bound_files["parity"].read_text(encoding="utf-8")
    bound_files["parity"].write_text(parity_contents + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parity evidence SHA-256"):
        check_release_model_card(
            artifact_path=artifact,
            model_card_path=card,
            report_path=report_path,
            performance_report_paths=[apple_performance, x86_performance],
        )
    bound_files["parity"].write_text(parity_contents, encoding="utf-8")

    report["artifact"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="report artifact hash"):
        check_release_model_card(
            artifact_path=artifact,
            model_card_path=card,
            report_path=report_path,
            performance_report_paths=[apple_performance, x86_performance],
        )


def test_release_evidence_is_recomputed_with_canonical_gates() -> None:
    artifact_hash = "a" * 64
    config = load_gate_config(GATE_CONFIG)
    report = _gate_report(_quality_row("release-row", -1.0), artifact_hash=artifact_hash)
    report["release_gates"] = evaluate_release_gates(report, config)

    _validate_evaluation_evidence(report)

    relaxed = json.loads(json.dumps(report))
    relaxed["release_gate_config"]["thresholds"]["mean_high_band_delta_db_max"] = 100.0
    with pytest.raises(ValueError, match="canonical release-gate config"):
        _validate_evaluation_evidence(relaxed)

    altered_aggregate = json.loads(json.dumps(report))
    altered_aggregate["aggregates"]["overall"]["metrics"]["delta_high_lsd_db"]["mean"] = -100.0
    with pytest.raises(ValueError, match="aggregates do not match"):
        _validate_evaluation_evidence(altered_aggregate)

    forged_gate = json.loads(json.dumps(report))
    forged_gate["release_gates"]["results"][0]["detail"] = "forged"
    with pytest.raises(ValueError, match="canonical recomputation"):
        _validate_evaluation_evidence(forged_gate)


def test_external_evidence_is_bound_to_files(tmp_path: Path) -> None:
    report = _gate_report(_quality_row("release-row", -1.0), artifact_hash="a" * 64)
    bound_files = _bind_external_evidence(report, tmp_path)

    _validate_external_evidence(report)

    bound_files["manifest"].write_text(
        json.dumps(
            {
                "row_id": "different-row",
                "split": "test",
                "recipe_hash": "d" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report["test_manifests"][0]["sha256"] = sha256_file(bound_files["manifest"])
    with pytest.raises(ValueError, match="row IDs do not match"):
        _validate_external_evidence(report)


def test_performance_checker_recomputes_raw_latency_percentiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(performance, "_MINIMUM_STRESS_SECONDS", 1)
    artifact = _write_production_sized_artifact(tmp_path / "model.onnx")
    artifact_hash = sha256_file(artifact)
    apple = _write_performance_report(
        tmp_path / "apple.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="aarch64",
        operating_system="macos",
        hardware_label="Apple reference",
    )
    x86 = _write_performance_report(
        tmp_path / "x86.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="x86_64",
        operating_system="linux",
        hardware_label="x86 reference",
    )

    result = check_performance_reports(
        artifact_path=artifact,
        report_paths=[apple, x86],
    )

    assert result["hardware"] == ["Apple reference", "x86 reference"]
    tampered = json.loads(apple.read_text(encoding="utf-8"))
    tampered["cases"][0]["latency"]["p99_us"] = 1.0
    apple.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="p99_us does not match raw samples"):
        check_performance_reports(
            artifact_path=artifact,
            report_paths=[apple, x86],
        )

    tampered = json.loads(apple.read_text(encoding="utf-8"))
    tampered["protocol"]["fft_size"] = 2048
    apple.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="frame protocol does not match ONNX metadata"):
        check_performance_reports(artifact_path=artifact, report_paths=[apple, x86])


def test_performance_checker_rejects_false_stress_coverage_and_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(performance, "_MINIMUM_STRESS_SECONDS", 1)
    artifact = _write_production_sized_artifact(tmp_path / "model.onnx")
    artifact_hash = sha256_file(artifact)
    apple = _write_performance_report(
        tmp_path / "apple.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="aarch64",
        operating_system="macos",
        hardware_label="Apple reference",
    )
    x86 = _write_performance_report(
        tmp_path / "x86.json",
        artifact_hash=artifact_hash,
        artifact_size=artifact.stat().st_size,
        architecture="x86_64",
        operating_system="linux",
        hardware_label="x86 reference",
    )
    original = json.loads(apple.read_text(encoding="utf-8"))

    shortened = json.loads(json.dumps(original))
    case = shortened["cases"][0]
    case["samples_ns"] = case["samples_ns"][:3]
    case["measured_hops"] = 3
    case["processing_seconds"] = sum(case["samples_ns"]) / 1_000_000_000.0
    case["audio_seconds"] = 3 * 128 / case["sample_rate_hz"]
    case["real_time_factor"] = case["processing_seconds"] / case["audio_seconds"]
    case["throughput_x_realtime"] = 1.0 / case["real_time_factor"]
    case["session_runs"] = 1 + case["warmup_hops"] + 3
    case["expected_session_runs"] = case["session_runs"]
    apple.write_text(json.dumps(shortened), encoding="utf-8")
    with pytest.raises(ValueError, match="raw hop count covers less"):
        check_performance_reports(artifact_path=artifact, report_paths=[apple, x86])

    excessive_rss = json.loads(json.dumps(original))
    excessive_rss["cases"][0]["rss_bytes"] = 50 * 1024 * 1024 + 1
    apple.write_text(json.dumps(excessive_rss), encoding="utf-8")
    with pytest.raises(ValueError, match="RSS exceeds 50 MiB"):
        check_performance_reports(artifact_path=artifact, report_paths=[apple, x86])

    oversized = tmp_path / "oversized.onnx"
    oversized.write_bytes(b"x" * (8 * 1024 * 1024))
    with pytest.raises(ValueError, match="smaller than 8 MiB"):
        check_performance_reports(artifact_path=oversized, report_paths=[apple, x86])


def _write_production_sized_artifact(path: Path) -> Path:
    graph = onnx.load(str(REPOSITORY / "tests" / "fixtures" / "low-latency-identity.onnx"))
    padding = graph.graph.initializer.add()
    padding.name = "performance_test_padding"
    padding.data_type = onnx.TensorProto.FLOAT
    padding.dims.extend([262_144])
    padding.raw_data = b"\0" * (262_144 * 4)
    onnx.save_model(graph, str(path), save_as_external_data=False)
    assert 1_000_000 <= path.stat().st_size < 8 * 1024 * 1024
    return path


def _read_artifact_metadata(path: Path) -> dict[str, str]:
    graph = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in graph.metadata_props}


def _write_performance_report(
    path: Path,
    *,
    artifact_hash: str,
    artifact_size: int,
    architecture: str,
    operating_system: str,
    hardware_label: str,
) -> Path:
    cases = []
    for sample_rate in (44_100, 48_000):
        measured_hops = (sample_rate + 127) // 128
        samples = [100_000 + index % 3 * 100_000 for index in range(measured_hops)]
        processing_seconds = sum(samples) / 1_000_000_000.0
        audio_seconds = measured_hops * 128 / sample_rate
        real_time_factor = processing_seconds / audio_seconds
        deadline_us = 128 / sample_rate * 1_000_000.0
        for channels in (1, 2):
            cases.append(
                {
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "algorithmic_latency_samples_per_channel": 128,
                    "algorithmic_latency_ms": 128 / sample_rate * 1_000.0,
                    "processor_construction_ms": 10.0,
                    "first_frame_us": 250.0,
                    "warmup_hops": 32,
                    "configured_measurement_seconds": 1,
                    "measurement_wall_seconds": 1.1,
                    "measured_hops": measured_hops,
                    "processing_seconds": processing_seconds,
                    "audio_seconds": audio_seconds,
                    "real_time_factor": real_time_factor,
                    "throughput_x_realtime": 1.0 / real_time_factor,
                    "deadline_us": deadline_us,
                    "deadline_misses": 0,
                    "nonfinite_output_samples": 0,
                    "session_runs": 1 + 32 + measured_hops,
                    "expected_session_runs": 1 + 32 + measured_hops,
                    "rss_bytes": 10_000_000,
                    "rss_measurement": "fixture-peak-rss",
                    "serial_latency_p99_ms": 128 / sample_rate * 1000.0 + 0.3,
                    "serial_latency_max_ms": 128 / sample_rate * 1000.0 + 0.3,
                    "latency_target_met": True,
                    "latency": {
                        "p50_us": 200.0,
                        "p95_us": 300.0,
                        "p99_us": 300.0,
                        "max_us": 300.0,
                    },
                    "samples_ns": samples,
                }
            )
    report = {
        "schema_version": 4,
        "report_type": "soundex-performance",
        "generated_unix_seconds": 1,
        "diagnostic": False,
        "artifact": {"sha256": artifact_hash, "size_bytes": artifact_size},
        "protocol": {"fft_size": 256, "hop_size": 128, "frequency_bins": 129},
        "hardware": {
            "label": hardware_label,
            "cpu": hardware_label,
            "architecture": architecture,
            "operating_system": operating_system,
            "host": "fixture-host",
            "power_mode": "performance",
            "rustc": "rustc fixture",
        },
        "ort": {
            "intra_threads": 1,
            "inter_threads": 1,
            "parallel_execution": False,
            "graph_optimization": "level3",
            "output_preallocation": True,
        },
        "cold_model_load_ms": 10.0,
        "peak_rss_bytes": 10_000_000,
        "cases": cases,
        "performance_gates": {
            "passed": True,
            "preferred_target_met": True,
            "failed": [],
            "thresholds": {
                "maximum_algorithmic_latency_samples": 128,
                "maximum_stereo_p99_us": 5_000.0,
                "target_serial_latency_us": 8000.0,
                "maximum_serial_latency_us": 10000.0,
                "maximum_deadline_misses": 0,
                "maximum_real_time_factor": 1.0,
                "maximum_peak_rss_bytes": 50 * 1024 * 1024,
                "minimum_warmup_hops": 1,
                "minimum_measurement_seconds_per_case": 1,
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _bind_external_evidence(report: dict[str, object], tmp_path: Path) -> dict[str, Path]:
    artifact = report["artifact"]
    assert isinstance(artifact, dict)
    metadata = artifact.get("metadata")
    assert isinstance(metadata, dict)
    recipe_hash = str(metadata["soundex.data_recipe_sha256"])

    parity = report["parity_evidence"]
    assert isinstance(parity, dict)
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    report["parity_evidence"] = {
        **parity,
        "path": str(parity_path.resolve()),
        "sha256": sha256_file(parity_path),
    }

    listening_path = tmp_path / "listening.md"
    listening_path.write_text("Recorded blinded listening result.\n", encoding="utf-8")
    report["listening_protocol"] = {
        "status": "recorded",
        "summary": "Recorded blinded listening result.",
        "path": str(listening_path.resolve()),
        "sha256": sha256_file(listening_path),
    }

    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "row_id": rows[0]["row_id"],
                "split": "test",
                "recipe_hash": recipe_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report["test_manifests"] = [
        {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "test_rows": 1,
        }
    ]
    return {
        "parity": parity_path,
        "listening": listening_path,
        "manifest": manifest_path,
    }


def _quality_row(row_id: str, high_delta: float) -> dict[str, object]:
    return {
        "row_id": row_id,
        "track_id": row_id,
        "corpus": "musdb18_hq",
        "codec_id": "mp3-cbr-64",
        "quality": "cbr:64",
        "sample_rate": 48_000,
        "channel_role": "mid",
        "metrics": {
            "baseline_high_lsd_db": 2.0,
            "enhanced_high_lsd_db": 2.0 + high_delta,
            "delta_high_lsd_db": high_delta,
            "low_band_preservation_lsd_db": 0.1,
            "chunk_max_abs_error": 0.0,
            "clipped_samples": 0.0,
            "nonfinite_output": 0.0,
            "gate_false_bypass": 0.0,
            "baseline_visqol": 3.0,
            "enhanced_visqol": 3.1,
            "delta_visqol": 0.1,
        },
    }


def _gate_report(row: dict[str, object], *, artifact_hash: str) -> dict[str, object]:
    config = load_gate_config(GATE_CONFIG)
    bootstrap = config["bootstrap"]
    aggregates = aggregate_metric_rows(
        [row],
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
        cluster_field="track_id",
    )
    stereo_rows = [
        {
            "row_id": "stereo-fixture",
            "track_id": str(row["track_id"]),
            "corpus": str(row["corpus"]),
            "codec_id": str(row["codec_id"]),
            "sample_rate": int(row["sample_rate"]),
            "metrics": {
                "baseline_correlation_error": 0.1,
                "enhanced_correlation_error": 0.05,
                "baseline_image_width_error_db": 0.2,
                "enhanced_image_width_error_db": 0.1,
                "baseline_interchannel_phase_error_rad": 0.1,
                "enhanced_interchannel_phase_error_rad": 0.05,
            },
        }
    ]
    stereo_aggregates = aggregate_metric_rows(
        stereo_rows,
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
        dimensions=("corpus", "codec_id", "sample_rate"),
        cluster_field="track_id",
    )
    return {
        "schema_version": 2,
        "artifact": {
            "sha256": artifact_hash,
            "metadata": {
                "soundex.artifact_schema": "1.2",
                "soundex.fft_size": "1024",
                "soundex.hop_size": "512",
                "soundex.input_shape": "batch,2,1,513",
                "soundex.output_shape": "batch,2,1,513",
                "soundex.crossover_width_hz": "1000.0",
                "soundex.data_recipe_sha256": "d" * 64,
            },
        },
        "rows": [row],
        "stereo_rows": stereo_rows,
        "aggregates": aggregates,
        "stereo_aggregates": stereo_aggregates,
        "protocol": {
            "audio_files_audited": True,
            "bootstrap_unit": "track_id",
            "fft_size": 1024,
            "hop_size": 512,
            "crossover_width_hz": 1000.0,
        },
        "perceptual": {"status": "available", "required_rows": 1, "scored_rows": 1},
        "parity_evidence": {
            "schema_version": 2,
            "suite": "pytorch-ort-rust-v2",
            "policy": PARITY_POLICY,
            "policy_sha256": POLICY_SHA256,
            "cases": [
                {
                    "name": name,
                    "pytorch_ort": {key: 0.0 for key in PARITY_POLICY},
                    "pytorch_rust": {key: 0.0 for key in PARITY_POLICY},
                }
                for name in sorted(REQUIRED_PARITY_CASES)
            ],
            "passed": True,
            "artifact_sha256": artifact_hash,
        },
        "release_gate_config": config,
    }
