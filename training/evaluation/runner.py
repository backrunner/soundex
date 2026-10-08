"""Held-out manifest runner using the actual Rust causal streaming path."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import soundfile as sf

from artifact_contract import (
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    frame_contract_from_metadata,
)
from data.protocol import (
    MANIFEST_NAME,
    DataRecipe,
    assert_deployment_coverage,
    load_manifest,
    sha256_file,
    validate_manifest_rows,
)
from evaluation.gates import evaluate_release_gates
from evaluation.metrics import evaluate_signal_triplet, evaluate_stereo_image
from evaluation.reporting import (
    aggregate_metric_rows,
    write_json_report,
    write_markdown_report,
)
from evaluation.reporting import (
    sha256_file as report_sha256_file,
)
from evaluation.rust_bridge import RustStreamEvaluator
from evaluation.visqol import ExternalVisqolAudio
from validation import quality_label

EVALUATION_SCHEMA_VERSION = 2
CHUNK_PATTERNS = ((1, 257, 509), (31, 512, 997, 7))


def run_manifest_evaluation(
    *,
    model_path: str | Path,
    manifests: list[str | Path],
    output_directory: str | Path,
    gate_config: dict[str, Any],
    recipe: DataRecipe,
    rust_binary: str | Path | None = None,
    visqol: ExternalVisqolAudio | None = None,
    visqol_supported_rates: tuple[int, ...] = (48_000,),
    parity_evidence_path: str | Path | None = None,
    listening_record_path: str | Path | None = None,
    audit_files: bool = False,
) -> dict[str, Any]:
    """Evaluate every test row, aggregate uncertainty, gate, and publish reports."""
    model = Path(model_path).resolve()
    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"evaluation requires an exported ONNX model: {model}")
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_record(model, recipe)
    fft_size, hop_size, crossover_width_hz = _artifact_frame_contract(artifact["metadata"])
    parity_evidence = _load_parity_evidence(parity_evidence_path)
    rows, manifest_records = _load_test_rows(
        manifests,
        recipe=recipe,
        required_corpora=frozenset(parity_evidence.get("training_manifest_corpora", [])),
        audit_files=audit_files,
    )
    evaluator = RustStreamEvaluator(rust_binary)
    metric_rows: list[dict[str, Any]] = []
    stereo_rows: list[dict[str, Any]] = []
    visqol_errors: list[str] = []
    visqol_required_rows = sum(
        int(row["sample_rate"]) in set(visqol_supported_rates) for row in rows
    )
    visqol_scored_rows = 0

    for unit in _evaluation_units(rows):
        clean, degraded = _load_unit_audio(unit)
        sample_rate = int(unit[0]["sample_rate"])
        offline = evaluator.process(
            model,
            degraded,
            sample_rate,
            mode="offline",
            fft_size=fft_size,
            hop_size=hop_size,
            crossover_width_hz=crossover_width_hz,
        )
        chunk_outputs = [
            evaluator.process(
                model,
                degraded,
                sample_rate,
                mode="chunked",
                chunk_frames=pattern,
                fft_size=fft_size,
                hop_size=hop_size,
                crossover_width_hz=crossover_width_hz,
            )
            for pattern in CHUNK_PATTERNS
        ]
        chunk_max = max(
            float(np.max(np.abs(chunk.audio.astype(np.float64) - offline.audio)))
            for chunk in chunk_outputs
        )
        analysis = offline.report["analysis"]
        processor_report = offline.report["processor"]
        frames_analyzed = int(analysis["frames_analyzed"])
        frames_needing = int(analysis["frames_needing_enhancement"])
        call_rate = frames_needing / max(frames_analyzed, 1)
        known_degradation = any(
            float(row["measured_cutoff_hz"]) < int(row["sample_rate"]) / 2.0 for row in unit
        )
        false_bypass = float(known_degradation and frames_needing == 0)

        for channel_index, row in enumerate(unit):
            clean_channel = clean[:, channel_index]
            degraded_channel = degraded[:, channel_index]
            enhanced_channel = offline.audio[:, channel_index]
            metrics = evaluate_signal_triplet(
                clean_channel,
                degraded_channel,
                enhanced_channel,
                sample_rate=sample_rate,
                cutoff_hz=float(row["measured_cutoff_hz"]),
                crossover_width_hz=crossover_width_hz,
                fft_size=fft_size,
                hop_size=hop_size,
            )
            metrics.update(
                {
                    "chunk_max_abs_error": chunk_max,
                    "gate_call_rate": call_rate,
                    "gate_false_bypass": false_bypass,
                    "clipped_samples": float(processor_report["clipped_samples"]),
                    "nonfinite_output": 0.0,
                }
            )
            if visqol is not None and visqol.supports(sample_rate):
                try:
                    baseline_visqol = visqol.score(clean_channel, degraded_channel, sample_rate)
                    enhanced_visqol = visqol.score(clean_channel, enhanced_channel, sample_rate)
                except (OSError, RuntimeError, ValueError) as error:
                    visqol_errors.append(f"{row['row_id']}: {error}")
                else:
                    metrics.update(
                        {
                            "baseline_visqol": baseline_visqol,
                            "enhanced_visqol": enhanced_visqol,
                            "delta_visqol": enhanced_visqol - baseline_visqol,
                        }
                    )
                    visqol_scored_rows += 1
            metric_rows.append(_metric_row(row, metrics))

        if len(unit) == 2:
            stereo_metrics = evaluate_stereo_image(
                clean,
                degraded,
                offline.audio,
                fft_size=fft_size,
                hop_size=hop_size,
            )
            stereo_rows.append(
                {
                    "row_id": f"{unit[0]['row_id']}+{unit[1]['row_id']}",
                    "track_id": str(unit[0]["track_id"]),
                    "corpus": str(unit[0]["corpus"]),
                    "codec_id": str(unit[0]["codec_id"]),
                    "quality": quality_label(
                        str(unit[0]["codec_mode"]), float(unit[0]["codec_setting"])
                    ),
                    "sample_rate": sample_rate,
                    "channel_role": "stereo",
                    "metrics": stereo_metrics,
                }
            )

    bootstrap = gate_config["bootstrap"]
    aggregates = aggregate_metric_rows(
        metric_rows,
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
        cluster_field="track_id",
    )
    stereo_aggregates = (
        aggregate_metric_rows(
            stereo_rows,
            samples=int(bootstrap["samples"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]),
            dimensions=("corpus", "codec_id", "sample_rate"),
            cluster_field="track_id",
        )
        if stereo_rows
        else None
    )
    perceptual_status = (
        "unavailable"
        if visqol is None
        else "failed"
        if visqol_errors
        else "available"
        if visqol_required_rows > 0 and visqol_scored_rows == visqol_required_rows
        else "incomplete"
    )
    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": artifact,
        "test_manifests": manifest_records,
        "protocol": {
            "split": "test",
            "rust_modes": ["offline", "chunked", "chunked"],
            "chunk_patterns": [list(pattern) for pattern in CHUNK_PATTERNS],
            "delay_compensation": "Rust same-length causal process_buffer/process_chunk+finalize",
            "fft_size": fft_size,
            "hop_size": hop_size,
            "crossover_width_hz": crossover_width_hz,
            "bootstrap": bootstrap,
            "bootstrap_unit": "track_id",
            "data_recipe_sha256": recipe.hash,
            "audio_files_audited": audit_files,
        },
        "rows": metric_rows,
        "stereo_rows": stereo_rows,
        "aggregates": aggregates,
        "stereo_aggregates": stereo_aggregates,
        "perceptual": {
            "name": "ViSQOLAudio",
            "status": perceptual_status,
            "required_rows": visqol_required_rows,
            "scored_rows": visqol_scored_rows,
            "errors": visqol_errors,
        },
        "parity_evidence": parity_evidence,
        "listening_protocol": _listening_record(listening_record_path),
        "release_gate_config": gate_config,
    }
    report["release_gates"] = evaluate_release_gates(report, gate_config)
    json_path = output_dir / "evaluation-report.json"
    markdown_path = output_dir / "evaluation-report.md"
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)
    return report


def _load_test_rows(
    manifests: list[str | Path],
    *,
    recipe: DataRecipe,
    required_corpora: frozenset[str],
    audit_files: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not manifests:
        raise ValueError("at least one test manifest is required")
    selected: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for raw_path in manifests:
        path = Path(raw_path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        path = path.resolve()
        rows = load_manifest(path)
        validate_manifest_rows(
            rows,
            path.parent,
            recipe=recipe,
            audit_files=audit_files,
            file_splits=frozenset({"test"}),
        )
        coverage_rows.extend(dict(row) for row in rows)
        test_rows = [dict(row) for row in rows if row["split"] == "test"]
        if not test_rows:
            raise ValueError(f"manifest contains no test rows: {path}")
        for row in test_rows:
            row["_root"] = str(path.parent)
            row["_manifest"] = str(path)
        selected.extend(test_rows)
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "test_rows": len(test_rows),
            }
        )
    row_ids = [str(row["row_id"]) for row in selected]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("test row IDs must be unique across manifests")
    assert_deployment_coverage(coverage_rows, recipe)
    evaluated_corpora = {str(row["corpus"]) for row in selected}
    missing_corpora = sorted(required_corpora - evaluated_corpora)
    if missing_corpora:
        raise ValueError(f"test manifests are missing trained corpus {missing_corpora[0]!r}")
    return sorted(selected, key=lambda row: str(row["row_id"])), sorted(
        records, key=lambda record: record["path"]
    )


def _evaluation_units(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["channel_role"] not in {"left", "right"}:
            continue
        key = (
            row["_manifest"],
            row["source_id"],
            int(row["start_sample"]),
            int(row["length_samples"]),
            row["codec_id"],
            int(row["sample_rate"]),
        )
        pairs[key][str(row["channel_role"])] = row
    paired_ids: set[str] = set()
    units: list[list[dict[str, Any]]] = []
    for roles in pairs.values():
        if set(roles) == {"left", "right"}:
            unit = [roles["left"], roles["right"]]
            units.append(unit)
            paired_ids.update(str(row["row_id"]) for row in unit)
    units.extend([row] for row in rows if str(row["row_id"]) not in paired_ids)
    return sorted(units, key=lambda unit: str(unit[0]["row_id"]))


def _load_unit_audio(unit: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    clean_channels: list[np.ndarray] = []
    degraded_channels: list[np.ndarray] = []
    expected_rate = int(unit[0]["sample_rate"])
    expected_length = int(unit[0]["length_samples"])
    for row in unit:
        root = Path(str(row["_root"]))
        clean, clean_rate = sf.read(root / str(row["clean_path"]), dtype="float32", always_2d=True)
        degraded, degraded_rate = sf.read(
            root / str(row["degraded_path"]), dtype="float32", always_2d=True
        )
        if clean.shape[1] != 1 or degraded.shape[1] != 1:
            raise ValueError(f"manifest role row must be mono: {row['row_id']}")
        if clean_rate != expected_rate or degraded_rate != expected_rate:
            raise ValueError(f"sample-rate mismatch for {row['row_id']}")
        if len(clean) != expected_length or len(degraded) != expected_length:
            raise ValueError(f"length mismatch for {row['row_id']}")
        clean_channels.append(clean[:, 0])
        degraded_channels.append(degraded[:, 0])
    return np.column_stack(clean_channels), np.column_stack(degraded_channels)


def _metric_row(row: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "row_id": str(row["row_id"]),
        "track_id": str(row["track_id"]),
        "corpus": str(row["corpus"]),
        "codec_id": str(row["codec_id"]),
        "quality": quality_label(str(row["codec_mode"]), float(row["codec_setting"])),
        "sample_rate": int(row["sample_rate"]),
        "channel_role": str(row["channel_role"]),
        "measured_cutoff_hz": float(row["measured_cutoff_hz"]),
        "metrics": dict(sorted(metrics.items())),
    }


def _artifact_record(model: Path, recipe: DataRecipe) -> dict[str, Any]:
    graph = onnx.load(str(model), load_external_data=False)
    metadata = {item.key: item.value for item in graph.metadata_props}
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ValueError(f"ONNX artifact metadata is missing {missing[0]}")
    if metadata.get("soundex.data_recipe_sha256") != recipe.hash:
        raise ValueError("ONNX artifact data recipe hash does not match evaluation config")
    _artifact_frame_contract(metadata)
    return {
        "path": str(model),
        "sha256": report_sha256_file(model),
        "size_bytes": model.stat().st_size,
        "metadata": dict(sorted(metadata.items())),
    }


def _artifact_frame_contract(metadata: dict[str, str]) -> tuple[int, int, float]:
    try:
        fft_size, hop_size = frame_contract_from_metadata(metadata)
        crossover_width_hz = float(metadata["soundex.crossover_width_hz"])
    except (ExportValidationError, KeyError, TypeError, ValueError) as error:
        raise ValueError("ONNX artifact frame metadata is malformed") from error
    if not np.isfinite(crossover_width_hz) or crossover_width_hz < 0.0:
        raise ValueError("ONNX artifact crossover width is invalid")
    return fft_size, hop_size, crossover_width_hz


def _load_parity_evidence(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"passed": False, "artifact_sha256": None, "reason": "not supplied"}
    evidence_path = Path(path).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("parity evidence must be a JSON object")
    if evidence.get("schema_version") != 2:
        raise ValueError("parity evidence schema is unsupported")
    if evidence.get("suite") != "pytorch-ort-rust-v2":
        raise ValueError("parity evidence suite is unsupported")
    return {**evidence, "path": str(evidence_path), "sha256": report_sha256_file(evidence_path)}


def _listening_record(path: str | Path | None) -> dict[str, Any]:
    summary = (
        "Randomized, level-matched, blinded degraded/enhanced/reference clips must be "
        "reviewed across every required stratum; no automated subjective pass is claimed."
    )
    if path is None:
        return {"status": "pending", "summary": summary, "path": None, "sha256": None}
    record_path = Path(path).resolve()
    if not record_path.is_file() or not record_path.read_text(encoding="utf-8").strip():
        raise ValueError("listening record must be a non-empty text file")
    return {
        "status": "recorded",
        "summary": summary,
        "path": str(record_path),
        "sha256": report_sha256_file(record_path),
    }
