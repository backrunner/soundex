"""Fail-closed release checker for model cards and evaluation evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import onnx

from artifact_contract import (
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    frame_contract_from_metadata,
)
from evaluation.gates import evaluate_release_gates, load_gate_config
from evaluation.performance import check_performance_reports
from evaluation.reporting import aggregate_metric_rows, sha256_file
from weight_licensing import check_license_review

_TABLE_FIELD = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|\s*(.*?)\s*\|\s*$")
_REQUIRED_FIELDS = {
    "Model name",
    "Version",
    "Artifact file(s)",
    "SHA-256",
    "Checkpoint / artifact schema",
    "Source checkpoint SHA-256",
    "Resolved config / manifest-set SHA-256",
    "Software commit",
    "Training config",
    "Release date",
    "Authors / maintainers",
    "Weight license",
    "License file",
    "Data rights review / SHA-256",
    "Recipe version / SHA-256",
    "Manifest file(s) / SHA-256",
    "Manifest row and track counts by split",
    "Track-ID split intersections",
    "Sample rates",
    "Codec/encoder/mode/settings",
    "Held-out codec setting",
    "Channel roles and weights",
    "Alignment tolerance / rejection count",
    "Evaluation report / SHA-256",
    "Test manifest version / SHA-256",
    "High-band LSD baseline / enhanced / delta 95% CI",
    "Low-band preservation LSD",
    "ViSQOLAudio baseline / enhanced",
    "Stereo correlation / phase / width",
    "Gate call / false-bypass rate",
    "Streaming chunk equivalence",
    "PyTorch / ORT / Rust parity evidence",
    "Listening protocol / anonymized aggregate",
    "Supported sample rates / codecs",
    "Algorithmic latency",
    "Real-model runtime evidence",
    "Known evaluation limitations",
}


def check_release_model_card(
    *,
    artifact_path: str | Path,
    model_card_path: str | Path,
    report_path: str | Path,
    performance_report_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Validate required fields and bind artifact, report, and card hashes."""
    artifact = Path(artifact_path).resolve()
    card = Path(model_card_path).resolve()
    report_file = Path(report_path).resolve()
    fields = _parse_fields(card.read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"model card is missing required field {missing[0]!r}")
    blank = sorted(name for name in _REQUIRED_FIELDS if _is_placeholder(fields[name]))
    if blank:
        raise ValueError(f"model card field {blank[0]!r} is blank or still instructional")

    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be a JSON object")
    if int(report.get("schema_version", 0)) != 2:
        raise ValueError("evaluation report schema is unsupported")

    artifact_sha256 = sha256_file(artifact)
    report_sha256 = sha256_file(report_file)
    if report.get("artifact", {}).get("sha256") != artifact_sha256:
        raise ValueError("evaluation report artifact hash does not match ONNX file")
    card_artifact_hash = _extract_sha256(fields["SHA-256"])
    if card_artifact_hash != artifact_sha256:
        raise ValueError("model card artifact hash does not match ONNX file")
    card_evaluation_hash = _extract_sha256(fields["Evaluation report / SHA-256"])
    if card_evaluation_hash != report_sha256:
        raise ValueError("model card evaluation report hash does not match JSON report")
    _validate_artifact_metadata_binding(report, artifact)
    rights_review_hash = check_license_review(
        fields, report["artifact"]["metadata"], artifact_sha256, card
    )
    _validate_evaluation_evidence(report)
    _validate_external_evidence(report)
    if report.get("listening_protocol", {}).get("status") != "recorded":
        raise ValueError("evaluation report lacks a recorded listening protocol")
    performance = check_performance_reports(
        artifact_path=artifact,
        report_paths=performance_report_paths,
    )
    performance_hashes = set(performance["performance_report_sha256"].values())
    card_performance_hashes = set(_extract_sha256s(fields["Real-model runtime evidence"]))
    if card_performance_hashes != performance_hashes:
        raise ValueError("model card performance report hashes do not match JSON reports")
    return {
        "artifact_sha256": artifact_sha256,
        "evaluation_report_sha256": report_sha256,
        "performance_report_sha256": sorted(performance_hashes),
        "model_card": str(card),
        "data_rights_review_sha256": rights_review_hash,
    }


def _validate_evaluation_evidence(report: dict[str, Any]) -> None:
    _report_frame_contract(report)
    canonical_config = load_gate_config(Path(__file__).with_name("release_gates.v1.yaml"))
    if report.get("release_gate_config") != canonical_config:
        raise ValueError("evaluation report does not use the canonical release-gate config")
    bootstrap = canonical_config["bootstrap"]
    rows = report.get("rows")
    stereo_rows = report.get("stereo_rows")
    if not isinstance(rows, list) or not isinstance(stereo_rows, list):
        raise ValueError("evaluation report row evidence is malformed")
    try:
        expected_aggregates = aggregate_metric_rows(
            rows,
            samples=int(bootstrap["samples"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]),
            cluster_field="track_id",
        )
        expected_stereo = (
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
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"evaluation row evidence cannot be aggregated: {error}") from error
    if report.get("aggregates") != expected_aggregates:
        raise ValueError("evaluation aggregates do not match row-level evidence")
    if report.get("stereo_aggregates") != expected_stereo:
        raise ValueError("stereo aggregates do not match row-level evidence")
    if report.get("protocol", {}).get("bootstrap_unit") != "track_id":
        raise ValueError("evaluation protocol does not declare track-cluster bootstrap")
    expected_gates = evaluate_release_gates(report, canonical_config)
    if report.get("release_gates") != expected_gates:
        raise ValueError("evaluation release gates do not match canonical recomputation")
    if not expected_gates["passed"]:
        raise ValueError(f"evaluation release gates failed: {', '.join(expected_gates['failed'])}")


def _validate_artifact_metadata_binding(report: dict[str, Any], artifact: Path) -> None:
    try:
        graph = onnx.load(str(artifact), load_external_data=False)
    except Exception as error:
        raise ValueError("release artifact is not a readable ONNX model") from error
    actual = {item.key: item.value for item in graph.metadata_props}
    missing = sorted(REQUIRED_METADATA_KEYS - set(actual))
    if missing:
        raise ValueError(f"release artifact metadata is missing {missing[0]}")
    embedded = report.get("artifact", {}).get("metadata")
    if not isinstance(embedded, dict):
        raise ValueError("evaluation report artifact metadata is malformed")
    if embedded != actual:
        differing = sorted(
            key for key in set(embedded) | set(actual) if embedded.get(key) != actual.get(key)
        )
        raise ValueError(
            f"evaluation report artifact metadata does not match ONNX file at {differing[0]}"
        )


def _report_frame_contract(report: dict[str, Any]) -> tuple[int, int]:
    metadata = report.get("artifact", {}).get("metadata")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("evaluation report artifact metadata is malformed")
    try:
        fft_size, hop_size = frame_contract_from_metadata(metadata)
        crossover_width_hz = float(metadata["soundex.crossover_width_hz"])
    except (ExportValidationError, KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation report artifact frame metadata is malformed") from error

    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("evaluation report protocol is malformed")
    if protocol.get("fft_size") != fft_size or protocol.get("hop_size") != hop_size:
        raise ValueError("evaluation protocol FFT/hop does not match artifact metadata")
    protocol_crossover = protocol.get("crossover_width_hz")
    if (
        isinstance(protocol_crossover, bool)
        or not isinstance(protocol_crossover, (int, float))
        or abs(float(protocol_crossover) - crossover_width_hz) > 1e-6
    ):
        raise ValueError("evaluation protocol crossover width does not match artifact metadata")
    return fft_size, hop_size


def _validate_external_evidence(report: dict[str, Any]) -> None:
    parity = report.get("parity_evidence")
    if not isinstance(parity, dict):
        raise ValueError("evaluation report parity evidence is malformed")
    parity_path = _bound_file(parity, label="parity evidence")
    parity_file = json.loads(parity_path.read_text(encoding="utf-8"))
    embedded_parity = {key: value for key, value in parity.items() if key not in {"path", "sha256"}}
    if not isinstance(parity_file, dict) or parity_file != embedded_parity:
        raise ValueError("parity evidence file does not match evaluation report")

    listening = report.get("listening_protocol")
    if not isinstance(listening, dict) or listening.get("status") != "recorded":
        raise ValueError("evaluation report lacks a recorded listening protocol")
    listening_path = _bound_file(listening, label="listening record")
    if not listening_path.read_text(encoding="utf-8").strip():
        raise ValueError("listening record is empty")

    records = report.get("test_manifests")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation report test-manifest evidence is missing")
    manifest_paths: set[Path] = set()
    manifest_row_ids: set[str] = set()
    expected_recipe_hash = (
        report.get("artifact", {}).get("metadata", {}).get("soundex.data_recipe_sha256")
    )
    if not isinstance(expected_recipe_hash, str) or len(expected_recipe_hash) != 64:
        raise ValueError("evaluation report lacks the artifact data-recipe hash")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("evaluation report test-manifest record is malformed")
        path = _bound_file(record, label="test manifest")
        if path in manifest_paths:
            raise ValueError("evaluation report repeats a test manifest")
        manifest_paths.add(path)
        test_rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"test manifest {path}:{line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"test manifest {path}:{line_number} row is malformed")
            if row.get("split") == "test":
                if row.get("recipe_hash") != expected_recipe_hash:
                    raise ValueError("test manifest recipe hash does not match artifact metadata")
                test_rows.append(row)
        expected_count = record.get("test_rows")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise ValueError("test manifest row count is malformed")
        if len(test_rows) != expected_count:
            raise ValueError("test manifest row count does not match evaluation report")
        for row in test_rows:
            row_id = str(row.get("row_id", ""))
            if not row_id or row_id in manifest_row_ids:
                raise ValueError("test manifest row IDs are missing or duplicated")
            manifest_row_ids.add(row_id)
    report_rows = report.get("rows", [])
    evaluated_row_ids = {str(row.get("row_id", "")) for row in report_rows if isinstance(row, dict)}
    if evaluated_row_ids != manifest_row_ids:
        raise ValueError("evaluation row IDs do not match bound test manifests")


def _bound_file(record: dict[str, Any], *, label: str) -> Path:
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_hash, str):
        raise ValueError(f"{label} path or SHA-256 is missing")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 does not match evaluation report")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("model_card", type=Path)
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("performance_reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        result = check_release_model_card(
            artifact_path=args.artifact,
            model_card_path=args.model_card,
            report_path=args.evaluation_report,
            performance_report_paths=args.performance_reports,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL release-model-card: {error}")
        raise SystemExit(2) from error
    print(
        "PASS release-model-card: "
        f"artifact={result['artifact_sha256']} report={result['evaluation_report_sha256']}"
    )


def _parse_fields(markdown: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in markdown.splitlines():
        match = _TABLE_FIELD.match(line)
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("*").strip()
    lowered = normalized.casefold()
    return (
        not normalized
        or "…" in normalized
        or "..." in normalized
        or "*(" in value
        or "e.g." in lowered
        or "yyyy" in lowered
        or "____________" in normalized
        or lowered in {"tbd", "todo", "n/a", "none"}
    )


def _extract_sha256(value: str) -> str:
    hashes = _extract_sha256s(value)
    if len(hashes) != 1:
        raise ValueError("hash field must contain exactly one lowercase SHA-256")
    return hashes[0]


def _extract_sha256s(value: str) -> list[str]:
    return re.findall(r"\b[0-9a-f]{64}\b", value.casefold())


if __name__ == "__main__":
    main()
