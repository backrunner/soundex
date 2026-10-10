"""Export a schema-1.2 SoundEx checkpoint to the artifact-schema-1.3 ONNX contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
import yaml
from onnx import TensorProto

from artifact_contract import (
    ARTIFACT_SCHEMA_VERSION,
    INPUT_NAME,
    MAX_ARTIFACT_BYTES,
    MAX_PARAMETERS,
    OPSET_VERSION,
    OUTPUT_NAME,
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    tensor_shape,
)
from checkpoint import load_checkpoint
from configuration import normalize_config
from export_validation import OutputTransform, validate_ort_parity
from models.generator import SoundExGenerator


@dataclass(frozen=True)
class ExportResult:
    """Published artifact identity and validation summary."""

    path: Path
    sha256: str
    size_bytes: int
    parameter_count: int
    max_absolute_error: float
    max_mean_error: float


class _RuntimeContract(torch.nn.Module):
    """Make static non-batch output dimensions explicit to the legacy exporter."""

    def __init__(self, generator: SoundExGenerator, output_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.generator = generator
        self.output_shape = output_shape

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        output = self.generator(input_features)
        return output.reshape(input_features.shape[0], *self.output_shape)


def sha256_file(path: str | Path) -> str:
    """Hash one artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_metadata(checkpoint: Mapping[str, Any], checkpoint_sha256: str) -> dict[str, str]:
    """Create the canonical ONNX custom metadata map from checkpoint data."""
    feature = checkpoint["feature_contract"]
    architecture = checkpoint["model"]["architecture"]
    hashes = checkpoint["hashes"]
    data = checkpoint["data"]
    input_shape = ",".join(str(value) for value in feature["input_shape"])
    output_shape = ",".join(str(value) for value in feature["output_shape"])
    return {
        "soundex.artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "soundex.model_architecture": str(architecture["name"]),
        "soundex.model_architecture_version": (
            f"{int(architecture['major'])}.{int(architecture['minor'])}"
        ),
        "soundex.input_name": INPUT_NAME,
        "soundex.output_name": OUTPUT_NAME,
        "soundex.tensor_layout": str(feature["tensor_layout"]),
        "soundex.input_shape": input_shape,
        "soundex.output_shape": output_shape,
        "soundex.input_channels": ",".join(feature["input_channels"]),
        "soundex.output_channels": ",".join(feature["output_channels"]),
        "soundex.sample_rates": ",".join(str(rate) for rate in feature["supported_sample_rates"]),
        "soundex.fft_size": str(feature["fft_size"]),
        "soundex.hop_size": str(feature["hop_size"]),
        "soundex.crossover_width_hz": str(feature["crossover_width_hz"]),
        "soundex.window": str(feature["window"]["name"]),
        "soundex.window_periodic": str(feature["window"]["periodic"]).lower(),
        "soundex.magnitude_scale": str(feature["magnitude"]["scale"]),
        "soundex.db_formula": str(feature["magnitude"]["formula"]),
        "soundex.db_floor": str(feature["magnitude"]["floor_db"]),
        "soundex.phase_units": str(feature["phase"]["units"]),
        "soundex.phase_range": str(feature["phase"]["range"]),
        "soundex.context_frames": str(feature["context_frames"]),
        "soundex.causal": str(feature["causal"]).lower(),
        "soundex.stateless": str(feature["stateless"]).lower(),
        "soundex.source_checkpoint_sha256": checkpoint_sha256,
        "soundex.resolved_config_sha256": str(hashes["resolved_config_sha256"]),
        "soundex.data_recipe_sha256": str(data["recipe_sha256"]),
        "soundex.manifest_set_sha256": str(data["manifest_set_sha256"]),
    }


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    expected_config_path: str | Path | None = None,
    ort_output_transform: OutputTransform | None = None,
) -> ExportResult:
    """Validate, stage, verify, and atomically publish one ONNX artifact."""
    expected_config = _load_optional_config(expected_config_path)
    checkpoint = load_checkpoint(checkpoint_path, expected_config=expected_config)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model = _build_generator(checkpoint)
    metadata = artifact_metadata(checkpoint, checkpoint_sha256)
    feature = checkpoint["feature_contract"]
    static_shape = tensor_shape(int(feature["fft_size"]), int(feature["hop_size"]))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging.onnx", dir=destination.parent
    )
    os.close(descriptor)
    staging = Path(staging_name)
    companion = Path(f"{staging}.data")
    try:
        _export_graph(model, staging, static_shape)
        _embed_metadata(staging, metadata)
        validate_onnx_contract(staging, metadata)
        max_error, max_mean_error = validate_ort_parity(
            model, staging, output_transform=ort_output_transform
        )
        size_bytes = staging.stat().st_size
        if size_bytes >= MAX_ARTIFACT_BYTES:
            raise ExportValidationError(
                f"ONNX artifact is {size_bytes / 1024 / 1024:.2f} MiB; expected <8 MiB"
            )
        artifact_sha256 = sha256_file(staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
        companion.unlink(missing_ok=True)

    result = ExportResult(
        path=destination,
        sha256=artifact_sha256,
        size_bytes=size_bytes,
        parameter_count=model.count_parameters(),
        max_absolute_error=max_error,
        max_mean_error=max_mean_error,
    )
    print(f"Published ONNX model: {destination}")
    print(f"SHA-256: {result.sha256}")
    print(f"Size: {result.size_bytes / 1024 / 1024:.2f} MiB")
    print(
        f"ORT parity passed (unit-aware v2): raw max={result.max_absolute_error:.3e}, "
        f"raw mean={result.max_mean_error:.3e} (diagnostics, not acceptance limits)"
    )
    return result


def _build_generator(checkpoint: Mapping[str, Any]) -> SoundExGenerator:
    generator_config = checkpoint["model"]["generator_config"]
    model = SoundExGenerator(
        channels=[int(channel) for channel in generator_config["channels"]],
        bottleneck_blocks=int(generator_config["bottleneck_blocks"]),
        expand_ratio=int(generator_config["expand_ratio"]),
        cross_stream_interactions=generator_config.get("cross_stream_interactions", False),
    )
    model.load_state_dict(checkpoint["model"]["generator_state"], strict=True)
    model.eval()
    model.float()
    parameter_count = model.count_parameters()
    if parameter_count > MAX_PARAMETERS:
        raise ExportValidationError(
            f"generator has {parameter_count:,} parameters; expected <= {MAX_PARAMETERS:,}"
        )
    return model


def _export_graph(
    model: SoundExGenerator,
    output_path: Path,
    static_shape: tuple[int, int, int],
) -> None:
    dummy_input = torch.zeros((1, *static_shape), dtype=torch.float32)
    export_model = _RuntimeContract(model, static_shape).eval()
    export_options: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        **export_options,
    )

    graph = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(graph, str(output_path), save_as_external_data=False)
    Path(f"{output_path}.data").unlink(missing_ok=True)


def _embed_metadata(output_path: Path, metadata: Mapping[str, str]) -> None:
    graph = onnx.load(str(output_path))
    del graph.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    graph.producer_name = "SoundEx"
    graph.producer_version = ARTIFACT_SCHEMA_VERSION
    graph.domain = "soundex.audio"
    graph.model_version = 1
    onnx.save_model(graph, str(output_path), save_as_external_data=False)


def validate_onnx_contract(path: str | Path, expected_metadata: Mapping[str, str]) -> None:
    """Require a self-contained, opset-17 graph with the exact static contract."""
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    opsets = {item.domain: item.version for item in graph.opset_import}
    if opsets.get("") != OPSET_VERSION:
        raise ExportValidationError(f"expected ONNX opset {OPSET_VERSION}, got {opsets}")
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ExportValidationError("ONNX graph must expose exactly one input and one output")
    expected_shape = tensor_shape(
        int(expected_metadata["soundex.fft_size"]),
        int(expected_metadata["soundex.hop_size"]),
    )
    _validate_value_info(graph.graph.input[0], INPUT_NAME, expected_shape)
    _validate_value_info(graph.graph.output[0], OUTPUT_NAME, expected_shape)

    actual_metadata = {item.key: item.value for item in graph.metadata_props}
    missing = sorted(REQUIRED_METADATA_KEYS - set(actual_metadata))
    if missing:
        raise ExportValidationError(f"ONNX metadata is missing {missing[0]}")
    if actual_metadata != dict(expected_metadata):
        differing = sorted(
            key
            for key in set(actual_metadata) | set(expected_metadata)
            if actual_metadata.get(key) != expected_metadata.get(key)
        )
        raise ExportValidationError(f"ONNX metadata mismatch at {differing[0]}")
    if any(
        initializer.data_location == TensorProto.EXTERNAL for initializer in graph.graph.initializer
    ):
        raise ExportValidationError("ONNX artifact contains external tensor data")


def _validate_value_info(
    value_info: Any,
    expected_name: str,
    expected_shape: tuple[int, int, int],
) -> None:
    if value_info.name != expected_name:
        raise ExportValidationError(
            f"expected tensor name {expected_name!r}, got {value_info.name!r}"
        )
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != TensorProto.FLOAT:
        raise ExportValidationError(f"{expected_name} must be float32")
    dimensions = tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ExportValidationError(f"{expected_name} must be rank 4")
    if dimensions[0].dim_param != "batch" or dimensions[0].HasField("dim_value"):
        raise ExportValidationError(f"{expected_name} batch axis must be the only dynamic axis")
    actual_static = tuple(int(dimension.dim_value) for dimension in dimensions[1:])
    if actual_static != expected_shape or any(dimension.dim_param for dimension in dimensions[1:]):
        raise ExportValidationError(
            f"{expected_name} must have shape [batch, {', '.join(map(str, expected_shape))}], "
            f"got {actual_static}"
        )


def _load_optional_config(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ExportValidationError(f"expected a YAML mapping in {path}")
    return normalize_config(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="../models/soundex-v1.onnx")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional profile to compare with checkpoint semantics; never overrides them",
    )
    args = parser.parse_args()
    export_onnx(
        args.checkpoint,
        args.output,
        expected_config_path=args.config,
    )


if __name__ == "__main__":
    main()
