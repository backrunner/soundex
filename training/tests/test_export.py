"""End-to-end checkpoint-to-ONNX export contract tests."""

from __future__ import annotations

import copy
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml
from onnx import TensorProto

from artifact_contract import (
    INPUT_NAME,
    MAX_ARTIFACT_BYTES,
    OUTPUT_NAME,
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
)
from checkpoint import CheckpointSchemaError, load_checkpoint
from export_onnx import (
    export_onnx,
    sha256_file,
)
from export_validation import deterministic_parity_inputs
from models.generator import SoundExGenerator


@pytest.fixture
def exported_model(tmp_path_factory, checkpoint_factory):
    directory = tmp_path_factory.mktemp("exported-model")
    checkpoint_path, _ = checkpoint_factory("export-source.pth")
    output_path = directory / "soundex.onnx"
    result = export_onnx(checkpoint_path, output_path)
    return output_path, result, checkpoint_path


def _dimensions(value_info: Any) -> list[str | int]:
    dimensions: list[str | int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        dimensions.append(dimension.dim_param or int(dimension.dim_value))
    return dimensions


def test_export_has_exact_graph_and_metadata_contract(exported_model) -> None:
    path, result, _ = exported_model
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)

    assert _dimensions(graph.graph.input[0]) == ["batch", 2, 1, 513]
    assert _dimensions(graph.graph.output[0]) == ["batch", 2, 1, 513]
    assert graph.graph.input[0].name == INPUT_NAME
    assert graph.graph.output[0].name == OUTPUT_NAME
    assert graph.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert {item.domain: item.version for item in graph.opset_import}[""] == 17
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert set(metadata) == REQUIRED_METADATA_KEYS
    assert metadata["soundex.artifact_schema"] == "1.2"
    assert metadata["soundex.fft_size"] == "1024"
    assert metadata["soundex.hop_size"] == "512"
    assert metadata["soundex.source_checkpoint_sha256"]
    assert metadata["soundex.crossover_width_hz"] == "1000.0"
    assert result.sha256 == sha256_file(path)
    assert result.size_bytes == path.stat().st_size < MAX_ARTIFACT_BYTES
    assert result.parameter_count <= 2_000_000


def test_exported_model_runs_dynamic_batch_one_and_two(exported_model) -> None:
    path, _, _ = exported_model
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for batch in (1, 2):
        input_array = np.zeros((batch, 2, 1, 513), dtype=np.float32)
        output = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        assert output.shape == input_array.shape
        assert np.isfinite(output).all()


def test_legacy_2048_checkpoint_exports_and_runs_in_rust(
    tmp_path: Path,
    checkpoint_factory,
    resolved_config: dict[str, Any],
) -> None:
    legacy_config = copy.deepcopy(resolved_config)
    legacy_config["audio"]["fft_size"] = 2048
    checkpoint_path, _ = checkpoint_factory("legacy-export.pth", config=legacy_config)
    model_path = tmp_path / "legacy.onnx"

    export_onnx(checkpoint_path, model_path)

    graph = onnx.load(str(model_path), load_external_data=False)
    assert _dimensions(graph.graph.input[0]) == ["batch", 2, 1, 1025]
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["soundex.artifact_schema"] == "1.2"
    assert metadata["soundex.fft_size"] == "2048"
    input_tensor = dict(deterministic_parity_inputs(fft_size=2048, hop_size=512))[
        "random_finite_spectra"
    ]
    model = _checkpoint_generator(checkpoint_path)
    with torch.no_grad():
        expected = model(input_tensor).numpy()
    input_path = tmp_path / "legacy.input.sxt"
    expected_path = tmp_path / "legacy.expected.sxt"
    _write_tensor(input_path, input_tensor.numpy())
    _write_tensor(expected_path, expected)

    result = _rust_parity(model_path, input_path, expected_path)

    assert result.returncode == 0, result.stderr
    assert "parity passed" in result.stdout


def test_external_config_is_comparison_only_and_rejects_mismatch(
    tmp_path: Path,
    checkpoint_factory,
    resolved_config: dict[str, Any],
) -> None:
    checkpoint_path, _ = checkpoint_factory("config-source.pth")
    config = copy.deepcopy(resolved_config)
    config["audio"]["fft_size"] = 4096
    config_path = tmp_path / "mismatch.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "must-not-exist.onnx"

    with pytest.raises(CheckpointSchemaError, match="fft_size"):
        export_onnx(checkpoint_path, output_path, expected_config_path=config_path)

    assert not output_path.exists()


def _write_tensor(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f4")
    if array.ndim != 4:
        raise ValueError(f"SXT tensor must be rank 4, got {array.shape}")
    header = struct.pack("<4s4I", b"SXT1", *array.shape)
    path.write_bytes(header + array.tobytes(order="C"))


def _rust_parity(
    model_path: Path,
    input_path: Path,
    expected_path: Path,
) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "soundex-core",
            "--bin",
            "soundex-model-parity",
            "--",
            str(model_path),
            str(input_path),
            str(expected_path),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _checkpoint_generator(checkpoint_path: Path) -> SoundExGenerator:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["model"]["generator_config"]
    model = SoundExGenerator(
        channels=config["channels"],
        bottleneck_blocks=config["bottleneck_blocks"],
        expand_ratio=config["expand_ratio"],
    )
    model.load_state_dict(checkpoint["model"]["generator_state"])
    model.eval()
    return model


def test_python_ort_and_rust_match_fixed_and_real_stft_tensors(
    tmp_path: Path, exported_model
) -> None:
    model_path, _, checkpoint_path = exported_model
    model = _checkpoint_generator(checkpoint_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cases = dict(deterministic_parity_inputs())

    for case_name in ("random_finite_spectra", "near_full_scale_sine"):
        input_tensor = cases[case_name]
        with torch.no_grad():
            python_output = model(input_tensor).numpy()
        ort_output = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor.numpy()})[0]
        np.testing.assert_allclose(ort_output, python_output, rtol=0, atol=1e-5)
        input_path = tmp_path / f"{case_name}.input.sxt"
        expected_path = tmp_path / f"{case_name}.expected.sxt"
        _write_tensor(input_path, input_tensor.numpy())
        _write_tensor(expected_path, python_output)

        result = _rust_parity(model_path, input_path, expected_path)

        assert result.returncode == 0, result.stderr
        assert "parity passed" in result.stdout

    swapped_path = tmp_path / "swapped.input.sxt"
    random_input = cases["random_finite_spectra"].numpy()
    _write_tensor(swapped_path, random_input[:, ::-1].copy())
    mismatch = _rust_parity(
        model_path,
        swapped_path,
        tmp_path / "random_finite_spectra.expected.sxt",
    )
    assert mismatch.returncode != 0
    assert "parity mismatch" in mismatch.stderr


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("soundex.artifact_schema", "1.0"),
        ("soundex.input_channels", "phase_radians,log_magnitude_db"),
        ("soundex.input_shape", "batch,2,2,513"),
        ("soundex.output_shape", "batch,2,1,1025"),
        ("soundex.crossover_width_hz", "2000.0"),
    ],
)
def test_rust_rejects_tampered_contract_metadata(
    tmp_path: Path,
    exported_model,
    key: str,
    value: str,
) -> None:
    model_path, _, _ = exported_model
    graph = onnx.load(str(model_path))
    metadata = {item.key: item for item in graph.metadata_props}
    metadata[key].value = value
    tampered_path = tmp_path / "tampered.onnx"
    onnx.save(graph, tampered_path)
    input_value = np.zeros((1, 2, 1, 513), dtype=np.float32)
    input_path = tmp_path / "input.sxt"
    expected_path = tmp_path / "expected.sxt"
    _write_tensor(input_path, input_value)
    _write_tensor(expected_path, input_value)

    result = _rust_parity(tampered_path, input_path, expected_path)

    assert result.returncode != 0
    assert "model contract mismatch" in result.stderr


def test_ort_perturbation_fails_without_publishing(tmp_path: Path, checkpoint_factory) -> None:
    checkpoint_path, _ = checkpoint_factory("perturb-source.pth")
    output_path = tmp_path / "failed.onnx"

    def perturb(case_name: str, output: np.ndarray) -> np.ndarray:
        if case_name == "silence":
            return output + np.float32(1e-3)
        return output

    with pytest.raises(ExportValidationError, match=r"silence.*mismatch"):
        export_onnx(checkpoint_path, output_path, ort_output_transform=perturb)

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.staging.onnx"))
    assert not list(tmp_path.glob("*.staging.onnx.data"))


def test_schema_zero_export_is_rejected_without_output(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy.pth"
    output_path = tmp_path / "legacy.onnx"
    torch.save({"generator": {}}, checkpoint_path)

    with pytest.raises(CheckpointSchemaError, match="schema-0"):
        export_onnx(checkpoint_path, output_path)

    assert not output_path.exists()


def test_tampered_architecture_checkpoint_is_not_exported(
    tmp_path: Path, checkpoint_factory
) -> None:
    _, checkpoint = checkpoint_factory("valid-before-tamper.pth")
    checkpoint["model"]["architecture"]["major"] = 2
    checkpoint_path = tmp_path / "tampered.pth"
    torch.save(checkpoint, checkpoint_path)
    output_path = tmp_path / "tampered.onnx"

    with pytest.raises(CheckpointSchemaError, match="architecture major"):
        export_onnx(checkpoint_path, output_path)

    assert not output_path.exists()
