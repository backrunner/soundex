"""Small-window artifacts require explicit versioning and exact FFT/hop binding."""

from pathlib import Path

import onnx
import pytest

from artifact_contract import ExportValidationError, frame_contract_from_metadata, tensor_shape


def test_schema_downgrade_cannot_mislabel_a_small_window_model() -> None:
    graph = onnx.load(str(Path(__file__).parents[2] / "tests/fixtures/low-latency-identity.onnx"))
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    assert frame_contract_from_metadata(metadata) == (256, 128)
    metadata["soundex.artifact_schema"] = "1.2"
    with pytest.raises(ExportValidationError, match=r"schema 1\.3"):
        frame_contract_from_metadata(metadata)


def test_hop_changes_cannot_keep_the_old_fft() -> None:
    assert tensor_shape(256, 128) == (2, 1, 129)
    assert tensor_shape(512, 256) == (2, 1, 257)
    with pytest.raises(ExportValidationError, match="unsupported FFT/hop"):
        tensor_shape(1024, 128)
