"""Registration isolation and diagnostic legacy compatibility."""

import onnx
import pytest
from onnx import TensorProto, helper

from evaluation.legacy_context import wrap_legacy
from scripts.run_spectral_training import training_roles


def test_legacy_wrapper_preserves_values_and_state(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input_features"], ["output_features"])],
        "identity",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch", 2, 1, 129])],
        [helper.make_tensor_value_info("output_features", TensorProto.FLOAT, ["batch", 2, 1, 129])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    source, destination = tmp_path / "source.onnx", tmp_path / "wrapped.onnx"
    onnx.save(model, source)
    report = wrap_legacy(source, destination)
    assert report["max_feature_error"] == 0
    assert report["dummy_state_exact"]
    properties = {p.key: p.value for p in onnx.load(destination).metadata_props}
    assert properties["soundex.deployable"] == "false"


def test_calibration_uses_connected_aliases_and_never_review():
    rows = {str(i): {"track_id": f"track-{i}", "source_checksum": f"source-{i}"} for i in range(80)}
    # One transitive recording/source component must never be split.
    rows["1"]["source_checksum"] = rows["0"]["source_checksum"]
    rows["2"]["track_id"] = rows["1"]["track_id"]
    old = {
        "cases": {
            "fit": [{"samples": {"calibration": [{"row": rows[str(i)]} for i in range(79)]}}],
            "review": [{"samples": {"calibration": [{"row": rows["79"]}]}}],
        }
    }
    roles = training_roles(rows, old)
    assert all("79" not in values for values in roles.values())
    assert any({"0", "1", "2"} <= set(values) for values in roles.values())
    assert set(roles["optimization"]) | set(roles["calibration"]) == set(rows) - {"79"}
    assert not set(roles["optimization"]) & set(roles["calibration"])
    old["cases"]["review"][0]["samples"]["calibration"].append({"row": rows["0"]})
    with pytest.raises(ValueError, match="leaks"):
        training_roles(rows, old)
