"""Own analytic four-input history fixture, not trained restoration weights.

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path

import onnx
from onnx import TensorProto as T
from onnx import helper as h

graph = h.make_graph(
    [
        h.make_node("ReduceMean", ["features"], ["mean"], axes=[1, 2, 3], keepdims=0),
        h.make_node("Unsqueeze", ["mean", "axis1"], ["current"]),
        h.make_node("Mul", ["state", "decay"], ["past"]),
        h.make_node("Mul", ["current", "step"], ["increment"]),
        h.make_node("Add", ["past", "increment"], ["next_state"]),
        h.make_node("Unsqueeze", ["state", "axes23"], ["history"]),
        h.make_node("Mul", ["history", "mag_only"], ["correction"]),
        h.make_node("Add", ["features", "correction"], ["output"]),
    ],
    "soundex_context_worker_probe",
    [
        h.make_tensor_value_info("features", T.FLOAT, ["batch", 2, 1, 129]),
        h.make_tensor_value_info("context", T.FLOAT, ["batch", 1, 1538]),
        h.make_tensor_value_info("condition", T.FLOAT, ["batch", 1, 3]),
        h.make_tensor_value_info("state", T.FLOAT, ["batch", 1]),
    ],
    [
        h.make_tensor_value_info("output", T.FLOAT, ["batch", 2, 1, 129]),
        h.make_tensor_value_info("next_state", T.FLOAT, ["batch", 1]),
    ],
    initializer=[
        h.make_tensor("axis1", T.INT64, [1], [1]),
        h.make_tensor("axes23", T.INT64, [2], [2, 3]),
        h.make_tensor("decay", T.FLOAT, [], [0.9]),
        h.make_tensor("step", T.FLOAT, [], [0.01]),
        h.make_tensor("mag_only", T.FLOAT, [1, 2, 1, 1], [1, 0]),
    ],
)
model = h.make_model(
    graph,
    producer_name="SoundEx synthetic worker test",
    opset_imports=[h.make_opsetid("", 17)],
    ir_version=9,
)
h.set_model_props(
    model,
    {
        "soundex.experimental": "p20-context-head-diagnostic",
        "soundex.deployable": "false",
        "soundex.width": "1",
    },
)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, Path(__file__).with_name("context-worker-probe.onnx"))
