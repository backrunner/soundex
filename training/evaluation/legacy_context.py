"""Lossless diagnostic ABI adapter for historical stateless ONNX models.

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from evaluation.head_export import session_for
from evaluation.reporting import sha256_file


def wrap_legacy(source: Path, destination: Path):
    """Rename I/O and pass through dummy state; retain all original model nodes.

    Context and condition are unused. Width=1 declares 63 dummy state values for
    the diagnostic runner only. This is not a production conversion or history.
    Actual native rendering uses the same warmup, gating, limiter and fusion as
    the P38/P53 heads, isolating model effects from different historical policies.
    """
    graph = onnx.load(source)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("single-input/output stateless legacy model required")
    renames = {graph.graph.input[0].name: "features", graph.graph.output[0].name: "output"}
    for node in graph.graph.node:
        for names in (node.input, node.output):
            for i, name in enumerate(names):
                names[i] = renames.get(name, name)
    graph.graph.input[0].name = "features"
    graph.graph.output[0].name = "output"
    graph.graph.input.extend(
        [
            helper.make_tensor_value_info("context", TensorProto.FLOAT, ["batch", 1, 1538]),
            helper.make_tensor_value_info("condition", TensorProto.FLOAT, ["batch", 1, 3]),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, ["batch", 63]),
        ]
    )
    graph.graph.node.append(helper.make_node("Identity", ["state"], ["next_state"]))
    graph.graph.output.append(
        helper.make_tensor_value_info("next_state", TensorProto.FLOAT, ["batch", 63])
    )
    helper.set_model_props(
        graph,
        {
            "soundex.experimental": "p20-context-head-diagnostic",
            "soundex.deployable": "false",
            "soundex.width": "1",
            "soundex.study": "p54-stateless-legacy-wrapper",
            "soundex.legacy_sha256": sha256_file(source),
            "soundex.state_clock": "dummy-pass-through-no-learned-history",
        },
    )
    onnx.checker.check_model(graph, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, destination)
    original, wrapped = session_for(source), session_for(destination)
    rng = np.random.default_rng(54)
    maximum = 0.0
    for channels in (1, 2, 5):
        features = rng.normal(size=(channels, 2, 1, 129)).astype(np.float32)
        features[:, 0] = features[:, 0] * 20 - 60
        context = rng.normal(size=(channels, 1, 1538)).astype(np.float32)
        state = rng.normal(size=(channels, 63)).astype(np.float32)
        condition = np.ones((channels, 1, 3), np.float32)
        expected = original.run(None, {original.get_inputs()[0].name: features})[0]
        actual, next_state = wrapped.run(
            None,
            {
                "features": features,
                "context": context,
                "condition": condition,
                "state": state,
            },
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(next_state, state)
        maximum = max(maximum, float(np.abs(actual - expected).max()))
    return {
        "source_sha256": sha256_file(source),
        "onnx_sha256": sha256_file(destination),
        "max_feature_error": maximum,
        "dummy_state_exact": True,
    }
