"""Generate the tiny dynamic-batch ONNX identity model used by Rust tests."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

METADATA = {
    "soundex.artifact_schema": "1.2",
    "soundex.model_architecture": "soundex-generator",
    "soundex.model_architecture_version": "1.0",
    "soundex.input_name": "input_features",
    "soundex.output_name": "output_features",
    "soundex.tensor_layout": "BCTF",
    "soundex.input_shape": "batch,2,1,513",
    "soundex.output_shape": "batch,2,1,513",
    "soundex.input_channels": "log_magnitude_db,phase_radians",
    "soundex.output_channels": "log_magnitude_db,phase_radians",
    "soundex.sample_rates": "44100,48000",
    "soundex.fft_size": "1024",
    "soundex.hop_size": "512",
    "soundex.crossover_width_hz": "1000.0",
    "soundex.window": "hann",
    "soundex.window_periodic": "true",
    "soundex.magnitude_scale": "decibels",
    "soundex.db_formula": "20*log10(max(abs(stft),1e-10))",
    "soundex.db_floor": "-200.0",
    "soundex.phase_units": "radians",
    "soundex.phase_range": "[-pi,pi]",
    "soundex.context_frames": "1",
    "soundex.causal": "true",
    "soundex.stateless": "true",
    "soundex.source_checkpoint_sha256": "0" * 64,
    "soundex.resolved_config_sha256": "1" * 64,
    "soundex.data_recipe_sha256": "2" * 64,
    "soundex.manifest_set_sha256": "3" * 64,
}


def generate(fft_size: int, hop_size: int, filename: str) -> None:
    shape = ["batch", 2, 1, fft_size // 2 + 1]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input_features"], ["output_features"])],
        "soundex_identity_fixture",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output_features", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="SoundEx test fixture",
        domain="soundex.audio",
        model_version=1,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    metadata = dict(METADATA)
    metadata.update(
        {
            "soundex.artifact_schema": "1.3" if fft_size < 1024 else "1.2",
            "soundex.input_shape": f"batch,2,1,{fft_size // 2 + 1}",
            "soundex.output_shape": f"batch,2,1,{fft_size // 2 + 1}",
            "soundex.fft_size": str(fft_size),
            "soundex.hop_size": str(hop_size),
        }
    )
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name(filename))


def generate_temporal() -> None:
    """Synthetic history-sensitive graph, not a trained/restoration model."""
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["input_features", "old_index"], ["old"], axis=2),
            helper.make_node("Gather", ["input_features", "now_index"], ["now"], axis=2),
            helper.make_node("Mul", ["old", "old_weight"], ["old_part"]),
            helper.make_node("Mul", ["now", "now_weight"], ["now_part"]),
            helper.make_node("Add", ["old_part", "now_part"], ["output_features"]),
        ],
        "soundex_temporal_probe_fixture",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch", 2, 9, 129])],
        [helper.make_tensor_value_info("output_features", TensorProto.FLOAT, ["batch", 2, 1, 129])],
        initializer=[
            helper.make_tensor("old_index", TensorProto.INT64, [1], [0]),
            helper.make_tensor("now_index", TensorProto.INT64, [1], [8]),
            helper.make_tensor("old_weight", TensorProto.FLOAT, [], [0.1]),
            helper.make_tensor("now_weight", TensorProto.FLOAT, [], [0.9]),
        ],
    )
    model = helper.make_model(
        graph, producer_name="SoundEx synthetic history test", domain="soundex.audio",
        model_version=1, opset_imports=[helper.make_opsetid("", 17)], ir_version=9,
    )
    metadata = dict(METADATA)
    metadata.update({
        "soundex.artifact_schema": "1.4", "soundex.model_architecture_version": "1.4",
        "soundex.input_shape": "batch,2,9,129", "soundex.output_shape": "batch,2,1,129",
        "soundex.fft_size": "256", "soundex.hop_size": "128", "soundex.context_frames": "9",
    })
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name("temporal-probe.onnx"))


def generate_stateful() -> None:
    """Analytic leaky memory, not trained weights. History affects magnitude only."""
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["input_features"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["mean", "axis1"], ["current"]),
        helper.make_node("Mul", ["input_state", "decay"], ["past"]),
        helper.make_node("Mul", ["current", "step"], ["increment"]),
        helper.make_node("Add", ["past", "increment"], ["output_state"]),
        helper.make_node("Unsqueeze", ["input_state", "axes23"], ["history"]),
        helper.make_node("Mul", ["history", "mag_only"], ["correction"]),
        helper.make_node("Add", ["input_features", "correction"], ["output_features"]),
    ], "soundex_stateful_probe", [
        helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch", 2, 1, 129]),
        helper.make_tensor_value_info("input_state", TensorProto.FLOAT, ["batch", 1]),
        helper.make_tensor_value_info("sample_rate_hz", TensorProto.FLOAT, ["batch", 1]),
    ], [
        helper.make_tensor_value_info("output_features", TensorProto.FLOAT, ["batch", 2, 1, 129]),
        helper.make_tensor_value_info("output_state", TensorProto.FLOAT, ["batch", 1]),
    ], initializer=[
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("axes23", TensorProto.INT64, [2], [2, 3]),
        helper.make_tensor("decay", TensorProto.FLOAT, [], [0.9]),
        helper.make_tensor("step", TensorProto.FLOAT, [], [0.01]),
        helper.make_tensor("mag_only", TensorProto.FLOAT, [1, 2, 1, 1], [1, 0]),
    ])
    model = helper.make_model(graph, producer_name="SoundEx synthetic state test",
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    metadata = dict(METADATA)
    metadata.update({
        "soundex.artifact_schema": "2.0", "soundex.model_architecture": "soundex-causal-complex",
        "soundex.input_shape": "batch,2,1,129", "soundex.output_shape": "batch,2,1,129",
        "soundex.fft_size": "256", "soundex.hop_size": "128", "soundex.stateless": "false",
        "soundex.state_size": "1", "soundex.state_layout": "batch,state",
        "soundex.state_reset": "zeros", "soundex.temporal_kind": "gru",
        "soundex.warmup_frames": "64", "soundex.high_band_gain": "model",
        "soundex.phase_policy": "model", "soundex.parameter_count": "1",
        "soundex.sample_rate_input_units": "Hz",
    })
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name("stateful-probe.onnx"))


if __name__ == "__main__":
    generate_stateful()
    generate_temporal()
    generate(1024, 512, "identity.onnx")
    generate(256, 128, "low-latency-identity.onnx")
    # A valid contract with deliberately invalid inference results. Never deploy.
    model = onnx.load(Path(__file__).with_name("low-latency-identity.onnx"))
    del model.graph.node[:]
    model.graph.node.append(
        helper.make_node("Mul", ["input_features", "bad"], ["output_features"])
    )
    model.graph.initializer.append(
        helper.make_tensor("bad", TensorProto.FLOAT, [], [float("nan")])
    )
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name("nonfinite-output.onnx"))
