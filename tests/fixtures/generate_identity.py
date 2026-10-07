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


if __name__ == "__main__":
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
