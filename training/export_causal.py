"""Export explicit-state causal study checkpoints under artifact schema 2.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from artifact_contract import MAX_ARTIFACT_BYTES, MAX_PARAMETERS
from evaluation.reporting import sha256_file
from models.causal_restorer import CausalRestorer


def verify_stateful_export(model: CausalRestorer, path: Path) -> None:
    """Fail closed on sequential batch-one/two parity, including a reset and silence.

    Compare Cartesian output: phase wrap at +-pi is physically equivalent. State
    must independently agree; a first-frame-only check cannot validate recurrence.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(20261011)
    for rate in (44100, 48000):
        for batch in (1, 2):
            state = np.zeros((batch, model.state_size), np.float32)
            reference_state = torch.from_numpy(state.copy())
            rates = np.full((batch, 1), rate, np.float32)
            for hop in range(80):
                if hop == 70:
                    state.fill(0)
                    reference_state.zero_()
                frame = rng.normal(size=(batch, 2, 1, 129)).astype(np.float32)
                frame[:, 0] = frame[:, 0] * 15 - 30
                if hop < 4:
                    frame[0, 0], frame[0, 1] = -200, 0
                with torch.no_grad():
                    expected, reference_state = model(
                        torch.from_numpy(frame), reference_state, torch.from_numpy(rates)
                    )
                actual, state = session.run(
                    None, {"input_features": frame, "input_state": state, "sample_rate_hz": rates}
                )

                def cartesian(features):
                    return np.power(10.0, features[:, 0] / 20.0) * np.exp(1j * features[:, 1])

                np.testing.assert_allclose(
                    cartesian(actual), cartesian(expected.numpy()), atol=1e-5, rtol=1e-4
                )
                np.testing.assert_allclose(state, reference_state.numpy(), atol=2e-5, rtol=2e-5)


def export_causal(checkpoint_path: Path, output_path: Path) -> dict[str, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "soundex-causal-study-v1":
        raise ValueError("unsupported causal checkpoint")
    model = CausalRestorer(checkpoint["kind"], checkpoint["width"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > MAX_PARAMETERS:
        raise ValueError("parameter budget exceeded")
    provenance = checkpoint["provenance"]
    for name in ("data_recipe_sha256", "manifest_set_sha256"):
        value = provenance[name]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid provenance {name}")
    configuration = {
        "kind": model.kind,
        "width": model.width,
        "training": checkpoint.get("config", {}),
    }
    metadata = {
        "artifact_schema": "2.0",
        "model_architecture": "soundex-causal-complex",
        "model_architecture_version": "1.0",
        "input_name": "input_features",
        "output_name": "output_features",
        "tensor_layout": "BCTF",
        "input_shape": "batch,2,1,129",
        "output_shape": "batch,2,1,129",
        "input_channels": "log_magnitude_db,phase_radians",
        "output_channels": "log_magnitude_db,phase_radians",
        "sample_rates": "44100,48000",
        "fft_size": "256",
        "hop_size": "128",
        "crossover_width_hz": "1000.0",
        "window": "hann",
        "window_periodic": "true",
        "magnitude_scale": "decibels",
        "db_formula": "20*log10(max(abs(stft),1e-10))",
        "db_floor": "-200.0",
        "phase_units": "radians",
        "phase_range": "[-pi,pi]",
        "context_frames": "1",
        "causal": "true",
        "stateless": "false",
        "state_size": str(model.state_size),
        "state_layout": "batch,state",
        "state_reset": "zeros",
        "warmup_frames": "64",
        "high_band_gain": "model",
        "phase_policy": "model",
        "temporal_kind": model.kind,
        "parameter_count": str(parameters),
        "sample_rate_input_units": "Hz",
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        **{name: provenance[name] for name in ("data_recipe_sha256", "manifest_set_sha256")},
    }
    metadata = {"soundex." + name: value for name, value in metadata.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(".partial.onnx")
    names = ["input_features", "input_state", "sample_rate_hz", "output_features", "output_state"]
    try:
        torch.onnx.export(
            model,
            (torch.zeros(2, 2, 1, 129), model.initial_state(2), torch.full((2, 1), 48000.0)),
            str(staged),
            input_names=names[:3],
            output_names=names[3:],
            dynamic_axes={name: {0: "batch"} for name in names},
            opset_version=17,
            dynamo=False,
        )
        graph = onnx.load(staged)
        onnx.helper.set_model_props(graph, metadata)
        # Freeze the declared step axes, then verify them by runtime tests before use.
        for value, dims in [
            (graph.graph.output[0], (2, 1, 129)),
            (graph.graph.output[1], (model.state_size,)),
        ]:
            for dim, size in zip(value.type.tensor_type.shape.dim[1:], dims, strict=True):
                dim.dim_param = ""
                dim.dim_value = size
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, staged)
        if staged.stat().st_size >= MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds 8 MiB")
        verify_stateful_export(model, staged)
        staged.replace(output_path)
    finally:
        staged.unlink(missing_ok=True)
    return metadata
