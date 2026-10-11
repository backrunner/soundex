"""Export untrained causal prototypes and check sequential PyTorch/ORT/Rust parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import RustStreamEvaluator
from export_causal import export_causal
from models.causal_restorer import CausalRestorer


def cartesian(features):
    magnitude = np.power(10.0, features[:, 0] / 20.0)
    return magnitude * np.exp(1j * features[:, 1])


def run(output: Path, manifests: list[Path], binary: Path, stream_binary: Path) -> dict:
    torch.set_num_threads(1)
    manifest_hashes = {str(p.resolve()): sha256_file(p) for p in manifests}
    recipes = {json.loads(p.open().readline())["recipe_hash"] for p in manifests}
    if len(recipes) != 1:
        raise ValueError("manifests must share a recipe")
    provenance = {
        "data_recipe_sha256": recipes.pop(),
        "manifest_set_sha256": hashlib.sha256(
            json.dumps(manifest_hashes, sort_keys=True).encode()
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("gru", "tcn"):
        torch.manual_seed(20261011)
        model = CausalRestorer(kind).eval()
        root = output / kind
        root.mkdir(exist_ok=True)
        checkpoint = root / "untrained.pth"
        torch.save(
            {
                "format": "soundex-causal-study-v1",
                "kind": kind,
                "width": model.width,
                "model": model.state_dict(),
                "provenance": provenance,
                "config": {"untrained_probe": True},
            },
            checkpoint,
        )
        model_path = root / "untrained.onnx"
        export_causal(checkpoint, model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        cases = []
        for rate in (44100, 48000):
            for channels in (1, 2):
                rng = np.random.default_rng(934)
                features = rng.normal(size=(96, channels, 2, 1, 129)).astype(np.float32)
                features[:, :, 0] = features[:, :, 0] * 15 - 30
                features[:8, 0, 0] = -200
                features[:8, 0, 1] = 0
                sr = np.full((channels, 1), rate, np.float32)
                state = np.zeros((channels, model.state_size), np.float32)
                torch_state = torch.from_numpy(state.copy())
                expected = []
                state_error = 0.0
                ort_error = 0.0
                for index, frame in enumerate(features):
                    if index == 75:
                        state.fill(0)
                        torch_state.zero_()
                    with torch.no_grad():
                        pred, torch_state = model(
                            torch.from_numpy(frame), torch_state, torch.from_numpy(sr)
                        )
                    actual, state = session.run(
                        None, {"input_features": frame, "input_state": state, "sample_rate_hz": sr}
                    )
                    torch_prediction = pred.numpy()
                    np.testing.assert_allclose(
                        cartesian(actual), cartesian(torch_prediction), atol=1e-5, rtol=1e-4
                    )
                    np.testing.assert_allclose(state, torch_state.numpy(), atol=2e-5, rtol=2e-5)
                    state_error = max(state_error, float(np.abs(state - torch_state.numpy()).max()))
                    ort_error = max(
                        ort_error,
                        float(np.abs(cartesian(actual) - cartesian(torch_prediction)).max()),
                    )
                    expected.append(torch_prediction)
                input_path, result_path = root / "parity-input.json", root / "parity-output.json"
                input_path.write_text(
                    json.dumps(
                        {
                            "sample_rate": rate,
                            "channels": channels,
                            "resets": [75],
                            "frames": features.reshape(96, -1).tolist(),
                        }
                    )
                )
                subprocess.run(
                    [str(binary), str(model_path), str(input_path), str(result_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rust = json.loads(result_path.read_text())
                if rust["runs"] != 96:
                    raise RuntimeError("Rust missed state updates")
                actual = np.asarray(rust["frames"], np.float32).reshape(features.shape)
                rust_error = 0.0
                for actual_frame, expected_frame in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(
                        cartesian(actual_frame), cartesian(expected_frame), atol=1e-5, rtol=1e-4
                    )
                    rust_error = max(
                        rust_error,
                        float(np.abs(cartesian(actual_frame) - cartesian(expected_frame)).max()),
                    )
                # Real framing and DSP, with a silent channel and partial final hop.
                pcm = (0.15 * np.sin(np.arange(rate + 37) * 0.07)).astype(np.float32)[:, None]
                if channels == 2:
                    pcm = np.concatenate((np.zeros_like(pcm), pcm), axis=1)
                evaluator = RustStreamEvaluator(stream_binary)
                offline = evaluator.process(
                    model_path, pcm, rate, mode="offline", high_band_gain="model"
                )
                chunked = evaluator.process(
                    model_path,
                    pcm,
                    rate,
                    mode="chunked",
                    chunk_frames=(31, 512, 997, 7),
                    high_band_gain="model",
                )
                np.testing.assert_array_equal(offline.audio, chunked.audio)
                if channels == 2:
                    np.testing.assert_array_equal(offline.audio[:, 0], 0)
                cases.append(
                    {
                        "sample_rate": rate,
                        "channels": channels,
                        "hops": 96,
                        "resets": [75],
                        "torch_ort_complex_max_abs": ort_error,
                        "torch_rust_complex_max_abs": rust_error,
                        "torch_ort_state_max_abs": state_error,
                        "chunk_max_abs": 0.0,
                    }
                )
        results[kind] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "bytes": model_path.stat().st_size,
            "onnx_sha256": sha256_file(model_path),
            "state_size": model.state_size,
            "cases": cases,
        }
        input_path.unlink()
        result_path.unlink()
    report = {
        "untrained": True,
        "quality_evaluated": False,
        "models": results,
        "manifests": manifest_hashes,
        "rust_parity_sha256": sha256_file(binary),
        "rust_stream_sha256": sha256_file(stream_binary),
    }
    write_json_report(report, output / "probe.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output.resolve(),
                args.manifest,
                args.binary.resolve(),
                args.stream_binary.resolve(),
            ),
            indent=2,
        )
    )
