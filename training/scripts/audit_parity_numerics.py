# SPDX-License-Identifier: Apache-2.0
"""Diagnose FP32 backend rounding against FP64; does not certify model quality."""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import load_checkpoint
from evaluation.reporting import write_json_report
from export_onnx import (
    _build_generator,
    _embed_metadata,
    _export_graph,
    artifact_metadata,
    sha256_file,
)
from export_validation import deterministic_parity_inputs
from parity_metrics import PARITY_POLICY, POLICY_SHA256, compare_features, validate_features


def audit(checkpoint_path: Path, output: Path) -> dict[str, object]:
    """Record case metrics at each optimization level without releasing weights."""
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    model = _build_generator(checkpoint)
    reference_model = _build_generator(checkpoint).double()
    feature = checkpoint["feature_contract"]
    fft_size, hop_size = int(feature["fft_size"]), int(feature["hop_size"])
    cases = deterministic_parity_inputs(fft_size=fft_size, hop_size=hop_size)
    references = []
    for name, tensor in cases:
        with torch.no_grad():
            expected = model(tensor).numpy()
            reference64 = reference_model(tensor.double()).numpy()
        references.append((name, tensor, expected, reference64))
    levels = {}
    with tempfile.TemporaryDirectory(prefix="soundex-numerical-audit-") as directory:
        graph = Path(directory) / "diagnostic.onnx"
        _export_graph(model, graph, (2, 1, fft_size // 2 + 1))
        _embed_metadata(graph, artifact_metadata(checkpoint, checkpoint_hash))
        artifact_hash = sha256_file(graph)
        for level in (
            "ORT_DISABLE_ALL",
            "ORT_ENABLE_BASIC",
            "ORT_ENABLE_EXTENDED",
            "ORT_ENABLE_ALL",
        ):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
            session = ort.InferenceSession(str(graph), sess_options=options)
            rows = []
            for name, tensor, expected, reference64 in references:
                actual = session.run(None, {"input_features": tensor.numpy()})[0]
                metrics = compare_features(expected, actual)
                try:
                    validate_features(expected, actual, label=name)
                except RuntimeError as error:
                    outcome: str | bool = str(error)
                else:
                    outcome = True
                expected_spectrum = 10 ** (expected[:, 0].astype(np.float64) / 20) * np.exp(
                    1j * expected[:, 1].astype(np.float64)
                )
                actual_spectrum = 10 ** (actual[:, 0].astype(np.float64) / 20) * np.exp(
                    1j * actual[:, 1].astype(np.float64)
                )
                waveform = np.fft.irfft(expected_spectrum, n=fft_size)
                difference = np.fft.irfft(actual_spectrum, n=fft_size) - waveform
                snr = 20 * np.log10(
                    max(np.linalg.norm(waveform), 1e-30) / max(np.linalg.norm(difference), 1e-30)
                )
                rows.append(
                    {
                        "case": name,
                        "metrics": metrics,
                        "policy_passed": outcome,
                        "inverse_fft_peak_error": float(np.abs(difference).max()),
                        "inverse_fft_snr_db": float(snr),
                        "torch_fp32_vs_fp64_max_magnitude_db": float(
                            np.abs(expected[:, 0] - reference64[:, 0]).max()
                        ),
                        "ort_vs_fp64_max_magnitude_db": float(
                            np.abs(actual[:, 0] - reference64[:, 0]).max()
                        ),
                    }
                )
            levels[level] = rows
    report = {
        "audit_kind": "diagnostic-numerical-parity",
        "checkpoint_sha256": checkpoint_hash,
        "artifact_sha256": artifact_hash,
        "torch": torch.__version__,
        "ort": ort.__version__,
        "architecture": platform.machine(),
        "system": platform.system(),
        "fft_size": fft_size,
        "hop_size": hop_size,
        "policy": PARITY_POLICY,
        "policy_sha256": POLICY_SHA256,
        "waveform_scope": "Normalized real inverse FFT of one windowed spectrum; before crossover, OLA, loudness and limiting. Not a deployed-quality or device test.",
        "levels": levels,
    }
    write_json_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.output)
    print(f"Numerical audit: {args.output}; levels={len(report['levels'])}")


if __name__ == "__main__":
    main()
