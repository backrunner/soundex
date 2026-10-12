#!/usr/bin/env python3
"""Fresh NumPy/ORT process peak RSS for diagnostic amplitude models, without torch."""

import argparse
import hashlib
import json
import resource
import sys

import numpy as np
import onnxruntime as ort


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def measure(path):
    baseline = peak_mib()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    if session.get_modelmeta().custom_metadata_map.get("soundex.production_compatible") != "false":
        raise ValueError("expected a diagnostic amplitude probe")
    loaded = peak_mib()
    state_size = next(value.shape[-1] for value in session.get_inputs() if value.name == "state")
    for channels in (1, 2):
        inputs = {
            "magnitude": np.ones((channels, 1, 129), np.float32),
            "context": np.zeros((channels, 1, 1538), np.float32),
            "condition": np.tile(np.array([[[1.0, 0.5, 1.0]]], np.float32), (channels, 1, 1)),
            "state": np.zeros((channels, state_size), np.float32),
        }
        for _ in range(200):
            amplitude, inputs["state"] = session.run(None, inputs)
            if not np.isfinite(amplitude).all() or not np.isfinite(inputs["state"]).all():
                raise ValueError("nonfinite inference")
    if "torch" in sys.modules:
        raise ValueError("memory harness unexpectedly imported torch")
    with open(path, "rb") as model_file:
        checksum = hashlib.file_digest(model_file, "sha256").hexdigest()
    return {
        "onnx_sha256": checksum,
        "baseline_import_peak_mib": baseline,
        "after_load_peak_mib": loaded,
        "after_inference_peak_mib": peak_mib(),
        "peak_increment_mib": peak_mib() - baseline,
        "scope": "fresh NumPy/ORT process peak RSS; excludes PyTorch, FFT frontend, detector, DSP, adapter and device; not deployment memory",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.model), allow_nan=False))
