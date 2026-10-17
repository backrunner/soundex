"""Separate frozen training-frontend error from synthesis parity on Rust observations."""

import json
import subprocess

import numpy as np
import torch

from models.causal_objective import render

TOLERANCE = 2e-5


def parity_report(
    audio,
    proxy,
    region,
    *,
    observed_binary=None,
    input_path=None,
    trace_path=None,
    prediction=None,
    batch=None,
):
    if (
        audio.shape != proxy.shape
        or audio.ndim != 1
        or not np.isfinite(audio).all()
        or not np.isfinite(proxy).all()
    ):
        raise ValueError("invalid aligned replay PCM")
    original = float(np.max(np.abs(audio[region] - proxy[region])))
    report = {
        "proxy_max_error": original,
        "training_frontend_parity_passed": bool(np.isfinite(original) and original <= TOLERANCE),
    }
    if observed_binary is None:
        if not report["training_frontend_parity_passed"]:
            raise ValueError(f"native parity failure: {original}")
        return report
    subprocess.run(
        [str(observed_binary), str(input_path), str(trace_path)], check=True, capture_output=True
    )
    trace = json.loads(trace_path.read_text())
    if (
        trace["schema_version"] != 1
        or trace["fft_size"] != 256
        or trace["hop_size"] != 128
        or trace["channels"] != 1
        or trace["sample_rate"] != int(batch["rate"][0, 0])
        or trace["layout"] != "hop,channel,magnitude_db_then_phase,bin"
    ):
        raise ValueError("unexpected runtime trace contract")
    observed = (
        torch.tensor(trace["features"], dtype=prediction.dtype)
        .reshape(-1, 1, 2, 129)
        .permute(1, 2, 0, 3)
    )
    if observed.shape != prediction.shape or not torch.isfinite(observed).all():
        raise ValueError("invalid runtime observed features")
    with torch.no_grad():
        corrected = render(
            prediction,
            observed,
            batch["mask"],
            batch["active"],
            batch["gate"],
            batch["dry"],
            limiter_mode="smooth-knee",
        )[0][0].numpy()
    error = float(np.max(np.abs(audio[region] - corrected[region])))
    report.update(
        runtime_observed_proxy_max_error=error,
        runtime_observed_parity_passed=bool(np.isfinite(error) and error <= TOLERANCE),
        scope="prediction unchanged; Rust observed spectrum used ONLY for synthesis diagnostic; original training/frontend discrepancy retained",
    )
    if not report["runtime_observed_parity_passed"]:
        raise ValueError(
            f"runtime-observed synthesis parity failure: {error}; training frontend: {original}"
        )
    return report
