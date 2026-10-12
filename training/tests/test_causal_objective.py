"""P2 renderer matches production DSP, and the new objective has finite gradients."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from evaluation.rust_bridge import read_sxa, write_sxa
from models.causal_objective import BURN, gate_envelope, objective, overlap_add, render
from models.losses import build_missing_band_mask, causal_overlap_add
from train import spectral_features

ROOT = Path(__file__).resolve().parents[2]


def make_batch(audio, rate, root):
    write_sxa(root / "input.sxa", audio, rate)
    detector = ROOT / "target/debug/soundex-gain-data"
    if not detector.exists():
        pytest.fail("prebuild soundex-gain-data and soundex-prediction-replay")
    subprocess.run(
        [
            str(detector),
            str(root / "input.sxa"),
            str(root / "input.sxa"),
            str(root / "trace.json"),
            "1",
        ],
        check=True,
        capture_output=True,
    )
    trace = json.loads((root / "trace.json").read_text())[: len(audio) // 128]
    cutoff = torch.tensor([[r["cutoff_hz"] for r in trace]])
    active = torch.tensor([[r["active"] for r in trace]])
    active[:, :BURN] = False
    features = spectral_features(F.pad(torch.from_numpy(audio)[None], (128, 0)), 256, 128)
    mask = build_missing_band_mask(
        cutoff.flatten(),
        torch.full_like(cutoff.flatten(), rate),
        fft_size=256,
        crossover_width_hz=1000,
    ).reshape(1, 1, len(trace), 129)
    return {
        "x": features,
        "y": features,
        "mask": mask,
        "active": active,
        "gate": gate_envelope(active),
        "rate": torch.tensor([[rate]], dtype=torch.float32),
        "cutoff": cutoff,
        "dry": F.pad(torch.from_numpy(audio)[None], (128, 0))[:, : len(audio)],
        "clean_wave": F.pad(torch.from_numpy(audio)[None], (128, 0))[:, : len(audio)],
    }


@pytest.mark.parametrize("limiter_mode", ["legacy", "smooth-knee"])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_renderer_matches_rust_gate_limiter_and_silence(tmp_path, rate, limiter_mode):
    torch.set_num_threads(1)
    n = np.arange(256 * 128)
    audio = (1.15 * np.sin(n * 2 * np.pi * 1700 / rate)).astype(np.float32)
    audio[128 * 90 : 128 * 120] = 0
    audio[128 * 200 :] = 0
    batch = make_batch(audio, rate, tmp_path)
    predicted = batch["x"].clone()
    predicted[:, 0, :, 50:] = -12
    predicted[:, 1, :, 50:] = 0.8
    pred_path = tmp_path / "features.json"
    pred_path.write_text(json.dumps(predicted.permute(2, 0, 1, 3).reshape(-1).tolist()))
    subprocess.run(
        [
            str(ROOT / "target/debug/soundex-prediction-replay"),
            str(ROOT / "tests/fixtures/stateful-probe.onnx"),
            str(tmp_path / "input.sxa"),
            str(pred_path),
            str(tmp_path / "output.sxa"),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "SOUNDEX_EVAL_LIMITER_MODE": limiter_mode},
    )
    actual, _ = read_sxa(tmp_path / "output.sxa")
    expected, _ = render(
        predicted,
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode=limiter_mode,
    )
    # Float32 analyzer/window differences can rotate almost-zero observed bins.
    # This adversarial pure-tone test then gives them nonzero generated energy.
    # Bound waveform peak AND RMS; do not compare undefined weak-bin phase.
    error = actual[:, 0] - expected[0].numpy()
    assert float(np.sqrt(np.mean(error**2))) < 2e-5
    np.testing.assert_allclose(actual[:, 0], expected[0].numpy(), atol=2e-4, rtol=2e-5)
    folded = overlap_add(batch["x"]).numpy()
    sequential = causal_overlap_add(batch["x"], fft_size=256, hop_size=128).numpy()
    # The two complex constructors can differ by an FFT ULP. Dividing by a
    # single Hann tail amplifies it at artificial crop edges (Linux sample 3).
    # Deployment/loss use full overlap support. Keep its strict comparison and
    # check the unnormalized signal at EVERY sample, including both tails.
    np.testing.assert_allclose(folded[:, 128:-128], sequential[:, 128:-128], atol=2e-5, rtol=2e-5)
    weights = np.zeros(folded.shape[-1])
    window = np.hanning(257)[:-1]
    for start in range(0, batch["x"].shape[2] * 128, 128):
        weights[start : start + 256] += window**2
    assert np.isfinite(folded).all() and np.isfinite(sequential).all()
    np.testing.assert_allclose(folded * weights, sequential * weights, atol=2e-6, rtol=2e-6)
    assert float(expected.abs().max()) <= 0.95
    assert bool((expected[:, 128 * 210 :] == 0).all())


def test_objective_identity_and_silence_have_finite_backward(tmp_path):
    for audio in [
        np.zeros(256 * 128, np.float32),
        np.random.default_rng(42).normal(0, 0.1, 256 * 128).astype(np.float32),
    ]:
        batch = make_batch(audio, 48000, tmp_path)
        prediction = batch["x"].clone().requires_grad_()
        values = objective(prediction, batch)
        values["total"].backward()
        assert all(bool(torch.isfinite(v)) for v in values.values())
        assert bool(torch.isfinite(prediction.grad).all())
        assert float(values["output_waveform"].detach()) < 1e-4


def test_smooth_limiter_gradient_and_invalid_mode():
    from models.causal_objective import limit

    wave = torch.tensor(
        [-1.1, -0.9, -0.855, 0.0, 0.855, 0.9, 1.1], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda x: limit(x, "smooth-knee"), (wave,))
    limit(wave, "smooth-knee").sum().backward()
    assert torch.isfinite(wave.grad).all()
    torch.testing.assert_close(wave.grad[[2, 3, 4]], torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError):
        limit(wave, "typo")
