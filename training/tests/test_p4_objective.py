"""Reference semantics and numerical checks for the registered P4 objective."""

import numpy as np
import pytest
import torch

from models.balanced_objective import ARMS, balanced_band_energy, objective
from tests.test_causal_objective import make_batch


@pytest.mark.parametrize("rate", [44100, 48000])
def test_band_power_deficit_excess_identity_and_gradcheck(rate):
    torch.set_num_threads(1)
    size = 256
    n = torch.arange(size, dtype=torch.float64)
    spec = torch.fft.rfft(
        0.1 * torch.sin(2 * torch.pi * 80 * n / size) * torch.hann_window(size, dtype=torch.float64)
    )[None, None]
    rates = torch.tensor([[rate]], dtype=torch.float64)
    # One interior 2 kHz band, with power amplitude^2/2 and reference floor 1e-6.
    for multiplier in [0.0, 0.5, 2.0]:
        actual = balanced_band_energy(multiplier * spec, spec, rates, size)
        expected = abs(np.log1p(multiplier**2 / 1e-6) - np.log1p(1 / 1e-6)) / 7
        assert float(actual) == pytest.approx(expected, rel=1e-9, abs=1e-10)
    for target in [spec, torch.zeros_like(spec)]:
        prediction = target.clone().requires_grad_()
        loss = balanced_band_energy(prediction, target, rates, size)
        loss.backward()
        assert loss == 0 and bool((prediction.grad == 0).all())
    # Dense positive reference avoids abs kinks at empty bands during differences.
    target = torch.full((1, 1, 129), 0.03 + 0.02j, dtype=torch.complex128)
    prediction = (target * 1.4).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda p: balanced_band_energy(p, target, rates, size),
        (prediction,),
        eps=1e-7,
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize("rate", [44100, 48000])
def test_true_restoration_beats_missing_and_extra_energy(tmp_path, rate):
    n = np.arange(32768)
    dry = (0.2 * np.sin(n * 2 * np.pi * 1700 / rate)).astype(np.float32)
    clean = dry + (0.02 * np.sin(n * 2 * np.pi * 15000 / rate)).astype(np.float32)
    batch, target = make_batch(dry, rate, tmp_path), make_batch(clean, rate, tmp_path)
    batch["y"], batch["clean_wave"] = target["y"], target["clean_wave"]
    for arm in ARMS:
        oracle = objective(batch["y"], batch, arm=arm)["total"]
        assert oracle < objective(batch["x"], batch, arm=arm)["total"]
        noisy = batch["y"].clone()
        noisy[:, 0, :, 100:] += 20
        noisy.requires_grad_()
        result = objective(noisy, batch, arm=arm)
        assert result["total"] > oracle
        result["total"].backward()
        assert bool(torch.isfinite(noisy.grad).all())


def test_reference_peak_is_allowed_but_additional_overshoot_is_not(tmp_path):
    n = np.arange(32768)
    wave = (0.94 * np.sin(n * 2 * np.pi * 15000 / 48000)).astype(np.float32)
    batch = make_batch(wave, 48000, tmp_path)
    # Isolate peak semantics with fully active blending of the known HF reference.
    batch["mask"].fill_(1)
    batch["active"].fill_(True)
    old = objective(batch["y"], batch, arm="smooth_control")
    new = objective(batch["y"], batch, arm="reference_peak")
    assert old["prelimit_excess"] > 1e-4
    assert new["prelimit_excess"] < 1e-10
    louder = batch["y"].clone()
    louder[:, 0] += 3
    assert objective(louder, batch)["prelimit_excess"] > 0.01
    # Clean target is NOT limited; DSP distortion remains visible in waveform loss.
    assert batch["clean_wave"].abs().max() > 0.93
    assert new["output_waveform"] > 0


@pytest.mark.parametrize("arm", ARMS)
def test_silence_backward_is_finite(tmp_path, arm):
    batch = make_batch(np.zeros(32768, np.float32), 48000, tmp_path)
    prediction = batch["x"].clone().requires_grad_()
    result = objective(prediction, batch, arm=arm)
    result["total"].backward()
    assert all(bool(torch.isfinite(v)) for v in result.values())
    assert bool(torch.isfinite(prediction.grad).all())


def test_common_validation_uses_same_objective_for_every_group(monkeypatch):
    from scripts import run_weak_band_study as runner

    calls = []
    sentinel = object()

    def validate(model, kind, clips, specs, device, *, objective_fn):
        assert objective_fn is sentinel
        calls.append(specs)
        return {"score": float(len(calls))}

    monkeypatch.setattr(runner, "validation", validate)
    result = runner.common_validation(
        None,
        "tcn",
        None,
        [{"row_id": "dev"}],
        "cpu",
        train_specs=[{"row_id": "train"}],
        objective_fn=sentinel,
        version="p4",
    )
    assert result["score"] == 0.875 * 1 + 0.125 * 2
    assert result["training_probe"]["score"] == 3
    assert result["selection_objective"] == "p4"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_loss_and_gradient_match_cpu(tmp_path):
    n = np.arange(32768)
    wave = (
        0.3 * np.sin(n * 2 * np.pi * 1700 / 48000) + 0.02 * np.sin(n * 2 * np.pi * 15000 / 48000)
    ).astype(np.float32)
    batch = make_batch(wave, 48000, tmp_path)
    batch["mask"].fill_(1)
    batch["active"].fill_(True)
    outputs = []
    for device in ["cpu", "mps"]:
        local = {k: v.to(device) for k, v in batch.items()}
        prediction = local["x"].clone()
        # Keep the gradient comparison away from L1 zero-error kinks: a pure
        # bin-centred stationary tone has exactly zero samples/temporal deltas,
        # whose subgradients can differ after CPU/MPS FFT rounding.
        perturbation = torch.randn(prediction.shape, generator=torch.Generator().manual_seed(18))
        prediction += perturbation.to(device) * 0.02
        prediction[:, 0, :, 60:] += 2
        prediction.requires_grad_()
        result = objective(prediction, local)
        result["total"].backward()
        outputs.append((result["total"].detach().cpu(), prediction.grad.cpu()))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=2e-4, atol=1e-5)
    relative_gradient_error = (outputs[0][1] - outputs[1][1]).norm() / outputs[0][1].norm()
    assert relative_gradient_error < 2e-3
