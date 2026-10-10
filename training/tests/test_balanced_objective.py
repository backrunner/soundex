"""Loss normalization must respect source draw mass, silence and phase geometry."""

import math

import pytest
import torch

from models.losses import GeneratorLoss, _blend_phase_like_runtime, deployment_waveform_loss
from train import spectral_features


def _pair(gain: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.arange(768) / 44100.0
    clean = gain * (0.15 * torch.sin(2 * torch.pi * 997 * time))
    degraded = clean * 0.8
    target = spectral_features(clean[None], 256, 128)
    source = spectral_features(degraded[None], 256, 128)
    return source, target, torch.ones((1, 1, 1, 129))


def test_antipodal_phase_fallback_has_finite_backward() -> None:
    generator = torch.Generator().manual_seed(1)
    original = (torch.rand(100_000, generator=generator) * 2 - 1) * torch.pi
    predicted = (original + torch.pi).requires_grad_()
    real = (original.cos() + predicted.detach().cos()) * 0.5
    imag = (original.sin() + predicted.detach().sin()) * 0.5
    assert ((real == 0) & (imag == 0)).any(), "exercise exact floating-point cancellation"
    output = _blend_phase_like_runtime(original, predicted, torch.full_like(original, 0.5))
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(predicted.grad).all()
    torch.testing.assert_close(predicted.grad, torch.full_like(original, 0.5), rtol=1e-5, atol=1e-6)


def test_v2_band_width_cannot_change_another_recordings_draw_mass() -> None:
    target = torch.zeros((2, 2, 2, 5))
    predicted = target.clone()
    predicted[0, 0] = 2.0
    predicted[1, 0] = 8.0
    mask = torch.tensor([[[[1, 0, 0, 0, 0]]], [[[1, 1, 1, 1, 1]]]], dtype=torch.float32)
    result = GeneratorLoss(objective_version=2, fft_size=8, hop_size=2)(
        predicted, target, target, mask
    )
    assert result["high_band_magnitude"].item() == pytest.approx(5.0)
    separate = [
        GeneratorLoss(objective_version=2, fft_size=8, hop_size=2)(
            predicted[i : i + 1], target[i : i + 1], target[i : i + 1], mask[i : i + 1]
        )["high_band_magnitude"]
        for i in range(2)
    ]
    torch.testing.assert_close(result["high_band_magnitude"], torch.stack(separate).mean())


def test_relative_waveform_defines_dry_as_one_and_reference_as_zero() -> None:
    degraded, target, mask = _pair()
    kwargs = dict(fft_size=256, hop_size=128, waveform_region="steady_state")
    dry = deployment_waveform_loss(
        degraded, target, degraded, mask, normalization="baseline_error", **kwargs
    )
    reference = deployment_waveform_loss(
        target, target, degraded, mask, normalization="baseline_error", **kwargs
    )
    assert dry.item() == pytest.approx(1.0, abs=1e-6)
    assert reference.item() == 0.0


def test_relative_waveform_is_gain_invariant_above_floors() -> None:
    values = []
    for gain in [0.5, 1.0, 2.0]:
        degraded, target, mask = _pair(gain)
        prediction = target.clone()
        prediction[:, 1] += 0.2
        values.append(
            deployment_waveform_loss(
                prediction,
                target,
                degraded,
                mask,
                fft_size=256,
                hop_size=128,
                waveform_region="steady_state",
                normalization="baseline_error",
            )
        )
    torch.testing.assert_close(torch.stack(values), values[1].expand(3), rtol=1e-5, atol=1e-6)


def test_silent_and_near_identical_sources_have_finite_bounded_gradients() -> None:
    target = torch.zeros((2, 2, 4, 129))
    target[:, 0] = -200
    predicted = target.clone().requires_grad_()
    result = GeneratorLoss(
        objective_version=2, fft_size=256, hop_size=128, waveform_region="steady_state"
    )(predicted, target, target, torch.ones((2, 1, 1, 129)))
    result["total"].backward()
    assert result["total"].item() == 0
    assert torch.isfinite(predicted.grad).all()
    _degraded, clean, mask = _pair()
    predicted = clean.clone().requires_grad_()
    loss = deployment_waveform_loss(
        predicted,
        clean,
        clean,
        mask,
        fft_size=256,
        hop_size=128,
        waveform_region="steady_state",
        normalization="baseline_error",
    )
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


def test_phase_loss_respects_wrap_and_has_a_useful_gradient() -> None:
    degraded, target, mask = _pair()
    prediction = target.clone()
    prediction[:, 1] += 2 * math.pi
    loss = GeneratorLoss(objective_version=2, fft_size=256, hop_size=128)
    assert loss(prediction, target, degraded, mask)["phase"].item() < 1e-6
    prediction = target.clone()
    prediction[:, 1] += 0.4
    prediction.requires_grad_()
    loss(prediction, target, degraded, mask)["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1].abs().sum() > 0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_normalization_scales_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GeneratorLoss(objective_version=2, magnitude_scale_db=value)


def test_legacy_profile_keeps_raw_objective_and_metric_keys() -> None:
    source, target, mask = _pair()
    old = GeneratorLoss(fft_size=256, hop_size=128)
    result = old(source, target, source, mask)
    expected = (
        result["high_band_magnitude"]
        + 0.2 * result["phase"]
        + 0.1 * result["low_band_identity"]
        + 0.05 * result["crossover_continuity"]
        + 0.1 * result["waveform"]
    )
    torch.testing.assert_close(result["total"], expected)
    assert "waveform_relative" not in result
