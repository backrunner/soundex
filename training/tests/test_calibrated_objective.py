"""Reconstruction calibration and circular phase differences must follow the reference."""

import pytest
import torch

from models.losses import GeneratorLoss, phase_gradient_terms, reconstruction_energy_terms
from train import spectral_features


def example():
    torch.manual_seed(23)
    target = spectral_features(torch.randn(2, 1152) * 0.1, 256, 128)
    mask = torch.zeros(2, 8, 129)
    mask[..., 70:] = 1
    return target, mask


def energy(predicted, target, mask):
    return reconstruction_energy_terms(
        predicted,
        target,
        target,
        mask,
        fft_size=256,
        hop_size=128,
        relative_floor=0.01,
        absolute_floor=1e-4,
    )


def test_energy_detects_phase_cancellation_after_ola_with_unchanged_raw_magnitudes():
    target, mask = example()
    prediction = target.clone()
    prediction[:, 1, ::2, 70:] += torch.pi
    prediction.requires_grad_()
    result = energy(prediction, target, mask)
    assert all(v > 0.1 for v in result.values())
    sum(result.values()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1].abs().sum() > 0
    assert all(v < 1e-5 for v in energy(target, target, mask).values())


def test_phase_gradients_are_circular_and_penalize_removed_reference_variations():
    target, mask = example()
    reference = target[:, 1]
    exact = phase_gradient_terms(reference + 2 * torch.pi, reference, mask)
    assert all(v < 1e-6 for v in exact.values())
    smoothed = reference[:, :1].expand_as(reference).clone().requires_grad_()
    result = phase_gradient_terms(smoothed, reference, mask)
    assert result["phase_time_gradient"] > 0.1
    sum(result.values()).backward()
    assert torch.isfinite(smoothed.grad).all()


@pytest.mark.parametrize("empty", [False, True])
def test_v5_silence_or_empty_band_has_finite_loss_and_backward(empty):
    target, mask = example()
    if empty:
        mask.zero_()
    else:
        target[:, 0] = -200
    prediction = target.clone().requires_grad_()
    loss = GeneratorLoss(
        objective_version=5, fft_size=256, hop_size=128, waveform_region="steady_state"
    )
    result = loss(prediction, target, target, mask)
    assert all(torch.isfinite(v) for v in result.values())
    assert result["total"] < 1e-5
    result["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="non-negative"):
        GeneratorLoss(objective_version=5, reconstruction_energy_weight=-1)
