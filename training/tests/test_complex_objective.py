"""Complex consistency must detect phase corruption and low-band leakage."""

import pytest
import torch

from models.losses import GeneratorLoss, reconstruction_complex_terms


def valid_features():
    torch.manual_seed(13)
    wave = torch.randn(2, 1152) * 0.1
    spectrum = torch.fft.rfft(wave.unfold(-1, 256, 128) * torch.hann_window(256))
    features = torch.stack((20 * spectrum.abs().clamp_min(1e-10).log10(), spectrum.angle()), dim=1)
    mask = torch.zeros(2, 8, 129)
    mask[..., 70:] = 1
    return features, mask


def terms(predicted, target, degraded, mask):
    return reconstruction_complex_terms(
        predicted,
        target,
        degraded,
        mask,
        fft_size=256,
        hop_size=128,
        relative_floor=0.01,
        absolute_floor=1e-4,
    )


def test_valid_reference_has_zero_complex_error_and_finite_backward():
    target, mask = valid_features()
    prediction = target.clone().requires_grad_()
    result = terms(prediction, target, target, mask)
    assert all(value.item() < 1e-6 for value in result.values())
    sum(result.values()).backward()
    assert torch.isfinite(prediction.grad).all()


def test_phase_corruption_causes_high_error_inconsistency_and_low_leakage():
    target, mask = valid_features()
    prediction = target.clone()
    prediction[:, 1, ::2, 70:] += torch.pi
    prediction.requires_grad_()
    result = terms(prediction, target, target, mask)
    assert all(value.item() > 1e-3 for value in result.values())
    sum(result.values()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1].abs().sum() > 0


def test_silence_and_empty_missing_band_are_finite():
    target, mask = valid_features()
    target[:, 0] = -200
    prediction = target.clone().requires_grad_()
    result = terms(prediction, target, target, mask)
    sum(result.values()).backward()
    assert all(torch.isfinite(v) for v in result.values())
    assert torch.isfinite(prediction.grad).all()
    target, mask = valid_features()
    prediction = target.clone()
    prediction[:, 1] += 1
    result = terms(prediction, target, target, mask.zero_())
    assert result["reconstruction_high_complex"].item() == 0
    assert result["spectral_consistency"].item() == 0
    assert result["reconstruction_low_complex"].item() < 1e-6


def test_v4_does_not_reward_collapsing_missing_high_band():
    target, mask = valid_features()
    objective = GeneratorLoss(
        objective_version=4,
        fft_size=256,
        hop_size=128,
        waveform_region="steady_state",
        adversarial_weight=0,
        feature_matching_weight=0,
    )
    exact = objective(target, target, target, mask)
    collapsed = target.clone()
    collapsed[:, 0, :, 70:] = -120
    result = objective(collapsed, target, target, mask)
    assert exact["total"].item() < 1e-5
    assert result["total"].item() > 0.5
    with pytest.raises(ValueError, match="non-negative"):
        GeneratorLoss(objective_version=4, spectral_consistency_weight=-1)
