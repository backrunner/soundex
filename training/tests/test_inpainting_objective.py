"""Inpainting must recover the reference rather than reward hiss or smoothing."""

import pytest
import torch

from models.losses import GeneratorLoss


def features() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(2, 2, 8, 129)
    target[:, 0] = -60.0
    target[:, 0, 2:5, 80:120] = -20.0
    target[:, 1] = torch.arange(129) * 0.02
    mask = torch.zeros(2, 1, 1, 129)
    mask[..., 70:] = 1.0
    return target, mask


def objective() -> GeneratorLoss:
    return GeneratorLoss(
        objective_version=3,
        fft_size=256,
        hop_size=128,
        waveform_region="steady_state",
        waveform_weight=0.25,
    )


def test_perfect_inpainting_is_zero_and_has_finite_backward() -> None:
    target, mask = features()
    degraded = target.clone()
    degraded[:, 0, :, 70:] = -100.0
    predicted = target.clone().requires_grad_()
    terms = objective()(predicted, target, degraded, mask)
    assert terms["total"].item() == pytest.approx(0.0, abs=1e-6)
    terms["total"].backward()
    assert torch.isfinite(predicted.grad).all()


def test_dry_missing_band_and_energy_overshoot_both_receive_gradients() -> None:
    target, mask = features()
    for shift in [-30.0, 30.0]:
        predicted = target.clone()
        predicted[:, 0, :, 70:] += shift
        predicted.requires_grad_()
        terms = objective()(predicted, target, target, mask)
        assert terms["high_band_spectral_convergence"].item() > 0
        assert terms["high_band_energy"].item() > 0
        terms["total"].backward()
        assert torch.isfinite(predicted.grad).all()
        assert predicted.grad[:, 0, :, 70:].abs().sum() > 0


def test_temporal_target_preserves_attack_instead_of_rewarding_flat_output() -> None:
    target, mask = features()
    flattened = target.clone()
    flattened[:, 0] = target[:, 0].mean(dim=1, keepdim=True)
    correct = objective()(target, target, target, mask)
    flattened_loss = objective()(flattened, target, target, mask)
    assert correct["high_band_temporal"].item() == 0.0
    assert flattened_loss["high_band_temporal"].item() > 0.5


def test_weak_bin_floor_cannot_make_hiss_a_success() -> None:
    target, mask = features()
    target[:, 0] = -200.0
    noise = target.clone()
    noise[:, 0, :, 70:] = -80.0
    terms = objective()(noise, target, target, mask)
    # Log magnitude is censored below the reference floor; linear losses
    # explicitly penalize filling silent high bins with noise.
    assert terms["high_band_spectral_convergence"].item() > 0
    assert terms["high_band_energy"].item() > 0
    assert terms["total"].item() > 0
    prediction = target.clone().requires_grad_()
    silent = objective()(prediction, target, target, mask)
    assert silent["total"].item() == 0.0
    silent["total"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_high_linear_terms_are_gain_invariant_above_absolute_floor() -> None:
    target, mask = features()
    target[:, 0] += 40.0  # Every frame remains above the absolute energy floor.
    predicted = target.clone()
    predicted[:, 0, :, 70:] += 3.0
    expected = objective()(predicted, target, target, mask)
    for gain_db in [-6.0, 6.0]:
        actual_target = target.clone()
        actual_predicted = predicted.clone()
        actual_target[:, 0] += gain_db
        actual_predicted[:, 0] += gain_db
        actual = objective()(actual_predicted, actual_target, actual_target, mask)
        for key in ("high_band_spectral_convergence", "high_band_energy", "high_band_temporal"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)


def test_ola_phase_artifacts_are_seen_by_reconstruction_high_term() -> None:
    target, mask = features()
    wrong_phase = target.clone()
    wrong_phase[:, 1, ::2, 70:] += torch.pi
    terms = objective()(wrong_phase, target, target, mask)
    assert terms["high_band_spectral_convergence"].item() == 0.0
    assert terms["reconstruction_high_spectral_convergence"].item() > 0.0
