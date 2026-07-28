"""Numerical tests for missing-band-aware deployment losses."""

import math

import pytest
import torch

from models.losses import (
    GeneratorLoss,
    blend_deployment_features,
    build_missing_band_mask,
    causal_overlap_add,
)


def _features(batch: int = 1, frames: int = 1, bins: int = 5) -> torch.Tensor:
    value = torch.zeros((batch, 2, frames, bins), dtype=torch.float32)
    value[:, 0].fill_(-20.0)
    return value


def _loss(fft_size: int = 8, hop_size: int = 2) -> GeneratorLoss:
    return GeneratorLoss(
        waveform_weight=0.0,
        fft_size=fft_size,
        hop_size=hop_size,
    )


def test_missing_band_mask_matches_raised_cosine_crossover() -> None:
    mask = build_missing_band_mask(
        torch.tensor([2.0]),
        torch.tensor([8.0]),
        fft_size=8,
        crossover_width_hz=2.0,
    )

    with pytest.raises(ValueError, match="finite"):
        build_missing_band_mask(
            torch.tensor([2.0]),
            torch.tensor([float("nan")]),
            fft_size=8,
            crossover_width_hz=2.0,
        )
    with pytest.raises(ValueError, match="Nyquist"):
        build_missing_band_mask(
            torch.tensor([5.0]),
            torch.tensor([8.0]),
            fft_size=8,
            crossover_width_hz=2.0,
        )

    torch.testing.assert_close(
        mask.flatten(),
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
        rtol=0,
        atol=1e-7,
    )


def test_primary_loss_changes_only_for_masked_high_band_error() -> None:
    target = _features()
    degraded = target.clone()
    mask = torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 1.0]]]])
    low_error = target.clone()
    low_error[:, 0, :, 0] += 10.0
    high_error = target.clone()
    high_error[:, 0, :, 4] += 10.0

    low_result = _loss()(low_error, target, degraded, mask)
    high_result = _loss()(high_error, target, degraded, mask)

    assert low_result["high_band_magnitude"].item() == 0.0
    assert high_result["high_band_magnitude"].item() == pytest.approx(5.0)
    assert low_result["low_band_identity"].item() > 0.0


def test_zero_energy_phase_is_ignored_but_meaningful_phase_is_weighted() -> None:
    target = _features()
    target[:, 0].fill_(-200.0)
    degraded = target.clone()
    predicted = target.clone()
    predicted[:, 1].fill_(math.pi)
    mask = torch.ones((1, 1, 1, 5))

    silent = _loss()(predicted, target, degraded, mask)
    target[:, 0, :, 2] = 0.0
    energetic = _loss()(predicted, target, degraded, mask)

    assert silent["phase"].item() == 0.0
    assert energetic["phase"].item() == pytest.approx(2.0, abs=1e-6)


def test_low_band_identity_and_crossover_are_separately_observable() -> None:
    target = _features(bins=3)
    degraded = target.clone()
    mask = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    constant = target.clone()
    constant[:, 0] += 2.0

    constant_result = _loss(fft_size=4, hop_size=1)(constant, target, degraded, mask)
    peaked = target.clone()
    peaked[:, 0, :, 1] += 2.0
    peaked_result = _loss(fft_size=4, hop_size=1)(peaked, target, degraded, mask)

    assert constant_result["low_band_identity"].item() == pytest.approx(2.0)
    assert constant_result["crossover_continuity"].item() == 0.0
    assert peaked_result["crossover_continuity"].item() == pytest.approx(2.0)


def test_causal_overlap_add_is_exact_for_identical_features() -> None:
    features = _features(frames=4)
    features[:, 1, :, 1] = 0.25

    first = causal_overlap_add(features, fft_size=8, hop_size=2)
    second = causal_overlap_add(features.clone(), fft_size=8, hop_size=2)

    assert first.shape == (1, 14)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()


def test_deployment_phase_blend_matches_rust_unit_phasors_and_fallback() -> None:
    degraded = _features(bins=4)
    predicted = degraded.clone()
    degraded[:, 1] = torch.tensor([0.125, 0.0, 0.0, -0.75])
    predicted[:, 1] = torch.tensor([2.75, 2.5, math.pi, 1.25])
    mask = torch.tensor([[[[0.0, 0.25, 0.5, 1.0]]]])

    blended = blend_deployment_features(degraded, predicted, mask)

    expected_intermediate = math.atan2(
        0.75 * math.sin(0.0) + 0.25 * math.sin(2.5),
        0.75 * math.cos(0.0) + 0.25 * math.cos(2.5),
    )
    assert blended[0, 1, 0, 0].item() == degraded[0, 1, 0, 0].item()
    assert blended[0, 1, 0, 1].item() == pytest.approx(expected_intermediate, abs=1e-7)
    assert blended[0, 1, 0, 2].item() == pytest.approx(-math.pi / 2.0, abs=1e-6)
    assert blended[0, 1, 0, 3].item() == predicted[0, 1, 0, 3].item()


def test_all_loss_terms_are_named_and_finite() -> None:
    target = _features(frames=2)
    result = GeneratorLoss(fft_size=8, hop_size=2)(
        target.clone(),
        target,
        target,
        torch.ones((1, 1, 1, 5)),
    )

    assert set(result) == {
        "total",
        "high_band_magnitude",
        "phase",
        "low_band_identity",
        "crossover_continuity",
        "waveform",
        "adversarial",
        "feature_matching",
    }
    assert all(torch.isfinite(value) for value in result.values())
