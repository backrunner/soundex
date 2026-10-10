"""Deployment gain recurrence and reconstructed transients are supervised explicitly."""

import pytest
import torch

from models.losses import (
    GeneratorLoss,
    build_missing_band_mask,
    match_edge_features,
    reconstruction_energy_terms,
)
from train import spectral_features


def test_match_edge_rounding_gain_recurrence_and_retained_bins():
    # Cutoff 3.5 bins rounds to four as in Rust; edge width 1.5 rounds to two.
    degraded = torch.full((1, 2, 3, 9), -20.0)
    predicted = degraded.clone()
    degraded[:, 0, :, 2:4] = 0.0
    predicted[:, 0, 1, 4:] = 20.0
    predicted.requires_grad_()
    mask = build_missing_band_mask(
        torch.tensor([3500.0]), torch.tensor([16000.0]), fft_size=16, crossover_width_hz=1500.0
    )
    actual, gains = match_edge_features(
        degraded,
        predicted,
        mask,
        cutoff_hz=torch.tensor([3500.0]),
        sample_rate=torch.tensor([16000.0]),
        crossover_width_hz=1500.0,
        initial_gain=torch.ones(1),
    )
    # Desired ratios clamp to [2, .25, 2]; exact Rust .1/.9 recurrence.
    torch.testing.assert_close(gains, torch.tensor([[1.1, 1.015, 1.1135]]))
    torch.testing.assert_close(actual[:, 0, :, :3], predicted[:, 0, :, :3])
    torch.testing.assert_close(actual[:, 1], predicted[:, 1])
    actual.sum().backward()
    assert torch.isfinite(predicted.grad).all()


def test_steady_bootstrap_is_causal_and_differentiable():
    torch.manual_seed(12)
    x = spectral_features(torch.randn(2, 1152) * 0.1, 256, 128)
    geometry = dict(
        cutoff_hz=torch.tensor([12000.0, 15000.0]),
        sample_rate=torch.tensor([44100.0, 48000.0]),
        crossover_width_hz=1000.0,
    )
    mask = build_missing_band_mask(
        geometry["cutoff_hz"], geometry["sample_rate"], fft_size=256, crossover_width_hz=1000.0
    )
    pred = x.clone().requires_grad_()
    out, gain = match_edge_features(x, pred, mask, **geometry)
    changed = pred.detach().clone()
    changed[:, 0, 4:] += 10
    prefix, _ = match_edge_features(x, changed, mask, **geometry)
    torch.testing.assert_close(out[:, :, :4], prefix[:, :, :4])
    assert ((gain >= 0.25) & (gain <= 2)).all()
    out.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_v6_requires_geometry_and_preserves_v5_raw_terms():
    torch.manual_seed(4)
    x = spectral_features(torch.randn(2, 1152) * 0.1, 256, 128)
    geometry = dict(
        cutoff_hz=torch.tensor([12000.0, 15000.0]),
        sample_rate=torch.tensor([44100.0, 48000.0]),
        crossover_width_hz=1000.0,
    )
    mask = build_missing_band_mask(
        geometry["cutoff_hz"], geometry["sample_rate"], fft_size=256, crossover_width_hz=1000.0
    )
    pred = x.clone()
    pred[:, 0] += 1
    pred.requires_grad_()
    loss = GeneratorLoss(
        objective_version=6, fft_size=256, hop_size=128, waveform_region="steady_state"
    )
    with pytest.raises(ValueError, match="geometry"):
        loss(pred, x, x, mask)
    result = loss(pred, x, x, mask, deployment_geometry=geometry)
    old = GeneratorLoss(
        objective_version=5, fft_size=256, hop_size=128, waveform_region="steady_state"
    )(pred, x, x, mask)
    for key in ("high_band_energy", "high_band_magnitude", "high_band_temporal", "phase"):
        torch.testing.assert_close(result[key], old[key])
    assert result["reconstruction_temporal"] > 0
    assert result["waveform"] != old["waveform"]
    result["total"].backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("empty", [False, True])
def test_reconstructed_transients_follow_true_attacks_and_decays(empty):
    torch.manual_seed(1)
    wave = torch.randn(1, 1152) * torch.linspace(0.01, 0.5, 1152)
    target = spectral_features(wave, 256, 128)
    mask = torch.ones(1, 8, 129) * (0 if empty else 1)

    def measure(pred):
        return reconstruction_energy_terms(
            pred,
            target,
            target,
            mask,
            fft_size=256,
            hop_size=128,
            relative_floor=0.01,
            absolute_floor=1e-4,
            include_temporal=True,
        )["reconstruction_temporal"]

    assert measure(target) < 1e-5
    flat = target[:, :, :1].expand_as(target).clone().requires_grad_()
    error = measure(flat)
    assert error == 0 if empty else error > 0.1
    error.backward()
    assert torch.isfinite(flat.grad).all()
