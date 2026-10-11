"""Long reconstruction supervision must preserve attacks and real DSP history."""

import pytest
import torch

from models.generator import SoundExGenerator
from models.losses import (
    GeneratorLoss,
    build_missing_band_mask,
    deployment_waveform_loss,
    match_edge_features,
    multiresolution_reconstruction_terms,
)
from train import infer_independent_frames, infer_loss_sequence, spectral_features


def example(frames=64):
    torch.manual_seed(17)
    wave = torch.randn(2, (frames + 1) * 128) * torch.linspace(0.01, 0.2, (frames + 1) * 128)
    x = spectral_features(wave, 256, 128)
    geometry = dict(
        cutoff_hz=torch.tensor([12000.0, 15000.0]),
        sample_rate=torch.tensor([44100.0, 48000.0]),
        crossover_width_hz=1000.0,
    )
    mask = build_missing_band_mask(
        geometry["cutoff_hz"], geometry["sample_rate"], fft_size=256, crossover_width_hz=1000.0
    )
    return x, geometry, mask


def measure(pred, target, geometry):
    return multiresolution_reconstruction_terms(
        pred,
        target,
        target,
        fft_size=256,
        hop_size=128,
        relative_floor=0.01,
        absolute_floor=1e-4,
        **geometry,
    )


def test_multires_reference_is_zero_and_rejects_flattened_attacks_and_excess_energy():
    target, geometry, _ = example(32)
    exact = measure(target, target, geometry)
    assert all(v < 1e-5 for v in exact.values())
    flat = target[:, :, :1].expand_as(target).clone().requires_grad_()
    errors = measure(flat, target, geometry)
    assert errors["multires_high_temporal"] > 0.1
    assert errors["multires_high_spectral"] > 0.1
    sum(errors.values()).backward()
    assert torch.isfinite(flat.grad).all() and flat.grad.abs().sum() > 0
    loud = target.clone()
    loud[:, 0, :, 90:] += 12
    assert measure(loud, target, geometry)["multires_high_energy"] > 1
    with pytest.raises(ValueError, match="longest-window"):
        measure(target[:, :, :8], target[:, :, :8], geometry)


def test_multires_silence_empty_band_and_floors_have_finite_gradients():
    target, geometry, _ = example(32)
    target[:, 0] = -200
    target[:, 1] = 0
    pred = target.clone().requires_grad_()
    result = measure(pred, target, geometry)
    sum(result.values()).backward()
    assert all(torch.isfinite(v) for v in result.values())
    assert torch.isfinite(pred.grad).all()
    target, geometry, _ = example(32)
    geometry["cutoff_hz"] = geometry["sample_rate"] / 2
    geometry["crossover_width_hz"] = 0.0
    result = measure(target + 1, target, geometry)
    assert all(v == 0 for k, v in result.items() if "high" in k)


def test_v7_gain_uses_real_prefix_but_never_supervises_prefix():
    x, geometry, mask = example()
    pred = x.clone()
    pred[:, 0, :32] += 10
    pred.requires_grad_()
    loss = GeneratorLoss(
        objective_version=7,
        reconstruction_warmup_frames=32,
        fft_size=256,
        hop_size=128,
        waveform_region="steady_state",
        adversarial_weight=0.0,
        feature_matching_weight=0.0,
    )
    result = loss(pred, x, x, mask, deployment_geometry=geometry)
    deployed, _ = match_edge_features(x, pred, mask, initial_gain=torch.ones(2), **geometry)
    expected = deployment_waveform_loss(
        deployed[:, :, 32:],
        x[:, :, 32:],
        x[:, :, 32:],
        mask,
        fft_size=256,
        hop_size=128,
        waveform_region="steady_state",
    )
    torch.testing.assert_close(result["waveform"], expected)
    assert result["high_band_magnitude"] == 0  # The changed prefix is excluded.
    alternative = loss(x, x, x, mask, deployment_geometry=geometry)
    assert result["waveform"] != alternative["waveform"]  # Gain state retains real history.
    result["total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[:, :, :32]) == 0
    assert pred.grad[:, :, 32:].abs().sum() > 0
    with pytest.raises(ValueError, match="warmup"):
        GeneratorLoss(objective_version=7)


def test_detached_dsp_history_matches_full_observed_inference():
    model = SoundExGenerator(
        channels=[4, 8, 16, 16],
        bottleneck_blocks=1,
        spectral_refiner=dict(bins=129, width=32, representation="gain_shape", history_frames=8),
    ).eval()
    x, _, _ = example(48)
    actual = infer_loss_sequence(model, x, warmup_frames=8, reconstruction_warmup_frames=8)
    expected = infer_independent_frames(model, x, warmup_frames=8)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    model.train()
    with pytest.raises(ValueError, match="frozen BN"):
        infer_loss_sequence(model, x, warmup_frames=8, reconstruction_warmup_frames=8)
