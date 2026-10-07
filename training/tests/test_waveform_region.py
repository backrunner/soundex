"""Cropped Hann boundaries must not dominate the continuous-stream training loss."""

from pathlib import Path

import pytest
import torch

from configuration import load_config
from models.losses import blend_deployment_features, causal_overlap_add, deployment_waveform_loss
from train import spectral_features


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.arange(640) / 44100.0
    target = spectral_features((0.2 * torch.sin(2 * torch.pi * 997 * time))[None], 256, 128)
    predicted = target.clone()
    predicted[:, 1] += 0.4
    predicted[:, 0, :, 90] = 0.0
    mask = torch.ones((1, 1, 1, 129))
    return predicted, target, mask


def test_steady_state_loss_excludes_only_incomplete_overlap_support() -> None:
    predicted, target, mask = _case()
    actual = deployment_waveform_loss(
        predicted, target, target, mask, fft_size=256, hop_size=128, waveform_region="steady_state"
    )
    blended = blend_deployment_features(target, predicted, mask)
    error = (
        causal_overlap_add(blended, fft_size=256, hop_size=128)
        - causal_overlap_add(target, fft_size=256, hop_size=128)
    ).abs()
    torch.testing.assert_close(actual, error[:, 128:-128].mean())
    full = deployment_waveform_loss(predicted, target, target, mask, fft_size=256, hop_size=128)
    assert full > actual * 2


def test_steady_state_loss_has_finite_nonzero_phase_gradients() -> None:
    predicted, target, mask = _case()
    predicted.requires_grad_()
    loss = deployment_waveform_loss(
        predicted, target, target, mask, fft_size=256, hop_size=128, waveform_region="steady_state"
    )
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, 1].abs().sum() > 0


def test_single_frame_cannot_supply_a_steady_state_region() -> None:
    predicted, target, mask = _case()
    with pytest.raises(ValueError, match="more causal sequence frames"):
        deployment_waveform_loss(
            predicted[:, :, :1],
            target[:, :, :1],
            target[:, :, :1],
            mask,
            fft_size=256,
            hop_size=128,
            waveform_region="steady_state",
        )


def test_default_profile_binds_low_latency_and_corrected_loss() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/default.yaml")
    assert config["audio"]["fft_size"] == 256
    assert config["audio"]["hop_size"] == 128
    assert config["training"]["objective"]["waveform_region"] == "steady_state"
