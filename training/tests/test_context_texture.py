import pytest
import torch

from models.context_texture import (
    ContextTextureCritic,
    observed_context,
    shifted_high,
    structured_loss,
)
from models.spectral_texture import SpectralTextureCritic


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.02])
def test_reference_stationarity_and_detached_paths(context, scale):
    torch.manual_seed(60)
    critic = ContextTextureCritic(context).double()
    target = (torch.randn(2, 1536, dtype=torch.float64) * scale).requires_grad_()
    observed = (target.detach() * 0.5).requires_grad_()
    predicted = target.detach().clone().requires_grad_()
    loss = critic.reference_loss(predicted, target, observed, torch.tensor([44100, 48000]))
    loss.sum().backward()
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))
    assert target.grad is None and observed.grad is None
    assert all(p.grad is None for p in critic.parameters())


def test_neutral_initialization_matches_original_and_equal_capacity():
    torch.manual_seed(60)
    original = SpectralTextureCritic()
    wave = torch.randn(2, 1536) * 0.02
    rates = torch.tensor([44100, 48000])
    counts = []
    for context in (False, True):
        torch.manual_seed(60)
        critic = ContextTextureCritic(context)
        counts.append(sum(p.numel() for p in critic.parameters()))
        for a, b in zip(
            original.features(wave, wave, rates), critic.features(wave, wave, rates), strict=True
        ):
            torch.testing.assert_close(a[1], b[1], atol=1e-7, rtol=1e-6)
    assert counts == [3970, 3970]


def test_context_is_physical_time_varying_and_reaches_high_band():
    size, length = 512, 2048
    rates = torch.tensor([48000])
    time = torch.arange(length, dtype=torch.float64) / 48000
    observed = (0.02 * torch.sin(2 * torch.pi * 1500 * time))[None]
    observed[:, length // 2 :] *= 0.1
    context = observed_context(observed, rates, size)
    assert context.shape == (1, 8, 1, 13)
    assert context[0, 0, 0, 1] > context[0, 0, 0, -2]
    assert context[0, 0].mean() > context[0, 5].mean() * 100
    torch.manual_seed(60)
    critic = ContextTextureCritic(True).double()
    # A constructive path proves low-band summaries can reach distant HF logits.
    with torch.no_grad():
        for layers in critic.blocks:
            layers[0].weight[:, 3, 1, 1] = 0.2
    a = critic.features(torch.zeros_like(observed), observed, rates)
    b = critic.features(torch.zeros_like(observed), observed.flip(-1), rates)
    assert (a[0][1][:, :, 180:] - b[0][1][:, :, 180:]).abs().max() > 1e-7


@pytest.mark.parametrize("structured", [False, True])
def test_negative_training_is_finite_detached_and_context_learns(structured):
    torch.manual_seed(60)
    critic = ContextTextureCritic(True)
    wave = (torch.randn(2, 1536) * 0.02).requires_grad_()
    rates = torch.tensor([44100, 48000])
    loss = structured_loss(
        critic, wave * 0.7, wave, wave * 0.4, rates, structured=structured, seed=60, step=0
    ).mean()
    loss.backward()
    assert wave.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert critic.blocks[0][0].weight.grad[:, 3:].abs().sum() > 0
    silent = torch.zeros_like(wave)
    a = structured_loss(critic, silent, silent, silent, rates, structured=True, seed=60, step=0)
    b = critic.discriminator_loss(silent, silent, silent, rates)
    torch.testing.assert_close(a, b)


def test_feature_gradient_finite_difference_and_shift_magnitudes():
    torch.manual_seed(60)
    critic = ContextTextureCritic(True).double()
    wave = torch.randn(2, 1536, dtype=torch.float64) * 0.02
    with torch.no_grad():
        for layers in critic.blocks:
            layers[0].weight[:, 3:].normal_(0, 0.02)
    rates = torch.tensor([44100, 48000])
    predicted = (wave * 0.8).requires_grad_()
    loss = critic.reference_loss(predicted, wave, wave * 0.4, rates).sum()
    (gradient,) = torch.autograd.grad(loss, predicted)
    direction = torch.randn_like(wave)
    direction /= direction.norm()
    delta = 1e-6
    plus = critic.reference_loss(predicted + delta * direction, wave, wave * 0.4, rates).sum()
    minus = critic.reference_loss(predicted - delta * direction, wave, wave * 0.4, rates).sum()
    assert float(((plus - minus) / (2 * delta)).detach()) == pytest.approx(
        float((gradient * direction).sum()), rel=1e-4, abs=1e-10
    )
    shifted = shifted_high(wave, rates, 3 / 8)
    torch.testing.assert_close(torch.fft.rfft(shifted).abs(), torch.fft.rfft(wave).abs())
