import pytest
import torch

from models.spectral_texture import SpectralTextureCritic, spectrum_pair


def setup():
    torch.manual_seed(58)
    critic = SpectralTextureCritic().double()
    wave = torch.randn(2, 1536, dtype=torch.float64) * 0.02
    return critic, wave, torch.tensor([44100, 48000])


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0])
def test_reference_zero_and_stationary(scale):
    critic, wave, rates = setup()
    reference = (wave * scale).requires_grad_()
    observed = wave.detach() * scale
    predicted = reference.detach().clone().requires_grad_()
    loss = critic.reference_loss(predicted, reference, observed, rates)
    (gradient,) = torch.autograd.grad(loss.sum(), predicted)
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert reference.grad is None
    assert all(p.grad is None and p.requires_grad for p in critic.parameters())


def test_gradient_finite_difference_and_batch_isolation():
    critic, wave, rates = setup()
    predicted = (wave * 0.7).requires_grad_()
    value = critic.reference_loss(predicted, wave, wave * 0.4, rates)
    (grad,) = torch.autograd.grad(value.sum(), predicted)
    direction = torch.randn_like(predicted)
    direction /= direction.norm()
    epsilon = 1e-6
    plus = critic.reference_loss(predicted + epsilon * direction, wave, wave * 0.4, rates).sum()
    minus = critic.reference_loss(predicted - epsilon * direction, wave, wave * 0.4, rates).sum()
    assert float((plus - minus) / (2 * epsilon)) == pytest.approx(
        float((grad * direction).sum()), rel=1e-4, abs=1e-10
    )
    single = critic.reference_loss(predicted[:1], wave[:1], wave[:1] * 0.4, rates[:1])
    assert float(single[0]) == pytest.approx(float(value[0]), rel=1e-12)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0])
def test_critic_update_detaches_audio_and_preserves_frozen_generator(scale):
    critic, wave, rates = setup()
    predicted = (wave * scale).requires_grad_()
    reference = (wave.roll(13, dims=-1) * scale).requires_grad_()
    before = [p.detach().clone() for p in critic.parameters()]
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    loss = critic.discriminator_loss(predicted, reference, wave, rates).mean()
    loss.backward()
    assert predicted.grad is None and reference.grad is None
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, critic.parameters(), strict=True))


def test_context_and_scale_are_reference_independent():
    _, wave, rates = setup()
    a, mask = spectrum_pair(wave, wave, rates, 512)
    b, _ = spectrum_pair(wave * 2, wave, rates, 512)
    assert torch.equal(a[:, 0], b[:, 0])
    assert torch.equal(a[:, 2], b[:, 2])
    assert not torch.equal(a[:, 1], b[:, 1])
    assert torch.all(a[:, :2] * (1 - mask) == 0)
    with pytest.raises(ValueError, match="windows"):
        spectrum_pair(wave[:, :512], wave[:, :512], rates, 512)
