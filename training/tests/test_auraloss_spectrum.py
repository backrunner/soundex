"""Reference, weak-signal, batch-isolation and derivative contracts."""

import copy

import pytest
import torch

from evaluation.spectral_training import adam_update
from models.auraloss_spectrum import spectral_loss


@pytest.mark.parametrize("level", [0.0, 1e-8, 0.1])
def test_reference_zero_and_finite_gradient(level):
    torch.manual_seed(54)
    reference = torch.randn(2, 2560, dtype=torch.float64) * level
    wave = reference.clone().requires_grad_()
    result = spectral_loss(wave, reference, torch.tensor([44100, 48000]))
    value = result["per_example"].sum()
    value.backward()
    assert value == 0
    assert torch.equal(wave.grad, torch.zeros_like(wave))


def test_weak_signal_is_not_a_hard_floor_dead_zone():
    torch.manual_seed(54)
    reference = torch.randn(1, 2560, dtype=torch.float64) * 1e-8
    wave = (reference * 0.7).requires_grad_()
    value = spectral_loss(wave, reference, torch.tensor([48000]))["per_example"].sum()
    value.backward()
    assert value > 0
    assert torch.isfinite(wave.grad).all()
    assert wave.grad.norm() > 0


def test_no_batch_amplitude_coupling_and_detached_target():
    torch.manual_seed(54)
    target = torch.randn(2, 2560, dtype=torch.float64).requires_grad_()
    wave = (target.detach() + torch.randn_like(target) * 0.1).requires_grad_()
    rates = torch.tensor([44100, 48000])
    together = spectral_loss(wave, target, rates)["per_example"]
    alone = torch.cat(
        [
            spectral_loss(wave[i : i + 1], target[i : i + 1], rates[i : i + 1])["per_example"]
            for i in range(2)
        ]
    )
    torch.testing.assert_close(together, alone, rtol=0, atol=0)
    changed = wave.detach().clone()
    changed[1] *= 1000
    assert spectral_loss(changed, target, rates)["per_example"][0] == together[0]
    together.sum().backward()
    assert target.grad is None


def test_directional_finite_difference():
    torch.manual_seed(54)
    target = torch.randn(1, 2560, dtype=torch.float64) * 0.1
    wave = (target + torch.randn_like(target) * 0.01).requires_grad_()
    direction = torch.randn_like(wave)
    direction /= direction.norm()
    rates = torch.tensor([48000])
    value = spectral_loss(wave, target, rates)["per_example"].sum()
    derivative = (torch.autograd.grad(value, wave)[0] * direction).sum()
    epsilon = 1e-7
    plus = spectral_loss(wave.detach() + epsilon * direction, target, rates)["per_example"]
    minus = spectral_loss(wave.detach() - epsilon * direction, target, rates)["per_example"]
    torch.testing.assert_close(
        (plus - minus).sum() / (2 * epsilon), derivative, rtol=2e-4, atol=1e-7
    )


def test_true_adam_matches_torch_and_exact_resume():
    initial = torch.tensor([0.3, -0.8, 0.0])
    actual = torch.nn.Parameter(initial.clone())
    expected = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([actual], lr=1e-6)
    control = torch.optim.Adam([expected], lr=1e-6)
    for step in range(6):

        def closure(step=step):
            loss = ((actual - 0.2 * step) ** 2).sum()
            loss.backward()
            return loss.detach()

        record = adam_update([actual], optimizer, closure)
        control.zero_grad()
        ((expected - 0.2 * step) ** 2).sum().backward()
        torch.nn.utils.clip_grad_norm_([expected], 5)
        control.step()
        assert torch.equal(actual, expected)
        assert record["actual_displacement_l2"] > 0
        if step == 2:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = torch.optim.Adam([actual], lr=9)
            optimizer.load_state_dict(state)


@pytest.mark.parametrize("rates", [torch.tensor([32000]), torch.tensor([48000, 44100])])
def test_invalid_geometry_rejected(rates):
    with pytest.raises(ValueError):
        spectral_loss(torch.zeros(1, 2560), torch.zeros(1, 2560), rates)
