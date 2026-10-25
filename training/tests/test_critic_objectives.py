import pytest
import torch

from evaluation.critic_objectives import (
    copied_adam,
    flattened_gradient,
    pcm_feature_probe,
    task_losses,
)
from models.context_texture import ContextTextureCritic, structured_loss


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.02])
def test_task_decomposition_matches_registered_loss_and_gradient(scale):
    torch.manual_seed(61)
    critic = ContextTextureCritic(True).double()
    target = torch.randn(2, 2560, dtype=torch.float64) * scale
    rates = torch.tensor([44100, 48000])
    tasks, weights = task_losses(critic, target * 0.7, target, target * 0.5, rates, 61, 13)
    value = (tasks * weights).sum(-1)
    original = structured_loss(
        critic, target * 0.7, target, target * 0.5, rates, structured=True, seed=61, step=13
    )
    torch.testing.assert_close(value, original, atol=1e-14, rtol=1e-12)
    parameters = list(critic.parameters())
    torch.testing.assert_close(
        flattened_gradient(value.mean(), parameters),
        flattened_gradient(original.mean(), parameters),
        atol=1e-12,
        rtol=1e-9,
    )


def test_copied_persistent_adam_matches_actual_multi_parameter_step():
    torch.manual_seed(61)
    parameters = [torch.nn.Parameter(torch.randn(3)), torch.nn.Parameter(torch.randn(2))]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    for p in parameters:
        p.grad = torch.randn_like(p)
    optimizer.step()
    before = torch.cat([p.detach().flatten() for p in parameters])
    gradient = torch.randn(5)
    delta = copied_adam(before, gradient, {"optimizer": optimizer.state_dict()}, persistent=True)
    parameters[0].grad, parameters[1].grad = gradient[:3].clone(), gradient[3:].clone()
    torch.nn.utils.clip_grad_norm_(parameters, 5)
    optimizer.step()
    actual = torch.cat([p.detach().flatten() for p in parameters]) - before
    torch.testing.assert_close(delta, actual.double(), atol=0, rtol=0)


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.02])
def test_pcm_probe_is_stationary_at_reference(scale):
    torch.manual_seed(61)
    critic = ContextTextureCritic(True)
    wave = torch.randn(2, 2560) * scale
    state = {k: v.clone() for k, v in critic.state_dict().items()}
    rows = pcm_feature_probe(critic, wave, wave, wave, torch.tensor([44100, 48000]))
    assert all(
        r["step_rms"] == 0 and r["feature_before"] == 0 and r["mr_before"] == 0 for r in rows
    )
    assert all(torch.equal(v, state[k]) for k, v in critic.state_dict().items())
