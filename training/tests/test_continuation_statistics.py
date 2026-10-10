"""Continuation must freeze inference statistics without freezing learning."""

import pytest
import torch

from train import configure_generator_training_mode


@pytest.mark.parametrize("policy", ["update", "frozen"])
def test_optimizer_updates_affine_parameters_but_respects_statistics_policy(policy):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.Conv2d(2, 2, 1))
    norm = model[0]
    before = {key: value.clone() for key, value in norm.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Validation preceding an epoch must not override the configured policy.
    model.eval()
    configure_generator_training_mode(model, policy)
    model(torch.randn(4, 2, 1, 17) + 4).square().mean().backward()
    assert norm.weight.grad.abs().sum() > 0
    optimizer.step()
    assert model.training and model[1].training
    assert not torch.equal(norm.weight, before["weight"])
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        if policy == "frozen":
            assert torch.equal(norm.state_dict()[name], before[name])
        else:
            assert not torch.equal(norm.state_dict()[name], before[name])


def test_untracked_or_misspelled_freeze_policy_is_rejected():
    with pytest.raises(ValueError, match="must be 'update' or 'frozen'"):
        configure_generator_training_mode(torch.nn.Identity(), "freeze")
    with pytest.raises(ValueError, match="require tracked"):
        configure_generator_training_mode(
            torch.nn.BatchNorm2d(2, track_running_stats=False), "frozen"
        )
