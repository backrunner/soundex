import copy

import pytest
import torch

from scripts.run_consistency_lr_study import restore


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(4, 3)).square().mean().backward()
    optimizer.step()


def test_adam_continuation_restores_moments_and_rng_without_mutating_checkpoint():
    torch.manual_seed(25)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        update(model, optimizer)
    checkpoint = copy.deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        }
    )
    update(model, optimizer)
    expected = copy.deepcopy(model.state_dict())
    resumed = restore(model, checkpoint, 0.001)
    update(model, resumed)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    reduced = restore(model, checkpoint, 0.0001)
    assert reduced.param_groups[0]["lr"] == 0.0001
    before = copy.deepcopy(reduced.state_dict())
    for key, value in before["state"].items():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                value[name], checkpoint["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    update(model, reduced)
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 0.001
    assert all(int(v["step"]) == 3 for v in checkpoint["optimizer"]["state"].values())


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_restore_rejects_invalid_learning_rate(rate):
    with pytest.raises(ValueError, match="learning rate"):
        restore(torch.nn.Linear(1, 1), {}, rate)
