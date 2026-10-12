import pytest
import torch

from models.linear_objective import ARMS, linear_objective, make_linear_model


@pytest.mark.parametrize("level", [0.0, 1e-10, 1e-5, 1.0])
@pytest.mark.parametrize("rate", [44100.0, 48000.0])
def test_linear_reference_optimum_and_finite_weak_gradients(level, rate):
    torch.manual_seed(15)
    target = torch.rand(1, 8, 129, dtype=torch.float64) * level
    pred = target.clone().requires_grad_()
    rates = torch.tensor([[rate]], dtype=torch.float64)
    loss = linear_objective(pred, target, rates, burn=0)
    assert loss["total"] == 0
    loss["total"].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().max() == 0
    wrong = (target + 1e-6).requires_grad_()
    result = linear_objective(wrong, target, rates, burn=0)
    assert result["total"] > 0
    assert result["weak_band_count"] == loss["weak_band_count"]
    result["total"].backward()
    assert torch.isfinite(wrong.grad).all() and (wrong.grad > 0).all()


def test_linear_ranking_does_not_reward_flat_fill_or_equate_over_and_under():
    target = torch.full((1, 8, 129), 0.1, dtype=torch.float64)
    target[..., 86:] = 0
    target[..., 100] = 0.01 * 43**0.5
    rate = torch.tensor([[48000.0]], dtype=torch.float64)
    losses = {}
    for name, factor in (("under100", 0.01), ("under10", 0.1), ("over10", 10.0), ("flat", None)):
        p = target.clone()
        p[..., 86:] = 0.01 if factor is None else target[..., 86:] * factor
        losses[name] = linear_objective(p, target, rate, burn=0)["total"]
    assert losses["flat"] > losses["under100"]
    assert losses["over10"] > 9 * losses["under10"]


def test_linear_finite_difference_batch_and_reference_dynamics():
    torch.manual_seed(16)
    p = torch.rand(2, 6, 129, dtype=torch.float64).requires_grad_()
    t = torch.rand_like(p)
    rate = torch.tensor([[44100.0], [48000.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda x: linear_objective(x, t, rate, burn=0)["total"], (p,), fast_mode=True
    )
    full = linear_objective(p, t, rate, burn=0)["total"]
    parts = [
        linear_objective(p[i : i + 1], t[i : i + 1], rate[i : i + 1], burn=0)["total"]
        for i in range(2)
    ]
    torch.testing.assert_close(full, sum(parts) / 2)
    smoothed = t.mean(dim=1, keepdim=True).expand_as(t)
    assert linear_objective(smoothed, t, rate, burn=0)["linear_dynamics"] > 0
    assert linear_objective(t, t, rate, burn=0)["linear_dynamics"] == 0


def test_linear_warmup_does_not_enter_loss_or_floor():
    p, t = torch.rand(1, 85, 129), torch.rand(1, 85, 129)
    rate = torch.tensor([[48000.0]])
    first = linear_objective(p, t, rate)
    p[:, :80], t[:, :80] = 1e6, 1e8
    second = linear_objective(p, t, rate)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_linear_arms_share_every_initial_parameter():
    models = []
    for arm in ARMS:
        torch.manual_seed(31)
        models.append(make_linear_model(arm, width=16))
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], atol=0, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_linear_mps_cpu_gradient_agreement():
    torch.manual_seed(17)
    t = torch.rand(2, 84, 129) * 0.1
    t[:, :, 86:] *= 1e-7
    p = (t + torch.rand_like(t) * 1e-4).requires_grad_()
    rate = torch.tensor([[44100.0], [48000.0]])
    gpu = p.detach().to("mps").requires_grad_()
    a = linear_objective(p, t, rate)["total"]
    b = linear_objective(gpu, t.to("mps"), rate.to("mps"))["total"]
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p.grad, gpu.grad.cpu(), atol=1e-4, rtol=1e-4)
