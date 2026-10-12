import pytest
import torch

from models.conditioned_shape import ARMS, make_shape_model
from models.linear_objective import linear_objective


@pytest.mark.parametrize("rate", [44100.0, 48000.0])
def test_exact_identity_initialization_and_stationary_identity_training(rate):
    torch.manual_seed(64)
    models = []
    for arm in ARMS:
        torch.manual_seed(65)
        models.append(make_shape_model(arm, width=8))
    for name, value in models[0].state_dict().items():
        for model in models[1:]:
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    magnitude = torch.rand(2, 5, 129)
    magnitude[:, :, 70:] *= 1e-8
    magnitude[1] = 0
    context = torch.rand(2, 5, 1538)
    context[1] = 0
    condition = torch.ones(2, 5, 3)
    condition[:, :, 0] = rate / 48000
    rates = torch.full((2, 1), rate)
    model = models[-1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(magnitude, context, condition, model.initial_state(2))
        torch.testing.assert_close(output, magnitude, atol=0, rtol=0)
        loss = linear_objective(output, magnitude, rates, burn=0)["total"]
        assert loss == 0
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all() and p.grad.abs().max() == 0 for p in model.parameters()
        )
        optimizer.step()


def test_missing_bins_can_generate_and_weak_gradient_is_finite():
    torch.manual_seed(66)
    model = make_shape_model("anchored_band", width=8)
    magnitude = torch.zeros(1, 4, 129)
    magnitude[:, :, :20] = 1
    context = torch.rand(1, 4, 1538)
    condition = torch.ones(1, 4, 3)
    target = magnitude.clone()
    target[:, :, 100] = 0.01
    prediction, _ = model(magnitude, context, condition, model.initial_state(1))
    loss = linear_objective(prediction, target, torch.tensor([[48000.0]]), burn=0)["total"]
    loss.backward()
    for head in (model.energy_head, model.decoder[-1]):
        assert head.weight.grad.norm() > 0
        assert torch.isfinite(head.weight.grad).all()
    with torch.no_grad():
        model.energy_head.bias[-1] = 1
        model.decoder[-1].bias[100] = 0.5
    generated, _ = model(magnitude, context, condition, model.initial_state(1))
    assert generated[0, :, 100].min() > 1e-5
    assert generated[0, :, 100].min() > generated[0, :, 99].max() * 10


def test_nonzero_shape_model_chunk_causality_and_channel_isolation():
    torch.manual_seed(67)
    model = make_shape_model("anchored_band", width=8).eval()
    with torch.no_grad():
        model.energy_head.weight.normal_(std=0.005)
        model.decoder[-1].weight.normal_(std=0.005)
    m, c = torch.rand(2, 12, 129), torch.rand(2, 12, 1538)
    condition = torch.ones(2, 12, 3)
    state = model.initial_state(2)
    whole, final = model(m, c, condition, state)
    left, mid = model(m[:, :5], c[:, :5], condition[:, :5], state)
    right, end = model(m[:, 5:], c[:, 5:], condition[:, 5:], mid)
    torch.testing.assert_close(whole, torch.cat((left, right), dim=1))
    torch.testing.assert_close(final, end)
    single, _ = model(m[1:], c[1:], condition[1:], model.initial_state(1))
    torch.testing.assert_close(whole[1:], single)
    m[:, 5:] = 100
    changed, _ = model(m, c, condition, state)
    torch.testing.assert_close(whole[:, :5], changed[:, :5])


def test_anchored_forward_finite_difference():
    torch.manual_seed(68)
    model = make_shape_model("anchored_band", width=8).double()
    with torch.no_grad():
        model.energy_head.weight.normal_(std=0.001)
        model.decoder[-1].weight.normal_(std=0.001)
    m = (torch.rand(1, 3, 129, dtype=torch.float64) + 0.1).requires_grad_()
    c = torch.rand(1, 3, 1538, dtype=torch.float64)
    condition = torch.ones(1, 3, 3, dtype=torch.float64)
    state = model.initial_state(1).double()
    assert torch.autograd.gradcheck(
        lambda x: model(x, c, condition, state)[0], (m,), fast_mode=True
    )
