import pytest
import torch

from models.envelope_objective import BANDS
from models.factorized_envelope import ARMS, factor_objective, make_factor_model


def inputs(batch=2, frames=90):
    g = torch.Generator().manual_seed(5)
    magnitude = torch.rand(batch, frames, 129, generator=g)
    context = torch.rand(batch, frames, 1538, generator=g)
    condition = torch.tensor([[[44100 / 48000, 0.5, 1.0]], [[1.0, 0.6, 0.0]]])[:batch].expand(
        -1, frames, -1
    )
    return magnitude, context, condition


def test_common_backbone_and_near_identity_initialization():
    models = []
    for arm in ARMS:
        torch.manual_seed(8)
        models.append(make_factor_model(arm, width=16))
    for key, value in models[0].state_dict().items():
        for model in models[1:]:
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    args = inputs()
    with torch.no_grad():
        for model in models:
            out, _ = model(*args, model.initial_state(2))
            torch.testing.assert_close(out, args[0], atol=2e-6, rtol=2e-6)


def test_factorized_energy_shape_independence_and_zero_band_generation():
    model = make_factor_model("factorized", width=16).eval()
    magnitude, context, condition = inputs()
    state = model.initial_state(2)
    frequency = torch.arange(129)[None] * torch.tensor([[44100.0], [48000.0]]) / 256
    with torch.no_grad():
        baseline, _ = model(magnitude, context, condition, state)
        # Every per-band RMS doubles while normalized shapes are unchanged.
        model.energy_head.bias.fill_(torch.log(torch.tensor(2.0)) / 4)
        boosted, _ = model(magnitude, context, condition, state)
        torch.testing.assert_close(boosted, 2 * baseline, rtol=2e-6, atol=2e-6)
        model.decoder[-1].bias.copy_(torch.linspace(-0.2, 0.2, 129))
        reshaped, _ = model(magnitude, context, condition, state)
        for low, high in BANDS:
            mask = ((frequency >= low) & (frequency < high))[:, None]
            torch.testing.assert_close(
                (reshaped.square() * mask).sum(-1),
                (boosted.square() * mask).sum(-1),
                atol=1e-4,
                rtol=2e-6,
            )
        assert (reshaped - boosted).abs().max() > 0.01
        magnitude[:, :, 100:] = 0
        model.energy_head.bias[-1] = 3.0
        filled, _ = model(magnitude, context, condition, state)
        assert (filled[:, :, 100:] > 0).all()


def test_factorized_streaming_future_independence_and_channels():
    model = make_factor_model("factorized", width=16).eval()
    with torch.no_grad():
        model.energy_head.weight.normal_(std=0.01)
        model.decoder[-1].weight.normal_(std=0.01)
        x, c, conditions = inputs()
        full, final = model(x, c, conditions, model.initial_state(2))
        state, pieces = model.initial_state(2), []
        for start, stop in ((0, 1), (1, 40), (40, 73), (73, 90)):
            output, state = model(
                x[:, start:stop], c[:, start:stop], conditions[:, start:stop], state
            )
            pieces.append(output)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(state, final, atol=3e-6, rtol=1e-5)
        single, _ = model(x[:1], c[:1], conditions[:1], model.initial_state(1))
        torch.testing.assert_close(single, full[:1], atol=3e-6, rtol=1e-5)
        x[:, 60:] *= 5
        c[:, 60:] *= 5
        changed, _ = model(x, c, conditions, model.initial_state(2))
        torch.testing.assert_close(changed[:, :60], full[:, :60], atol=0, rtol=0)


@pytest.mark.parametrize("level", [0.0, 1e-10, 1e-5, 1.0])
@pytest.mark.parametrize("rate", [44100.0, 48000.0])
def test_factor_loss_reference_and_weak_gradients(level, rate):
    target = torch.rand(1, 5, 129, dtype=torch.float64) * level
    prediction = target.clone().requires_grad_()
    rates = torch.tensor([[rate]], dtype=torch.float64)
    exact = factor_objective(prediction, target, rates, burn=0)
    assert all(float(v.detach()) == 0 for v in exact.values())
    exact["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    prediction = (target + 1e-6).requires_grad_()
    loss = factor_objective(prediction, target, rates, burn=0)["total"]
    loss.backward()
    assert loss > 0 and torch.isfinite(prediction.grad).all()


def test_factor_loss_finite_difference():
    torch.manual_seed(27)
    p = (torch.rand(1, 4, 129, dtype=torch.float64) + 0.1).requires_grad_()
    t = torch.rand_like(p)
    rate = torch.tensor([[48000.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: factor_objective(value, t, rate, burn=0)["total"],
        (p,),
        fast_mode=True,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_factor_loss_mps_cpu_gradient_parity():
    torch.manual_seed(26)
    target = torch.rand(2, 90, 129)
    rate = torch.tensor([[44100.0], [48000.0]])
    cpu = (torch.rand_like(target) + 0.01).requires_grad_()
    mps = cpu.detach().to("mps").requires_grad_()
    a = factor_objective(cpu, target, rate)["total"]
    b = factor_objective(mps, target.to("mps"), rate.to("mps"))["total"]
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cpu.grad, mps.grad.cpu(), atol=2e-6, rtol=3e-5)


def test_factorized_parameter_gradients_finite_for_zero_and_weak_input():
    for level in (0.0, 1e-8, 1.0):
        model = make_factor_model("factorized", width=16)
        x, c, condition = inputs(frames=84)
        out, _ = model(x * level, c * level, condition, model.initial_state(2))
        target = x * level * 1.1
        loss = factor_objective(out, target, torch.tensor([[44100.0], [48000.0]]))["total"]
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
