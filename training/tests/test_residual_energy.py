import pytest
import torch
from test_factorized_envelope import inputs

from models.factorized_envelope import factor_objective
from models.residual_energy import ARMS, make_energy_model


def test_energy_coordinates_match_initial_weights_capacity_and_identity():
    models = []
    for arm in ARMS:
        torch.manual_seed(13)
        models.append(make_energy_model(arm, width=16))
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], atol=0, rtol=0)
    x, c, condition = inputs()
    x[:, :, 100:] = 0
    for model in models:
        out, _ = model(x, c, condition, model.initial_state(2))
        torch.testing.assert_close(out, x, atol=2e-6, rtol=2e-6)


def test_residual_energy_has_unattenuated_missing_band_jacobian():
    scale = torch.tensor([0.1], dtype=torch.float64)
    rms = scale * 1e-8
    grads = []
    for arm in ARMS:
        model = make_energy_model(arm, width=16).double()
        delta = torch.zeros_like(scale, requires_grad=True)
        energy = model.band_energy(rms, scale, delta)
        grads.append(torch.autograd.grad(energy.sum(), delta)[0])
    torch.testing.assert_close(grads[0], rms)
    torch.testing.assert_close(grads[1], scale / 2**0.5)
    assert grads[1] > grads[0] * 1e6


def test_energy_map_finite_difference_and_upper_bound():
    model = make_energy_model("rms_residual", width=16).double()
    rms = torch.tensor([0.02, 1e-9], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.1], dtype=torch.float64)
    delta = torch.tensor([-0.1, 0.01], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda d: model.band_energy(rms, scale, d), (delta,))
    assert (
        model.band_energy(rms, scale, delta * 10000) <= scale * torch.exp(torch.tensor(2.0))
    ).all()


@pytest.mark.parametrize("level", [0.0, 1e-10, 1.0])
def test_residual_full_model_weak_input_gradients_and_shape_independence(level):
    model = make_energy_model("rms_residual", width=16)
    x, c, condition = inputs(frames=84)
    x, c = x * level, c * level
    out, _ = model(x, c, condition, model.initial_state(2))
    loss = factor_objective(out, x * 1.1, torch.tensor([[44100.0], [48000.0]]))["total"]
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.energy_head.bias[-1] = 0.01
        before, _ = model(x, c, condition, model.initial_state(2))
        model.decoder[-1].bias.copy_(torch.linspace(-0.2, 0.2, 129))
        after, _ = model(x, c, condition, model.initial_state(2))
        for index, rate in enumerate((44100, 48000)):
            mask = torch.arange(129) * rate / 256 >= 16000
            torch.testing.assert_close(
                before[index, :, mask].square().mean(-1),
                after[index, :, mask].square().mean(-1),
                atol=1e-7,
                rtol=1e-5,
            )


def test_residual_streaming_channel_and_future_independence():
    model = make_energy_model("rms_residual", width=16).eval()
    with torch.no_grad():
        model.energy_head.weight.normal_(std=0.001)
        model.decoder[-1].weight.normal_(std=0.001)
        x, c, condition = inputs()
        full, final = model(x, c, condition, model.initial_state(2))
        state, pieces = model.initial_state(2), []
        for start, stop in ((0, 1), (1, 43), (43, 90)):
            part, state = model(x[:, start:stop], c[:, start:stop], condition[:, start:stop], state)
            pieces.append(part)
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(state, final, atol=3e-6, rtol=1e-5)
        single, _ = model(x[:1], c[:1], condition[:1], model.initial_state(1))
        torch.testing.assert_close(single, full[:1], atol=3e-6, rtol=1e-5)
        x[:, 60:] *= 5
        c[:, 60:] *= 5
        changed, _ = model(x, c, condition, model.initial_state(2))
        torch.testing.assert_close(changed[:, :60], full[:, :60], atol=0, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_residual_model_mps_cpu_forward_and_gradient():
    torch.manual_seed(7)
    cpu = make_energy_model("rms_residual", width=16)
    mps = make_energy_model("rms_residual", width=16).to("mps")
    mps.load_state_dict(cpu.state_dict())
    x, c, condition = inputs(frames=84)
    x[:, :, 100:] = 1e-10
    target = x + 0.001
    values = []
    for model, device in ((cpu, "cpu"), (mps, "mps")):
        out, _ = model(
            x.to(device), c.to(device), condition.to(device), model.initial_state(2, device=device)
        )
        loss = factor_objective(
            out, target.to(device), torch.tensor([[44100.0], [48000.0]], device=device)
        )["total"]
        loss.backward()
        values.append((out.detach().cpu(), model.energy_head.bias.grad.cpu()))
    torch.testing.assert_close(values[0][0], values[1][0], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(values[0][1], values[1][1], atol=1e-3, rtol=1e-4)
