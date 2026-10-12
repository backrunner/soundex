import copy

import torch

from models.conditioned_energy import make_conditioned_model
from models.factorized_envelope import BURN
from models.residual_energy import ResidualEnergyEnvelope
from scripts import audit_linear_optimization as audit


def test_energy_coordinate_amplifies_small_updates_in_weak_bands():
    # The deployed coordinate's local sensitivity is in full-spectrum units.
    # Compare identical optimizer-scale perturbations at strong and weak inputs.
    model = ResidualEnergyEnvelope(width=8).double()
    rms = torch.tensor([1.0, 1e-5], dtype=torch.float64)
    scale = torch.ones_like(rms)
    delta = torch.zeros_like(rms, requires_grad=True)
    initial = model.band_energy(rms, scale, delta)
    gradient = torch.autograd.grad(initial.sum(), delta)[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=1e-6, rtol=1e-6)
    changed = model.band_energy(rms, scale, torch.full_like(rms, 1e-3))
    relative = (changed - initial) / rms
    assert relative[1] > relative[0] * 99999


def test_first_step_scan_preserves_inputs_and_separates_zero_initialized_heads(monkeypatch):
    original = audit.make_linear_model
    monkeypatch.setattr(audit, "make_linear_model", lambda arm: original(arm, width=8))
    torch.manual_seed(45)
    frames = BURN + 4
    magnitude = torch.rand(1, frames, 129) * 0.5 + 0.01
    target = magnitude * 0.9
    batch = {
        "x": (20 * magnitude.log10())[:, None],
        "y": (20 * target.log10())[:, None],
        "rate": torch.tensor([[48000.0]]),
        "cutoff": torch.full((1, frames), 12000.0),
        "active": torch.ones(1, frames),
    }
    context = torch.rand(1, frames, 1538)
    saved = copy.deepcopy(batch)
    result = audit.first_step(46, (batch, context))
    assert result["gradients"]["backbone"]["norm"] == 0
    assert result["gradients"]["energy"]["norm"] > 0
    assert result["gradients"]["shape"]["norm"] > 0
    assert -1 <= result["amplitude_dynamics_gradient_cosine"] <= 1
    for scan in result["scans"]:
        assert torch.isfinite(torch.tensor(scan["loss"]["total"]))
    for key in saved:
        torch.testing.assert_close(batch[key], saved[key], atol=0, rtol=0)


def test_conditioned_energy_retains_identity_and_can_generate_missing_band():
    model = make_conditioned_model(width=8).double()
    scale = torch.ones(3, dtype=torch.float64)
    rms = torch.tensor([1.0, 1e-5, 1e-8], dtype=torch.float64)
    delta = torch.zeros_like(rms, requires_grad=True)
    value = model.band_energy(rms, scale, delta)
    old = ResidualEnergyEnvelope(width=8).double().band_energy(rms, scale, delta)
    torch.testing.assert_close(value, old, atol=0, rtol=0)
    gradient = torch.autograd.grad(value.sum(), delta)[0]
    assert 0 < gradient[-1] < 1e-4
    assert gradient[-1] < gradient[0] * 1e-3
    assert torch.autograd.gradcheck(lambda d: model.band_energy(rms, scale, d), (delta,), eps=1e-8)
    generated = model.band_energy(rms, scale, torch.ones_like(delta))
    assert generated[-1] > value[-1] * 1000
    capped = model.band_energy(rms, scale, torch.full_like(delta, 1e8))
    assert capped.max() <= torch.exp(torch.tensor(2.0, dtype=torch.float64))


def test_conditioned_model_has_identical_initial_state_and_function():
    torch.manual_seed(51)
    old = audit.make_linear_model("linear_reconstruction", width=8)
    torch.manual_seed(51)
    new = make_conditioned_model(width=8)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)
    magnitude = torch.rand(2, 5, 129)
    magnitude[1, :, 70:] = 0
    context = torch.rand(2, 5, 1538)
    condition = torch.ones(2, 5, 3)
    state = old.initial_state(2)
    a, _ = old(magnitude, context, condition, state)
    b, _ = new(magnitude, context, condition, state)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    # A silent physical channel remains finite and independent of the other.
    magnitude[1] = 0
    context[1] = 0
    prediction, _ = new(magnitude, context, condition, state)
    prediction.sum().backward()
    assert torch.isfinite(prediction).all()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters())
    single, _ = new(magnitude[1:], context[1:], condition[1:], new.initial_state(1))
    torch.testing.assert_close(prediction[1:], single)
