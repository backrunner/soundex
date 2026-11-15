"""P76 routing, continued Adam and direct-loss reference controls."""

import copy
from types import SimpleNamespace

import pytest
import torch

from evaluation.phase_routed_update import auxiliary, backward, update
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import balanced_risk, phase_objective
from models.phase_temporal import phase_temporal_model
from models.restoration_head import predict_budget
from scripts.audit_phase_routing import continued, exact
from scripts.run_band_capacity_training import update as canonical_update
from scripts.run_phase_temporal_training import pack
from tests.test_fused_spectrum import batch_for


def fixture():
    batch = batch_for(0.01)
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0, dtype=torch.float64)
    context = torch.zeros(2, 160, 1538, dtype=torch.float64)
    return phase_temporal_model("phase", seed=76, nonzero=True).double(), batch, context


@pytest.mark.parametrize("arm", ["realized_phase", "direct_phase"])
def test_auxiliary_routing_preserves_entire_old_base_derivative(arm):
    model, batch, context = fixture()
    prediction, _ = predict_budget(model, batch, context)
    params = list(model.parameters())
    old = torch.autograd.grad(
        phase_objective(prediction, batch, include_fused=True), params, retain_graph=True
    )
    extra = auxiliary(prediction, batch, arm)
    full_extra = torch.autograd.grad(extra, params, retain_graph=True)
    # The real computation depends on base features: detaching magnitudes alone
    # would not remove these base derivatives. Test the parameter routing instead.
    assert any(
        g.abs().sum() > 0
        for (k, _), g in zip(model.named_parameters(), full_extra, strict=True)
        if k.startswith("base.")
    )
    backward(prediction, batch, model, arm, 1)
    for (name, p), a, b in zip(model.named_parameters(), old, full_extra, strict=True):
        torch.testing.assert_close(
            p.grad, a + b if name.startswith("phase_head.") else a, rtol=1e-12, atol=1e-14
        )
    # A fixed-state finite difference in an actual head parameter, separate from
    # the nonscalar routed update and detached model warmup.
    parameter = model.phase_head[-1].bias
    index = 90
    analytic = torch.autograd.grad(
        auxiliary(predict_budget(model, batch, context)[0], batch, arm), parameter
    )[0][index]
    original = parameter.detach()[index].clone()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter[index] = original + sign * 1e-4
        values.append(auxiliary(predict_budget(model, batch, context)[0], batch, arm).detach())
    with torch.no_grad():
        parameter[index] = original
    torch.testing.assert_close(analytic, (values[0] - values[1]) / 2e-4, rtol=2e-3, atol=1e-9)


def test_direct_control_is_exact_risk_difference():
    model, batch, context = fixture()
    prediction, _ = predict_budget(model, batch, context)
    direct = direct_spectrum_loss(prediction, batch)
    expected = balanced_risk(
        (direct["per_example"] + 0.1 * direct["components"]["complex"]).log1p(), batch["identity"]
    )
    before = balanced_risk(direct["per_example"].log1p(), batch["identity"])
    torch.testing.assert_close(before + auxiliary(prediction, batch, "direct_phase"), expected)


def test_old_arm_matches_canonical_update_and_copied_moments_exactly():
    model, batch, context = fixture()
    model = model.float()
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    items = [(SimpleNamespace(batch=batch), context.float())] * 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    history = [canonical_update(model, items, optimizer)]
    saved = copy.deepcopy(pack(model, "phase", 76, optimizer, history, "test-only"))
    a, oa = continued(saved)
    b, ob = continued(saved)
    original_state = copy.deepcopy(saved["optimizer"])
    for _ in range(2):
        update(a, items, oa, "old")
        canonical_update(b, items, ob)
        exact(a.state_dict(), b.state_dict())
        exact(oa.state_dict(), ob.state_dict())
    exact(saved["optimizer"], original_state)
    assert all(int(s["step"]) == 3 for s in oa.state_dict()["state"].values())


@pytest.mark.parametrize("scale", [0.0, 1e-8])
@pytest.mark.parametrize("arm", ["realized_phase", "direct_phase"])
def test_auxiliary_correct_reference_and_weak_gradients(scale, arm):
    batch = batch_for(scale)
    batch["identity"] = torch.tensor([False, True])
    prediction = batch["x"].clone().requires_grad_()
    value = auxiliary(prediction, batch, arm)
    gradient = torch.autograd.grad(value, prediction)[0]
    assert abs(float(value.detach())) < 1e-8
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() < 1e-7
