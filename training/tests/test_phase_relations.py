"""P80 reference, support, cusp and actual parameter-routing controls."""

import math

import pytest
import torch

from evaluation.phase_routed_update import backward
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import phase_objective
from models.losses import phase_gradient_terms
from models.phase_relations import auxiliary, distance, phase_relations, relation_inputs
from models.restoration_head import predict_budget
from scripts.audit_phase_relations import joint_gate
from tests.test_fused_spectrum import batch_for
from tests.test_phase_routed_update import fixture


def test_missing_or_undefined_spectral_evidence_never_passes():
    import copy

    values = {
        k: dict(mean_delta=-1e-4, p77_numerical_scale=1e-6, undefined_sources=[], sources=20)
        for k in ("mr", "hf_sc", "realized_complex")
    }
    assert joint_gate(values)
    assert not joint_gate({})
    for field, value in [
        ("mean_delta", None),
        ("mean_delta", float("nan")),
        ("mean_delta", -1e-8),
        ("p77_numerical_scale", None),
        ("undefined_sources", ["source"]),
        ("sources", 0),
    ]:
        changed = copy.deepcopy(values)
        changed["hf_sc"][field] = value
        assert not joint_gate(changed)


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e-5, 0.01])
def test_original_reference_stationary_with_finite_weak_gradients(kind, scale):
    b = batch_for(scale)
    b["identity"] = torch.tensor([False, True])
    p = b["y"].clone().requires_grad_()
    terms = phase_relations(p, b, kind)
    value = terms["per_example"].sum()
    g = torch.autograd.grad(value, p)[0]
    assert value == 0
    assert torch.isfinite(g).all() and g.count_nonzero() == 0
    changed = p.detach().clone()
    changed[:, 1, 100:, 80:] += 0.3
    changed.requires_grad_()
    q = phase_relations(changed, b, kind)["per_example"].sum()
    h = torch.autograd.grad(q, changed)[0]
    assert torch.isfinite(h).all()
    if scale in (0.0, 1e-8):
        assert not terms["support_mass"]["instantaneous"].any()
        assert q == 0 and h.count_nonzero() == 0
    else:
        assert q > 0 and h.abs().sum() > 0


def test_unsupported_silent_phase_does_not_exempt_invented_magnitude():
    b = batch_for(0.0)
    b["identity"] = torch.tensor([False, True])
    p = b["y"].clone()
    p[:, 0, 100:, 80:] = -20
    p.requires_grad_()
    for kind in ("circular", "wrapped"):
        assert not phase_relations(p, b, kind)["per_example"].any()
    direct = direct_spectrum_loss(p, b)["total"]
    g = torch.autograd.grad(direct, p)[0]
    assert direct > 0 and torch.isfinite(g).all() and g[:, 0].abs().sum() > 0


def test_legacy_circular_relation_is_the_same_kernel_with_declared_half_scale():
    b = batch_for(0.01)
    b["identity"] = torch.tensor([False, True])
    p = b["y"].clone()
    p[:, 1, 100:, 80:] += 0.7
    delta, weight = relation_inputs(p, b)["instantaneous"]
    legacy = phase_gradient_terms(delta, torch.zeros_like(delta), weight)
    got = phase_relations(p, b, "circular")["components"]
    for current, old in [
        ("frequency", "phase_frequency_gradient"),
        ("time", "phase_time_gradient"),
    ]:
        torch.testing.assert_close(got[current].mean(), legacy[old] / 2, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
def test_wrapping_antipodes_and_finite_difference_away_from_cusps(kind):
    x = torch.tensor([-3.0, -0.2, 0.4, 3.0], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(distance(x, kind), distance(x + 2 * math.pi, kind))
    assert torch.autograd.gradcheck(lambda z: distance(z, kind), (x,))
    near = torch.tensor(math.pi - 1e-4, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(distance(near, kind), near)[0]
    assert g > 0
    assert float(g) < 1e-4 if kind == "circular" else float(g) == pytest.approx(1 / math.pi)
    antipode = torch.tensor(math.pi, dtype=torch.float64, requires_grad=True)
    exact = torch.autograd.grad(distance(antipode, kind), antipode)[0]
    assert torch.isfinite(exact)
    if kind == "wrapped":
        # The cusp is not differentiable: opposing one-sided slopes, same value.
        left = (distance(antipode.detach(), kind) - distance(antipode.detach() - 1e-6, kind)) / 1e-6
        right = (
            distance(antipode.detach() + 1e-6, kind) - distance(antipode.detach(), kind)
        ) / 1e-6
        assert left > 0 and right < 0
    else:
        assert exact.abs() < 1e-15


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
def test_support_warmup_prefix_and_magnitude_independence(kind):
    b = batch_for(0.01)
    b["identity"] = torch.tensor([False, True])
    b["mask"][..., :70] = 0
    b["active"][:, 99] = False
    b["y"][0, 0, :, 75] = -150
    p = b["y"].clone()
    p[:, 1] += torch.linspace(0, 0.5, 160)[None, :, None]
    p.requires_grad_()
    value = phase_relations(p, b, kind)["per_example"][0]
    g = torch.autograd.grad(value, p)[0]
    assert not g[1].any() and not g[:, 0].any()
    assert not g[:, :, :89].any() and not g[..., :70].any()
    assert not g[..., 128].any() and not g[:, :, 99].any() and not g[0, :, :, 75].any()
    # Truncated prefix supervision must not depend on any later output frame.
    prefix = {
        k: v[:, :, :120] if k in ("x", "y", "mask") else v[:, :120] if k == "active" else v
        for k, v in b.items()
    }
    q = phase_relations(p[:, :, :120], prefix, kind)["per_example"].sum()
    h = torch.autograd.grad(q, p)[0]
    assert not h[:, :, 120:].any()


@pytest.mark.parametrize("arm", ["circular_relation", "wrapped_relation"])
def test_real_head_routing_and_parameter_finite_difference(arm):
    model, batch, context = fixture()
    prediction, _ = predict_budget(model, batch, context)
    params = list(model.parameters())
    old = torch.autograd.grad(
        phase_objective(prediction, batch, include_fused=True), params, retain_graph=True
    )
    extra = torch.autograd.grad(auxiliary(prediction, batch, arm), params, retain_graph=True)
    backward(prediction, batch, model, arm, 1, auxiliary_fn=auxiliary)
    for (name, p), a, b in zip(model.named_parameters(), old, extra, strict=True):
        torch.testing.assert_close(
            p.grad, a + b if name.startswith("phase_head.") else a, rtol=1e-12, atol=1e-14
        )
    parameter = model.phase_head[-1].bias
    index = 90
    analytic = torch.autograd.grad(
        auxiliary(predict_budget(model, batch, context)[0], batch, arm), parameter
    )[0][index]
    original = parameter.detach()[index].clone()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter[index] = original + sign * 1e-5
        values.append(auxiliary(predict_budget(model, batch, context)[0], batch, arm).detach())
    with torch.no_grad():
        parameter[index] = original
    torch.testing.assert_close(analytic, (values[0] - values[1]) / 2e-5, rtol=2e-3, atol=1e-9)
