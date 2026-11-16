"""P82 common target, finite differences and waveform correspondence controls."""

import pytest
import torch

from evaluation.phase_routed_update import backward
from models.complex_synthesis import analyze
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import phase_objective
from models.realized_phase_relations import (
    COMPONENTS,
    auxiliary,
    correspondence,
    domain_spectra,
    prepare_relations,
    relations,
)
from models.restoration_head import predict_budget
from tests.test_fused_spectrum import batch_for
from tests.test_phase_routed_update import fixture


def batch(scale=0.01):
    b = batch_for(scale)
    b["identity"] = torch.tensor([False, True])
    # Production uses aligned original PCM, never clean-feature reconstruction.
    b["clean_wave"] = b["dry"].clone()
    return b


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e-5, 0.01])
def test_original_pcm_reference_has_zero_risk_and_finite_weak_derivatives(kind, scale):
    b = batch(scale)
    plan = prepare_relations(b)
    p = plan["reference"].clone().requires_grad_()
    result = relations(p, plan, kind)
    g = torch.autograd.grad(result["per_example"].sum(), p)[0]
    assert result["per_example"].count_nonzero() == 0
    assert torch.isfinite(g).all() and not g.any()
    changed = (p.detach() * torch.exp(torch.tensor(0.3j))).requires_grad_()
    loss = relations(changed, plan, kind)["per_example"].sum()
    grad = torch.autograd.grad(loss, changed)[0]
    assert torch.isfinite(grad).all()
    if scale in (0, 1e-8):
        assert loss == 0 and not grad.any()
    else:
        assert loss > 0 and grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
def test_zero_predictions_and_silent_targets_do_not_remove_magnitude_supervision(kind):
    b = batch()
    plan = prepare_relations(b)
    zero = torch.zeros_like(plan["reference"], requires_grad=True)
    g = torch.autograd.grad(relations(zero, plan, kind)["per_example"].sum(), zero)[0]
    assert torch.isfinite(g).all() and not g.any()
    b = batch(0)
    p = b["y"].clone()
    p[:, 0, 100:, 80:] = -20
    p.requires_grad_()
    assert auxiliary(p, b, "raw/" + kind) == 0
    value = direct_spectrum_loss(p, b)["total"]
    g = torch.autograd.grad(value, p)[0]
    assert value > 0 and torch.isfinite(g).all() and g[:, 0].abs().sum() > 0


@pytest.mark.parametrize("domain", ["direct", "raw", "fused"])
@pytest.mark.parametrize("kind", ["circular", "wrapped"])
def test_domain_finite_differences_and_pcm_frame_support(domain, kind):
    b = batch()
    plan = prepare_relations(b)
    assert plan["reference"].shape == (2, 69, 129)
    assert torch.equal(plan["reference"], analyze(b["clean_wave"])[:, 89:])
    p = b["y"].clone()
    p[:, 0, 100:, 70:] += 0.5
    p[:, 1, 100:, 70:] += torch.linspace(0.1, 0.6, 60)[None, :, None]
    # Frequency-varying error stays away from wrapped-L1's zero cusps.
    # The initial constant-frequency fixture and its failed central difference
    # remain in the P82 controls receipt; the kernel/tolerances are unchanged.
    p[:, 1, 100:, 70:] += torch.linspace(0.07, 0.8, 59)[None, None, :]
    p.requires_grad_()
    value = relations(domain_spectra(p, b, plan)[domain], plan, kind)["per_example"][0]
    g = torch.autograd.grad(value, p)[0]
    assert torch.isfinite(g).all() and not g[1].any()
    assert not g[:, :, :88].any() and not g[:, :, 159:].any()
    if domain == "direct":
        assert g[:, 0].abs().max() < 1e-14 and not g[:, :, :89].any()
    for axis in (0, 1):
        i = (0, axis, 110, 80)
        values = []
        for sign in (1, -1):
            q = p.detach().clone()
            q[i] += sign * 1e-5
            values.append(
                relations(domain_spectra(q, b, plan)[domain], plan, kind)["per_example"][0]
            )
        torch.testing.assert_close(g[i], (values[0] - values[1]) / 2e-5, rtol=2e-3, atol=1e-8)


def test_support_is_reference_only_and_limiter_does_not_change_target():
    b = batch(0.6)
    original = b["clean_wave"].clone()
    b["mask"][..., :70] = 0
    b["active"][:, 100] = False
    plan = prepare_relations(b)
    assert not plan["weight"][..., :70].any() and not plan["weight"][:, 11].any()
    p = b["y"].clone().requires_grad_()
    value = relations(domain_spectra(p, b, plan)["fused"], plan, "circular")["per_example"].sum()
    g = torch.autograd.grad(value, p)[0]
    assert torch.isfinite(g).all() and not g[..., :70].any()
    assert torch.equal(original, b["clean_wave"]) and original.abs().max() > 0.95
    assert value > 0


@pytest.mark.parametrize("kind", ["circular", "wrapped"])
def test_geodesic_correspondence_bound_includes_zero_and_wraps(kind):
    torch.manual_seed(82)
    target = torch.randn(2, 5, 9, dtype=torch.complex128)
    plan = {"reference": target, "weight": torch.rand(2, 5, 9, dtype=torch.float64)}
    a, b = [torch.randn_like(target) for _ in range(2)]
    a[0, 0, 0] = 0
    b[0, 0, 1] = 0
    a[1, 0, 0] = complex(-1, 1e-8)
    b[1, 0, 0] = complex(-1, -1e-8)
    for other in (a.clone(), b, a + torch.randn_like(a) * 1e-7):
        result = correspondence(a, other, plan, kind)
        assert torch.all(result["risk_difference"] <= result["risk_bound"] + 1e-14)
        for name in COMPONENTS:
            assert torch.all(
                result["component_difference"][name] <= result["component_bound"][name] + 1e-14
            )


@pytest.mark.parametrize("domain", ["direct", "raw", "fused"])
def test_actual_phase_head_routing_and_parameter_derivative(domain):
    model, b, context = fixture()
    arm = domain + "/circular"
    p, _ = predict_budget(model, b, context)
    params = list(model.parameters())
    old = torch.autograd.grad(phase_objective(p, b, include_fused=True), params, retain_graph=True)
    extra = torch.autograd.grad(auxiliary(p, b, arm), params, retain_graph=True)
    backward(p, b, model, arm, 1, auxiliary_fn=auxiliary)
    for (name, parameter), a, z in zip(model.named_parameters(), old, extra, strict=True):
        torch.testing.assert_close(
            parameter.grad, a + z if name.startswith("phase_head.") else a, rtol=1e-12, atol=1e-14
        )
    parameter = model.phase_head[-1].bias
    expected = torch.autograd.grad(
        auxiliary(predict_budget(model, b, context)[0], b, arm), parameter
    )[0][90]
    initial = parameter.detach()[90].clone()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter[90] = initial + sign * 1e-5
        values.append(auxiliary(predict_budget(model, b, context)[0], b, arm).detach())
    with torch.no_grad():
        parameter[90] = initial
    torch.testing.assert_close(expected, (values[0] - values[1]) / 2e-5, rtol=2e-3, atol=1e-9)


def test_mixed_objective_projection_matches_actual_routed_training_backward():
    from types import SimpleNamespace

    from models.fused_spectrum import BURN, fused_stages, prepare_fusion
    from models.multires_spectrum import spectral_loss
    from scripts.audit_phase_gradients import cosine, vector
    from scripts.audit_realized_relations import probe

    model, b, context = fixture()
    indices = torch.tensor([0, 0, 1, 1])
    b = {k: v[indices] if isinstance(v, torch.Tensor) else v for k, v in b.items()}
    context = context[indices]
    result = probe(model, (SimpleNamespace(batch=b), context), ["music", "speech"] * 2)
    for arm in ("old", "raw/circular"):
        model.zero_grad(set_to_none=True)
        prediction, _ = predict_budget(model, b, context)
        final = fused_stages(prediction, b["x"], prepare_fusion(b))["final"]
        region = slice(BURN * 128, final.shape[-1] - 128)
        target = spectral_loss(final[:, region], b["clean_wave"][:, region], b["rate"])[
            "per_example"
        ].log1p()[0]
        gradient = vector(torch.autograd.grad(target, list(model.parameters()), retain_graph=True))
        backward(prediction, b, model, arm, 1, auxiliary_fn=auxiliary)
        actual = cosine(vector([p.grad for p in model.parameters()]), gradient)
        reported = result["strata"]["music/paired"]["balanced_routed_full_direction_cosines"][arm][
            "fused_mr"
        ]
        assert actual == pytest.approx(reported, abs=1e-12)
