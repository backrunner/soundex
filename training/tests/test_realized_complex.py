"""Numerical gates for a diagnostic, not a new approved training recipe."""

import pytest
import torch

from evaluation.realized_complex import realized_hf_terms
from models.fused_spectrum import BURN, fused_stages, prepare_fusion
from tests.test_fused_spectrum import batch_for


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.01])
def test_reference_stationarity_and_finite_weak_gradients(scale):
    torch.manual_seed(75)
    reference = torch.randn(2, 4096, dtype=torch.float64) * scale
    wave = reference.clone().requires_grad_()
    rates = torch.tensor([44100, 48000])
    result = realized_hf_terms(wave, reference, rates)
    gradient = torch.autograd.grad(result["complex"].sum(), wave)[0]
    torch.testing.assert_close(result["complex"], torch.zeros(2, dtype=torch.float64))
    assert not gradient.any()
    changed = (wave.detach() + torch.randn_like(wave) * max(scale, 1e-9)).requires_grad_()
    value = realized_hf_terms(changed, reference, rates)["complex"].mean()
    derivative = torch.autograd.grad(value, changed)[0]
    assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0


def test_phase_inversion_exposes_magnitude_blind_spot_and_absolute_power():
    rates = torch.tensor([[44100], [48000]])
    time = torch.arange(4096, dtype=torch.float64)[None] / rates
    wave = 0.1 * (2 * torch.pi * 10000 * time).sin()
    result = realized_hf_terms(-wave, wave, rates)
    assert result["magnitude"].abs().max() < 1e-12
    assert result["complex"].min() > 0.5
    first = result["bands"][0]
    torch.testing.assert_close(
        first["reference_power"], torch.full((2,), 0.005, dtype=torch.float64), rtol=2e-4, atol=1e-8
    )
    torch.testing.assert_close(first["complex_error_power"], 4 * first["reference_power"])
    torch.testing.assert_close(first["phase_interaction_power"], first["complex_error_power"])


@pytest.mark.parametrize("axis", [0, 1])
def test_native_proxy_feature_finite_difference_and_supervised_region(axis):
    batch = batch_for(0.01)
    prediction = batch["x"].clone()
    prediction[:, 0, 100:, 60:] += 0.5
    prediction[:, 1, 100:, 60:] += 0.1
    prediction.requires_grad_()
    plan = prepare_fusion(batch)

    def objective(p):
        final = fused_stages(p, batch["x"], plan)["final"]
        region = slice(BURN * 128, final.shape[-1] - 128)
        return realized_hf_terms(final[:, region], batch["clean_wave"][:, region], batch["rate"])[
            "complex"
        ].mean()

    gradient = torch.autograd.grad(objective(prediction), prediction)[0]
    assert not gradient[:, :, :88].any()
    index = (0, axis, 110, 80)
    plus, minus = prediction.detach().clone(), prediction.detach().clone()
    plus[index] += 1e-5
    minus[index] -= 1e-5
    numeric = (objective(plus) - objective(minus)) / 2e-5
    torch.testing.assert_close(gradient[index], numeric, rtol=1e-4, atol=1e-8)


def test_limiter_does_not_redefine_target_or_guarantee_zero_reference_loss():
    batch = batch_for(0.6)
    target = batch["clean_wave"].clone()
    final = fused_stages(batch["x"], batch["x"], prepare_fusion(batch))["final"]
    region = slice(BURN * 128, final.shape[-1] - 128)
    value = realized_hf_terms(final[:, region], target[:, region], batch["rate"])
    assert value["complex"].min() > 0
    assert torch.equal(target, batch["clean_wave"])


def test_full_joint_parameter_gradient_decomposition_in_double_precision():
    from types import SimpleNamespace

    from models.phase_temporal import phase_temporal_model
    from scripts.audit_realized_supervision import probe

    batch = batch_for(0.01)
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0, dtype=torch.float64)
    context = torch.zeros(2, 160, 1538, dtype=torch.float64)
    model = phase_temporal_model("phase", 75).double()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = probe(model, (SimpleNamespace(batch=batch), context))
    assert result["gradient_decomposition_max"] < 1e-10
    assert result["parameters"]["phase_head"]["norms"]["fused_complex"] > 0
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
