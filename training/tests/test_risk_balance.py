import pytest
import torch

from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss
from models.risk_balance import reconstruction_risk
from scripts.run_risk_balance_study import ARMS, protocol
from tests.test_multires_spectrum import feature_batch


def test_compression_has_exact_derivative_without_dropping_large_errors():
    x = torch.tensor([0.1, 100.0, 0.001, 0.01], dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    identity = torch.tensor([False, False, True, True])
    value = reconstruction_risk(x, y, identity, compress=True, identity_weight=0.1)
    value.backward()
    expected = torch.tensor([0.25, 0.25, 0.025, 0.025]) / (1 + x.detach())
    torch.testing.assert_close(x.grad, expected.to(torch.float64), rtol=1e-7, atol=1e-12)
    assert (x.grad > 0).all()
    eps = 1e-5
    with torch.no_grad():
        plus, minus = x.clone(), x.clone()
        plus[1] += eps
        minus[1] -= eps
    numerical = (
        reconstruction_risk(plus, y, identity, compress=True, identity_weight=0.1)
        - reconstruction_risk(minus, y, identity, compress=True, identity_weight=0.1)
    ) / (2 * eps)
    torch.testing.assert_close(x.grad[1], numerical, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0])
def test_reference_silence_weak_and_normal_are_stationary(scale):
    b = feature_batch()
    b["clean_wave"] *= scale
    # Preserve this fixture's exact analysis/synthesis alignment.
    b["x"][:, 0] += 20 * torch.tensor(max(scale, 1e-20)).log10()
    if scale == 0:
        b["x"][:, 0] = -200
        b["x"][:, 1] = 0
    b["y"] = b["x"].clone()
    p = b["x"].clone().requires_grad_()
    direct = direct_spectrum_loss(p, b)["per_example"]
    mr = multires_spectrum_loss(p, b)["per_example"]
    risk = reconstruction_risk(direct, mr, b["identity"], compress=True)
    risk.backward()
    assert risk.abs() < 1e-8
    assert torch.isfinite(p.grad).all() and p.grad.abs().max() < 1e-7


def test_missing_hf_has_descent_gradient_and_identity_reweight_keeps_paired_scale():
    b = feature_batch()
    p = b["x"].clone()
    p[:, 0, 89:, 48:] -= 4
    p.requires_grad_()
    direct = direct_spectrum_loss(p, b)["per_example"]
    mr = multires_spectrum_loss(p, b)["per_example"]
    high = reconstruction_risk(direct, mr, b["identity"], compress=True)
    low = reconstruction_risk(direct, mr, b["identity"], compress=True, identity_weight=0.1)
    gh = torch.autograd.grad(high, p, retain_graph=True)[0]
    gl = torch.autograd.grad(low, p)[0]
    torch.testing.assert_close(gl[0], gh[0])
    torch.testing.assert_close(gl[1], gh[1] * 0.1)
    assert gl[0, 0, 89:, 48:].sum() < 0
    assert torch.isfinite(gl).all()


def test_original_control_and_invalid_roles():
    direct = torch.tensor([1.0, 2.0, 0.1, 0.3])
    mr = direct * 2
    identity = torch.tensor([False, False, True, True])
    torch.testing.assert_close(reconstruction_risk(direct, mr, identity), (direct + mr).mean())
    with pytest.raises(ValueError, match="roles"):
        reconstruction_risk(direct, mr, torch.ones(4))
    with pytest.raises(ValueError, match="identity weight"):
        reconstruction_risk(direct, mr, identity, identity_weight=0)


def test_protocol_is_one_architecture_fixed_seed_equal_budget_and_train_only():
    config = protocol({"multires_loss": {"version": 1}})
    assert config["steps"] == 1024 and config["seeds"] == [20261012]
    assert len({str(v) for v in config["model_configs"].values()}) == 1
    assert config["loss_routes"] == {a: {a: 1} for a in ARMS}
    assert "TRAINING" in config["review_scope"]
    assert protocol({"multires_loss": {}}, preflight=True)["steps"] == 2
    followup = protocol({"multires_loss": {}}, followup=True)
    assert followup["arms"] == ["compressed_identity_mid"] and followup["steps"] == 1024
    assert (
        followup["model_configs"]["compressed_identity_mid"] == config["model_configs"]["control"]
    )
