import pytest
import torch

from models.direct_spectrum_objective import direct_spectrum_loss
from scripts import run_direct_spectrum_study as runner
from scripts.run_direct_spectrum_study import ARMS, make_model, optimized_loss


def batch(level=-20.0):
    torch.manual_seed(29)
    y = torch.randn(2, 2, 96, 129, dtype=torch.float64) * 0.2
    y[:, 0] += level
    x = y.clone()
    x[0, 0, :, 48:] -= 30
    return {
        "x": x,
        "y": y,
        "rate": torch.tensor([[48000], [44100]]),
        "identity": torch.tensor([0, 1]),
    }


@pytest.mark.parametrize("level", [-200.0, -140.0, -20.0])
def test_reference_is_zero_with_finite_zero_gradient(level):
    b = batch(level)
    p = b["y"].clone().requires_grad_()
    result = direct_spectrum_loss(p, b)
    assert abs(result["total"].item()) < 1e-15
    result["total"].backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().max() < 1e-10


@pytest.mark.parametrize("level", [-190.0, -140.0, -20.0])
def test_missing_high_band_gets_finite_descent_gradient_without_dsp(level):
    b = batch(level)
    p = b["x"].clone().requires_grad_()
    # Stay above the feature floor when testing the direction of a weak gradient.
    p.data[0, 0, :, 48:] = level - 5
    direct_spectrum_loss(p, b)["total"].backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[0, 0, 89:, 48:].sum() < 0
    assert p.grad[:, :, :89].abs().max() == 0


def test_direct_target_is_independent_of_dsp_and_identity_uses_input():
    b = batch()
    p = b["x"].clone()
    initial = direct_spectrum_loss(p, b)["total"]
    b.update(mask=torch.zeros_like(p[:, :1]), active=torch.zeros(2, 96))
    b["y"][1] += 10
    torch.testing.assert_close(direct_spectrum_loss(p, b)["total"], initial, atol=0, rtol=0)
    b["y"][0, 0, :, 48:] += 5
    assert direct_spectrum_loss(p, b)["total"] != initial


def test_true_transient_and_phase_are_not_penalized_or_smoothed():
    b = batch()
    b["y"][0, 0, 92, 60:] += 20
    p = b["y"].clone()
    assert direct_spectrum_loss(p, b)["total"] == 0
    p[0, 0, 92, 60:] -= 20
    assert direct_spectrum_loss(p, b)["components"]["dynamics"][0] > 0
    p = b["y"].clone()
    p[0, 1, 89:, 60:] += 0.5
    result = direct_spectrum_loss(p, b)
    assert result["components"]["complex"][0] > 0
    assert result["components"]["amplitude"][0] < 1e-10


@pytest.mark.parametrize("coordinate", [(0, 0, 92, 75), (0, 1, 92, 75)])
def test_finite_difference(coordinate):
    b = batch()
    p = b["y"].clone()
    p[0, 0, 89:, 48:] -= 3
    p[0, 1, 89:, 48:] += 0.3
    p.requires_grad_()
    direct_spectrum_loss(p, b)["total"].backward()
    analytical = p.grad[coordinate].item()
    with torch.no_grad():
        v, eps = p[coordinate].item(), 1e-5
        p[coordinate] = v + eps
        plus = direct_spectrum_loss(p, b)["total"].item()
        p[coordinate] = v - eps
        minus = direct_spectrum_loss(p, b)["total"].item()
        p[coordinate] = v
    assert analytical == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-9)


def test_raw_target_does_not_receive_gradients_and_invalid_regions_fail():
    b = batch()
    b["y"].requires_grad_()
    p = b["x"].clone().requires_grad_()
    direct_spectrum_loss(p, b)["total"].backward()
    assert b["y"].grad is None
    with pytest.raises(ValueError, match="two supervised"):
        direct_spectrum_loss(p, b, burn=95)


def test_study_arms_have_explicit_loss_routes():
    for arm, expected in zip(ARMS, [(1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 1, 0)], strict=True):
        terms = {k: torch.tensor(1.0, requires_grad=True) for k in ("deployed", "raw", "budget")}
        optimized_loss(arm, terms).backward()
        assert tuple(int(v.grad) if v.grad is not None else 0 for v in terms.values()) == expected


def test_budget_diagnostic_preserves_initial_tensors_and_fixes_unit_scale():
    control = make_model(29, "deployed")
    unit = make_model(29, "raw_unit_budget")
    for k, v in control.state_dict().items():
        torch.testing.assert_close(unit.state_dict()[k], v, atol=0, rtol=0)
    assert all(not p.requires_grad for p in unit.budget_head.parameters())
    latent = torch.randn(3, 7, 192)
    torch.testing.assert_close(
        torch.nn.functional.softplus(unit.budget_head(latent)), torch.ones(3, 7, 5)
    )


@pytest.mark.parametrize("arm", ARMS)
def test_diagnostic_resume_is_exact_and_rejects_objective_drift(tmp_path, monkeypatch, arm):
    from types import SimpleNamespace

    def make(seed, _arm):
        torch.manual_seed(seed)
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(runner, "make_model", make)
    monkeypatch.setattr(runner, "predict_budget", lambda m, b, c: (m(c), m(c).square()))
    monkeypatch.setattr(
        runner,
        "objective_terms",
        lambda p, b, o, w: {
            "raw": (p - 0.5).square().mean(),
            "deployed": p.square().mean(),
            "budget": b.mean(),
        },
    )
    monkeypatch.setattr(runner, "progress", lambda *args, **kwargs: None)
    batches = [(SimpleNamespace(batch={}), torch.ones(2, 2))]
    config = {"steps": 128, "rate": 1e-4, "weight": 0.13}
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    runner.train(arm, 29, full, config, batches)
    save = runner.atomic_save

    def interrupt(value, path):
        save(value, path)
        if path.name == "latest.pth" and value["step"] == 64:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.train(arm, 29, resumed, config, batches)
    monkeypatch.setattr(runner, "atomic_save", save)
    with pytest.raises(ValueError, match="signature drift"):
        runner.train(arm, 29, resumed, {**config, "weight": 0.2}, batches)
    runner.train(arm, 29, resumed, config, batches)
    a, b = [torch.load(r / f"{arm}-29/step-128.pth", weights_only=False) for r in (full, resumed)]
    assert a["history"] == b["history"]
    for k, v in a["model"].items():
        torch.testing.assert_close(v, b["model"][k], atol=0, rtol=0)
    for k, v in a["optimizer"]["state"].items():
        for n, tensor in v.items():
            torch.testing.assert_close(tensor, b["optimizer"]["state"][k][n], atol=0, rtol=0)
