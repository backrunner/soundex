import pytest
import torch

from models.factorized_envelope import factor_objective
from scripts.audit_energy_objective import band_terms, synthetic_rankings


@pytest.mark.parametrize("rate", [44100.0, 48000.0])
@pytest.mark.parametrize("level", [0.0, 1e-8, 1.0])
def test_audit_decomposition_matches_existing_loss_and_gradient(rate, level):
    torch.manual_seed(19)
    target = torch.rand(1, 7, 129, dtype=torch.float64) * level
    pred = (torch.rand_like(target) * level + 1e-9).requires_grad_()
    rates = torch.tensor([[rate]], dtype=torch.float64)
    actual = factor_objective(pred, target, rates, burn=0)
    bands = band_terms(pred, target, rates, burn=0)
    decomposed = sum(b["total"] for b in bands)
    torch.testing.assert_close(decomposed, actual["total"], atol=1e-12, rtol=1e-12)
    for key in ("band_envelope", "band_shape", "envelope_temporal", "amplitude"):
        torch.testing.assert_close(sum(b["weight"] * b[key] for b in bands), actual[key])
    a = torch.autograd.grad(actual["total"], pred, retain_graph=True)[0]
    b = torch.autograd.grad(decomposed, pred)[0]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-10)


def test_fixed_counterexample_exposes_energy_shape_ranking_tradeoff():
    rows = {r["name"]: r for r in synthetic_rankings()}
    assert rows["reference"]["loss"]["total"] == 0
    under, flat = rows["correct_shape_100x_under"], rows["correct_energy_flat_shape"]
    assert flat["loss"]["total"] < under["loss"]["total"]
    assert flat["metrics"]["band_16000_24001_sc"] > under["metrics"]["band_16000_24001_sc"]
    over, under = rows["correct_shape_10x_over"], rows["correct_shape_10x_under"]
    assert abs(over["loss"]["total"] - under["loss"]["total"]) < 0.001
    assert over["metrics"]["band_16000_24001_sc"] > 9 * under["metrics"]["band_16000_24001_sc"]
