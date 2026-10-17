"""Structural scope checks, not evidence of trained generalization or audio quality."""

import math

import pytest
import torch

from evaluation.head_export import export_graph, verify
from models.restoration_head import RestorationHead
from scripts.audit_crop_coverage import union_length


def setup(scope):
    torch.manual_seed(51)
    model = RestorationHead("complex", width=8, generation_budget=True, budget_scope=scope)
    with torch.no_grad():
        model.backbone.decoder[-1].bias.fill_(0.1)
    x = torch.zeros(1, 2, 3, 129)
    x[:, 0] = -40
    x[:, 0, :, 100:] = -200
    return model, (x, torch.ones(1, 3, 1538) * 0.01, torch.ones(1, 3, 3), model.initial_state(1))


def test_unit_budget_matches_legacy_and_missing_bins_remain_expressible():
    old, args = setup("global")
    new, _ = setup("residual")
    torch.testing.assert_close(old(*args)[0], new(*args)[0], atol=2e-5, rtol=1e-6)
    output = new(*args)[0]
    assert (output[:, 0, :, 100:] > args[0][:, 0, :, 100:]).all()


def test_zero_budget_removes_the_uncontrolled_local_residual_path():
    for scope in ("global", "residual"):
        model, args = setup(scope)
        hook = model.budget_head.register_forward_hook(
            lambda _m, _i, out: torch.full_like(out, -torch.inf)
        )
        result = model(*args)[0]
        hook.remove()
        if scope == "residual":
            torch.testing.assert_close(result, args[0], atol=0, rtol=0)
        else:
            assert (result[:, 0, :, :100] - args[0][:, 0, :, :100]).abs().max() > 0.1


@pytest.mark.parametrize("level", [-200.0, -140.0, -40.0])
def test_residual_budget_has_finite_gradient_and_matches_finite_difference(level):
    model, args = setup("residual")
    args[0][:, 0] = level

    def score():
        y = model(*args)[0]
        return y[:, 0].mean()

    loss = score()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    analytical = float(model.budget_head.bias.grad[2])
    with torch.no_grad():
        original = model.budget_head.bias[2].item()
        eps = 0.01
        model.budget_head.bias[2] = original + eps
        plus = float(score())
        model.budget_head.bias[2] = original - eps
        minus = float(score())
        model.budget_head.bias[2] = original
    numerical = (plus - minus) / (2 * eps)
    assert math.isfinite(numerical)
    assert abs(analytical - numerical) < max(0.005, abs(numerical) * 0.02)


def test_invalid_budget_scope_is_rejected():
    with pytest.raises(ValueError):
        RestorationHead("complex", budget_scope="residual")


def test_residual_scope_nonzero_export_state_and_reset_parity(tmp_path):
    model, _ = setup("residual")
    model.eval()
    result = verify(model, export_graph(model, tmp_path / "residual.onnx"))
    assert result["reset"] and result["silence"]
    assert result["sequential_hops"] == 320


def test_coverage_merges_nested_and_overlapping_intervals():
    assert union_length([(0, 10), (2, 4), (8, 12), (20, 22), (22, 24)]) == 16
    assert union_length([]) == 0
