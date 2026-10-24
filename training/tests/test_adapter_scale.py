"""Finite-step descent, no-op, restoration and numerical-floor regressions."""

import pytest
import torch

from evaluation.adapter_scale import FACTORS, backtrack


def setup():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    return p, [p.detach().clone()], [torch.tensor([-10.0])]


def test_overshoot_backtracks_on_actual_objective_and_restores():
    p, original, delta = setup()
    calls = []

    def evaluate():
        calls.append(float(p.detach()))
        return {"total": float(p.detach().square().sum())}

    result = backtrack([p], original, delta, -20.0, 1.0, evaluate)
    assert result["selected_scale"] == FACTORS[2]
    assert [x["accepted"] for x in result["points"]] == [False, False, True]
    assert calls == [1 + s * -10 for s in FACTORS[:3]]
    assert torch.equal(p, original[0])


def test_non_descent_is_explicit_noop_without_metric_access():
    p, original, delta = setup()
    result = backtrack([p], original, delta, 1.0, 1.0, lambda: pytest.fail("not evaluated"))
    assert result["selected_scale"] == 0 and result["points"] == []
    assert torch.equal(p, original[0])


def test_tiny_or_zero_change_cannot_pass_and_all_attempts_remain():
    p, original, delta = setup()
    result = backtrack([p], original, delta, -20.0, 1.0, lambda: {"total": 1 - 1e-8})
    assert result["selected_scale"] == 0 and len(result["points"]) == len(FACTORS)
    assert not any(v["accepted"] for v in result["points"])
    assert torch.equal(p, original[0])
    result = backtrack([p], original, [torch.zeros_like(p)], -1.0, 1.0, lambda: {"total": 0.0})
    assert result["selected_scale"] == 0


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_errors_restore_original_parameters(failure):
    p, original, delta = setup()

    def evaluate():
        if failure == "exception":
            raise RuntimeError("metric failure")
        return {"total": float("nan")}

    with pytest.raises((RuntimeError, ValueError)):
        backtrack([p], original, delta, -20.0, 1.0, evaluate)
    assert torch.equal(p, original[0])


def test_invalid_local_derivative_rejected():
    p, original, delta = setup()
    with pytest.raises(ValueError, match="finite"):
        backtrack([p], original, delta, float("nan"), 1.0, lambda: {})
    assert torch.equal(p, original[0])
