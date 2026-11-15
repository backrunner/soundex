"""Persistent Adam algebra and observational replay controls."""

import torch

from evaluation.adam_attribution import components


def test_decomposition_matches_actual_double_adam_with_history():
    p = torch.nn.Parameter(torch.tensor([0.7, -0.4, 0.02, 0.0], dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=0.003, betas=(0.9, 0.99), eps=1e-8)
    for gradient in ([0.4, -0.7, 0.001, 0], [-0.3, 0.6, 0.003, 0]):
        p.grad = torch.tensor(gradient, dtype=p.dtype)
        optimizer.step()
    before = p.detach().clone()
    state = optimizer.state[p]
    old = torch.tensor([0.02, 0.03, 0.001, 0.0], dtype=p.dtype)
    routed = old + torch.tensor([0.01, -0.02, -0.004, 0.0], dtype=p.dtype)
    parts = components(
        state["exp_avg"],
        state["exp_avg_sq"],
        old,
        routed,
        lr=0.003,
        betas=(0.9, 0.99),
        eps=1e-8,
        step=3,
    )
    p.grad = routed
    optimizer.step()
    torch.testing.assert_close(p - before, parts["routed_step"], atol=2e-16, rtol=1e-12)
    torch.testing.assert_close(
        parts["routed_step"] - parts["old_step"], parts["increment"], atol=2e-18, rtol=1e-12
    )
    assert parts["routed_step"][-1] == 0
    # The auxiliary also changes normalization; its numerator alone is incomplete.
    assert parts["increment_denominator"].norm() > 0


def test_no_extra_gradient_has_no_increment_even_with_momentum():
    old = torch.tensor([1e-12, 0.0], dtype=torch.float64)
    parts = components(
        torch.ones_like(old),
        torch.ones_like(old),
        old,
        old,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        step=16385,
    )
    assert torch.equal(parts["increment"], torch.zeros_like(old))
    torch.testing.assert_close(parts["routed_step"], parts["old_step"], atol=1e-20, rtol=1e-14)
