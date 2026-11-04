"""Full-magnitude gradient path and same-timeline capacity expansion."""

import pytest
import torch

from models.band_capacity import capacity_model


@pytest.mark.parametrize("kind,count,state", [("small", 204418, 12354), ("double", 408023, 34026)])
def test_capacity_contract_gradients_causality_and_chunking(kind, count, state):
    model = capacity_model(kind, nonzero=True)
    assert sum(p.numel() for p in model.parameters()) == count
    assert all(p.requires_grad for p in model.parameters())
    assert model.state_size == state
    torch.manual_seed(71)
    x = torch.randn(2, 2, 14, 129)
    x[:, 0] = x[:, 0] * 10 - 40
    x.requires_grad_()
    ctx = torch.rand(2, 14, 1538)
    cond = torch.tensor([1.0, 0.6, 1.0]).expand(2, 14, 3).clone()
    out, h = model(x, ctx, cond, model.initial_state(2))
    grad = torch.autograd.grad(out[0, :, :7].sum(), x, retain_graph=True)[0]
    assert not grad[1].any() and not grad[0, :, 7:].any()
    out[:, 0].square().mean().backward()
    assert model.base.backbone.encoders[0][0].weight.grad.abs().sum() > 0
    assert model.base.adapter.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    pieces, s = [], model.initial_state(2)
    with torch.no_grad():
        for start, stop in ((0, 3), (3, 8), (8, 14)):
            y, s = model(x[:, :, start:stop], ctx[:, start:stop], cond[:, start:stop], s)
            pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces, 2), out, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(s, h, rtol=2e-5, atol=2e-5)


def test_explicit_training_scope_preserves_frozen_default_inference():
    joint = capacity_model("small", nonzero=True)
    from models.band_phase import BandPhaseCorrection
    from models.band_tcn import BandTCN

    frozen = BandPhaseCorrection(BandTCN(), phase_support="native", phase_observation="adjacent")
    frozen.load_state_dict(joint.state_dict())
    x = torch.zeros(1, 2, 4, 129)
    x[:, 0] = -200
    context = torch.zeros(1, 4, 1538)
    condition = torch.tensor([1.0, 0.6, 1.0]).expand(1, 4, 3)
    with torch.no_grad():
        a, ah = joint(x, context, condition, joint.initial_state(1))
        b, bh = frozen(x, context, condition, frozen.initial_state(1))
    assert torch.equal(a, b) and torch.equal(ah, bh)
    assert not any(p.requires_grad for p in frozen.base.parameters())
