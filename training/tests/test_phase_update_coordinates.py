"""Physical-unit equivalence includes clipped gradients and persistent moments."""

import copy

import pytest
import torch

from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import (
    NAMES,
    SCALE,
    canonical_model_state,
    canonical_optimizer_state,
    continued,
    original_clip,
    stepper,
)
from models.restoration_head import predict_budget
from scripts.audit_phase_routing import continued as original
from scripts.audit_phase_routing import exact
from scripts.run_phase_temporal_training import pack
from tests.test_phase_routed_update import fixture


def saved_fixture():
    from types import SimpleNamespace

    model, batch, context = fixture()
    model = model.float()
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    items = [(SimpleNamespace(batch=batch), context.float())] * 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    history = [update(model, items, optimizer, "old")]
    return copy.deepcopy(pack(model, "phase", 76, optimizer, history, "unit-test")), items


@pytest.mark.parametrize("arm", ["old", "realized_phase"])
def test_forward_and_two_continued_adam_updates_are_bitwise_equivalent(arm):
    saved, items = saved_fixture()
    a, oa = original(saved)
    b, ob = continued(saved, 1)
    exact(canonical_model_state(b), a.state_dict())
    exact(canonical_optimizer_state(b, ob, saved["optimizer"]), oa.state_dict())
    for _ in range(2):
        pa, _ = predict_budget(a, items[0][0].batch, items[0][1])
        pb, _ = predict_budget(b, items[0][0].batch, items[0][1])
        assert torch.equal(pa, pb)
        ra = update(a, items, oa, arm)
        rb = update(b, items, ob, arm, adam_step=stepper(b))
        assert ra["gradient_norm"] == rb["gradient_norm"]
        exact(canonical_model_state(b), a.state_dict())
        exact(canonical_optimizer_state(b, ob, saved["optimizer"]), oa.state_dict())


def test_clipping_uses_original_units_even_when_active():
    saved, _ = saved_fixture()
    a, _ = original(saved)
    b, _ = continued(saved, 1)
    for (name, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        p.grad = torch.full_like(p, 2.0)
        q.grad = p.grad * (SCALE if name in NAMES else 1)
    norm_a = torch.nn.utils.clip_grad_norm_(a.parameters(), 5, error_if_nonfinite=True)
    norm_b = original_clip(b)(list(b.parameters()), 5, error_if_nonfinite=True)
    assert norm_a > 5 and torch.equal(norm_a, norm_b)
    for (name, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad / (SCALE if name in NAMES else 1))


def test_gain_changes_only_readout_learning_rate_not_forward_or_moments():
    saved, items = saved_fixture()
    a, oa = continued(saved, 1)
    b, ob = continued(saved, 64)
    exact(a.state_dict(), b.state_dict())
    exact(
        canonical_optimizer_state(a, oa, saved["optimizer"]),
        canonical_optimizer_state(b, ob, saved["optimizer"]),
    )
    assert oa.param_groups[0]["lr"] == ob.param_groups[0]["lr"]
    assert ob.param_groups[1]["lr"] == 64 * oa.param_groups[1]["lr"]
    update(a, items, oa, "old", adam_step=stepper(a))
    update(b, items, ob, "old", adam_step=stepper(b))
    for (name, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        if name not in NAMES:
            assert torch.equal(p, q)
    assert not torch.equal(a.phase_head[-1].weight, b.phase_head[-1].weight)
    # Canonical deployment conversion keeps the actual changed forward function.
    deployed, _ = original({**saved, "model": canonical_model_state(b)})
    actual = predict_budget(b, items[0][0].batch, items[0][1])[0]
    assert torch.equal(actual, predict_budget(deployed, items[0][0].batch, items[0][1])[0])
