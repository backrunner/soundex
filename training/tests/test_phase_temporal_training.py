"""P74 exact Adam continuation and observed-phase learning diagnostics."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_fused_spectrum import batch_for

from scripts.run_band_capacity_training import update
from scripts.run_phase_temporal_training import fresh, inspect, load, pack


@pytest.mark.parametrize("kind", ["magnitude", "phase"])
def test_new_observations_learn_with_exact_adam_resume(kind):
    model = fresh(kind, 12)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    batch = batch_for(0.01)
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0)
    batch["y"][0, 0, 89:, 85:100] += 1.0
    items = [(SimpleNamespace(batch=batch), torch.zeros(2, 160, 1538))] * 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    first = update(model, items, optimizer)
    saved = copy.deepcopy(pack(model, kind, 12, optimizer, [first], "test"))
    other = load(saved)
    resumed = torch.optim.Adam(other.parameters(), lr=1e-5)
    resumed.load_state_dict(saved["optimizer"])
    assert update(model, items, optimizer) == update(other, items, resumed)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    changed = {k: int(torch.count_nonzero(v != before[k])) for k, v in model.state_dict().items()}
    report = inspect(model, items[:1], changed)
    derivative = report["records"][0]["phase_history_gradient_l1"]
    assert (derivative > 0) if kind == "phase" else (derivative == 0)
