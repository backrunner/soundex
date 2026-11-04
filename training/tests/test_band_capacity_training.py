"""P72 full Adam state, amplitude updates and final role/codec comparisons."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_fused_spectrum import batch_for

from models.band_capacity import capacity_model
from scripts.run_band_capacity_training import load, pack, panel_comparisons, update


@pytest.mark.parametrize("kind", ["small", "double"])
def test_joint_adam_resume_is_exact_and_updates_amplitude(kind):
    model = capacity_model(kind, 12)
    before = model.base.adapter.weight.detach().clone()
    batch = batch_for(0.01)
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0)
    batch["y"][0, 0, 89:, 85:100] += 1.0
    items = [(SimpleNamespace(batch=batch), torch.zeros(2, 160, 1538))] * 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    first = update(model, items, optimizer)
    assert first["changed_parameters"] > 0
    assert not torch.equal(before, model.base.adapter.weight)
    saved = copy.deepcopy(pack(model, kind, 12, optimizer, [first], "test"))
    other = load(saved)
    resumed = torch.optim.Adam(other.parameters(), lr=1e-5)
    resumed.load_state_dict(saved["optimizer"])
    assert update(model, items, optimizer) == update(other, items, resumed)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())


def test_panel_comparisons_keep_identity_codec_and_native_roles():
    rows = [
        {
            "source": "source",
            "case": role + codec,
            "role": role,
            "codec": codec,
            "path": path,
            "scores": {"mr": value, "hf_sc": value},
        }
        for role in ("paired", "identity")
        for codec in ("mp3", "aac")
        for path, value in (
            ("untreated", 1.0),
            ("raw/12-double", 0.8),
            ("native/12-double", 0.9),
            ("raw/12-small", 1.0),
            ("native/12-small", 1.0),
            ("raw/legacy", 1.0),
            ("native/legacy", 1.0),
            ("raw/identity_dsp", 1.0),
            ("native/identity_dsp", 1.0),
        )
    ]
    result = panel_comparisons(rows, [12])
    assert len(result) == 48
    assert result["paired/all/native/12-double-vs-native/12-small"]["mr"][
        "mean_delta"
    ] == pytest.approx(-0.1)
    assert result["identity/aac/raw/12-double-vs-untreated"]["hf_sc"]["sources"] == 1
