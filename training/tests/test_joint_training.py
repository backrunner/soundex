"""P57 full/adapter optimizer continuity and downstream diagnostic export."""

import copy
from types import SimpleNamespace

import onnx
import pytest
import torch

from models.adapter_trainability import load_trainable
from models.complex_synthesis import complex_features, synthesize
from scripts.run_joint_training import assert_frozen, export_joint, pack, step
from tests.test_adapter_trainability import checkpoint
from tests.test_spectral_adapter import fixture


def items():
    _, x, context, condition = fixture(frames=120)
    target = x.clone()
    target[0, 0, 89:, 80:] += 2
    batch = {
        "x": x,
        "y": target,
        "clean_wave": synthesize(complex_features(target)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0),
        "active": torch.ones(2, 120),
    }
    return [(SimpleNamespace(batch=batch), context)] * 4


@pytest.mark.parametrize("scope", ("adapter", "full"))
def test_joint_training_exact_resume_and_frozen_statistics(scope):
    start, _ = checkpoint(torch.float32)
    model = load_trainable(start, scope)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-7)
    batch = items()
    first = step(model, batch, optimizer)
    assert first["changed_parameters"] > 0
    assert_frozen(model, start)
    saved = copy.deepcopy(pack(model, start, optimizer, [first], "test"))
    other = load_trainable(saved, scope)
    resumed = torch.optim.Adam([p for p in other.parameters() if p.requires_grad], lr=1e-7)
    resumed.load_state_dict(saved["optimizer"])
    left, right = step(model, batch, optimizer), step(other, batch, resumed)
    assert left == right
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    assert saved["step"] == 1 and saved["training_scope"] == scope
    if scope == "full":
        assert not torch.equal(
            model.backbone.encoder[0].weight, start["model"]["backbone.encoder.0.weight"]
        )
    assert_frozen(other, start)


def test_full_training_export_is_causal_and_has_correct_provenance(tmp_path):
    start, _ = checkpoint(torch.float32)
    model = load_trainable(start, "full")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    step(model, items(), optimizer)
    path = export_joint(model, tmp_path / "export", "full")
    graph = onnx.load(path)
    meta = {p.key: p.value for p in graph.metadata_props}
    assert meta["soundex.study"] == "p57-joint-small-training"
    assert meta["soundex.training_scope"] == "full"
    assert meta["soundex.deployable"] == "false"
    assert meta["soundex.lookahead_samples"] == "0"
    # Reuse must validate the hash even after the diagnostic metadata annotation.
    assert export_joint(model, tmp_path / "export", "full") == path
