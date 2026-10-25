"""Same-function unfreezing, objective isolation and reserved-source rejection."""

import copy

import pytest
import torch

from evaluation.trainability_audit import assert_reserved_isolation, objective, terms
from models.adapter_trainability import SCOPES, load_trainable
from models.complex_synthesis import complex_features
from models.spectral_adapter import from_base, load_adapter
from scripts.audit_adapter_trainability import proposal, save_proposal
from tests.test_spectral_adapter import CONFIG, fixture


def checkpoint(dtype=torch.float64):
    base, x, context, condition = fixture(dtype)
    old = from_base({"model_config": CONFIG, "model": base.state_dict()}, "high").to(dtype).eval()
    with torch.no_grad():
        old.adapter.weight.normal_(0, 0.1)
    return {
        "format": "soundex-spectral-adapter-v1",
        "base_config": CONFIG,
        "support": "high",
        "model": copy.deepcopy(old.state_dict()),
    }, (x, context, condition)


@pytest.mark.parametrize("scope", SCOPES)
def test_exact_forward_state_parity_and_parameter_budget(scope):
    saved, inputs = checkpoint(torch.float32)
    old, new = load_adapter(saved).eval(), load_trainable(saved, scope)
    a = old.forward_details(*inputs, old.initial_state(2))
    b = new.forward_details(*inputs, new.initial_state(2))
    assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    for name, p in new.named_parameters():
        assert p.requires_grad == (
            scope == "full"
            or name.startswith("adapter.")
            or (scope == "decoder" and name.startswith("backbone.decoder."))
        )


def test_full_scope_connects_adapter_features_and_has_correct_derivative():
    saved, inputs = checkpoint()
    new = load_trainable(saved, "full").double()
    old = load_adapter(saved).double().eval()
    for p in old.parameters():
        p.requires_grad_(True)
    name, p = next((n, p) for n, p in new.named_parameters() if n.startswith("backbone.encoder."))
    old_p = dict(old.named_parameters())[name]

    def value(model):
        output = model(*inputs, model.initial_state(2))[0]
        return complex_features(output).abs().square().mean()

    g = torch.autograd.grad(value(new), p)[0]
    g_old = torch.autograd.grad(value(old), old_p)[0]
    assert torch.isfinite(g).all() and not torch.equal(g, g_old)
    direction = g / g.norm()
    before = p.detach().clone()
    with torch.no_grad():
        p.copy_(before - 1e-4 * direction)
        left = value(new)
        p.copy_(before + 1e-4 * direction)
        right = value(new)
        p.copy_(before)
    torch.testing.assert_close((right - left) / 2e-4, (g * direction).sum(), rtol=1e-4, atol=1e-10)


def test_identity_ablation_keeps_paired_scale_and_cannot_change_other_terms():
    values = {
        k: torch.tensor(float(i + 1), requires_grad=True)
        for i, k in enumerate(
            ("paired_direct", "paired_rendered", "identity_direct", "identity_rendered")
        )
    }
    assert objective(values, 1) == 5
    assert objective(values, 0) == 1.5
    gradient = torch.autograd.grad(objective(values, 0), list(values.values()))
    assert [float(g) for g in gradient] == [0.5, 0.5, 0, 0]
    with pytest.raises(ValueError):
        objective(values, 0.1)


def test_reserved_work_alias_transitive_bridge_is_rejected():
    selected = {"source_checksum": "c", "split": "train", "held_out": False}
    universe = [
        {"audio_sha256": "a", "work_id": "work"},
        {"source_checksum": "c", "work_id": "work"},
    ]
    with pytest.raises(ValueError, match="reserved"):
        assert_reserved_isolation([selected], universe, [{"audio_sha256": "a"}])
    assert_reserved_isolation([selected], universe, [{"audio_sha256": "unrelated"}])


def test_actual_proposal_matches_adam_and_freezes_other_parameters():
    saved, _ = checkpoint(torch.float32)
    reference = load_trainable(saved, "adapter")
    parameters = [p for p in reference.parameters() if p.requires_grad]
    count = sum(p.numel() for p in parameters)
    parts = {
        k: torch.randn(count) * 0.01
        for k in ("paired_direct", "paired_rendered", "identity_direct", "identity_rendered")
    }
    optimizer = torch.optim.Adam(parameters, lr=1e-6)
    gradient = 0.5 * (
        parts["paired_direct"]
        + parts["paired_rendered"]
        + (parts["identity_direct"] + parts["identity_rendered"])
    )
    offset = 0
    for p in parameters:
        p.grad = gradient[offset : offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(parameters, 5)
    optimizer.step()
    actual, _, report = proposal(saved, "adapter", parts, 1, 1e-6, "fresh")
    assert report["changed_parameters"] > 0
    assert all(
        torch.equal(a, b)
        for a, b in zip(actual.state_dict().values(), reference.state_dict().values(), strict=True)
    )


def test_first_proposal_creates_directory_and_binds_real_checkpoint(tmp_path):
    from evaluation.reporting import sha256_file

    path = tmp_path / "proposals" / "first.pth"
    entry = save_proposal({"model": {"value": torch.ones(3)}}, path)
    assert entry["sha256"] == sha256_file(path)
    assert torch.equal(torch.load(path, weights_only=True)["model"]["value"], torch.ones(3))


def test_terms_reproduce_p54_objective():
    from types import SimpleNamespace

    from evaluation.spectral_training import objective_value
    from models.complex_synthesis import synthesize

    saved, _ = checkpoint(torch.float32)
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
    item = (SimpleNamespace(batch=batch), context)
    model = load_trainable(saved, "adapter")
    torch.testing.assert_close(
        objective(terms(model, item)[0], 1), objective_value(model, item, "common")
    )
