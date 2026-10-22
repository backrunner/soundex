import copy

import pytest
import torch

from evaluation.head_export import export_graph, verify
from models.restoration_head import RestorationHead, load_restoration_head
from scripts.run_detector_condition_study import compare_endpoints, protocol
from scripts.run_direct_spectrum_study import MODEL
from tests.test_restoration_head import inputs


def models():
    torch.manual_seed(17)
    native = RestorationHead(**{**MODEL, "width": 8})
    fixed = RestorationHead(**{**MODEL, "width": 8, "detector_condition": "fixed"})
    with torch.no_grad():
        native.backbone.decoder[-1].weight.normal_(0, 0.01)
        native.budget_head.weight.normal_(0, 0.01)
    fixed.load_state_dict(native.state_dict())
    return native, fixed


def test_fixed_condition_matches_explicit_intervention_through_history_and_gradients():
    native, fixed = models()
    x, context, condition = inputs()
    original = condition.clone()
    condition.requires_grad_()
    expected = condition.detach().clone()
    expected[..., 1], expected[..., 2] = 0.5, 1.0
    actual, state = fixed(x, context, condition, fixed.initial_state(2))
    reference, reference_state = native(x, context, expected, native.initial_state(2))
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    torch.testing.assert_close(condition, original)
    (actual.square().mean() + state.square().mean()).backward()
    assert condition.grad[..., 1:].count_nonzero() == 0
    assert condition.grad[..., 0].abs().max() > 0
    assert all(torch.isfinite(p.grad).all() for p in fixed.parameters() if p.grad is not None)
    # Changed detector conditions, including warmup/history, cannot affect the model.
    changed = condition.detach().clone()
    changed[..., 1:] = torch.rand_like(changed[..., 1:])
    other, end = fixed(x, context, changed, fixed.initial_state(2))
    torch.testing.assert_close(other, actual, atol=0, rtol=0)
    torch.testing.assert_close(end, state, atol=0, rtol=0)


def test_initial_parameters_match_parent_and_checkpoint_keeps_policy():
    torch.manual_seed(20261012)
    native = RestorationHead(**MODEL)
    torch.manual_seed(20261012)
    fixed = RestorationHead(**MODEL, detector_condition="fixed")
    assert sum(p.numel() for p in fixed.parameters()) == 431015
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, fixed.state_dict()[name], atol=0, rtol=0)
    loaded = load_restoration_head(
        dict(
            arm="fixed_detector",
            model_config={**MODEL, "detector_condition": "fixed"},
            model=fixed.state_dict(),
        )
    )
    assert loaded.detector_condition == "fixed"
    with pytest.raises(ValueError):
        RestorationHead(**MODEL, detector_condition="typo")


def test_fixed_condition_exports_nonzero_streaming_model(tmp_path):
    _, fixed = models()
    result = verify(fixed.eval(), export_graph(fixed, tmp_path / "fixed.onnx"))
    assert result["sequential_hops"] == 320


def test_protocol_keeps_budget_loss_and_parameter_size():
    parent = dict(
        development=[],
        direct_loss={"version": 1},
        multires_loss={"version": 1},
        weight=0.1,
        model_configs={"compressed": MODEL},
    )
    full, smoke = protocol(parent), protocol(parent, preflight=True)
    assert full["steps"] == 16192 and len(full["seeds"]) == 2
    assert smoke["steps"] == 4 and len(smoke["seeds"]) == 1
    assert full["model_configs"]["control"] == MODEL
    assert full["training_device"] == "cpu"
    assert full["model_configs"]["fixed_detector"] == {**MODEL, "detector_condition": "fixed"}


def test_assessment_matches_controls_within_each_path_and_requires_both_seeds():
    config = dict(steps=4, seeds=[1, 2])
    groups = {
        f"{c}/{r}/native_output": {
            m: dict(mean=1.0, recordings={"a": 1.0}) for m in ("total", "dynamics")
        }
        for c in ("music_library", "speech_library")
        for r in ("paired", "identity")
    }
    cases = {
        f"{a}-4-{s}": {f: copy.deepcopy(groups) for f in ("current--native", "phase-only--native")}
        for a in ("control", "fixed_detector")
        for s in (1, 2)
    }
    for seed in (1, 2):
        metric = cases[f"fixed_detector-4-{seed}"]["phase-only--native"][
            "music_library/paired/native_output"
        ]["total"]
        metric.update(mean=0.9, recordings={"a": 0.9})
    result = compare_endpoints(cases, config)
    assert result["phase-only--native"]["research_screen_pass"]
    assert not result["current--native"]["research_screen_pass"]
    cases["fixed_detector-4-2"]["phase-only--native"]["speech_library/identity/native_output"][
        "total"
    ].update(mean=2, recordings={"a": 2})
    assert not compare_endpoints(cases, config)["phase-only--native"]["research_screen_pass"]
