"""Causal structure and gradient routing, not trained restoration quality."""

import copy

import pytest
import torch

from evaluation.head_export import export_graph, verify
from models.restoration_head import RestorationHead, load_restoration_head, shared_state
from scripts.bench_context_deployment import final_labels
from scripts.run_small_mechanism_study import protocol


def make(**options):
    torch.manual_seed(28)
    return RestorationHead(
        "complex",
        width=8,
        generation_budget=True,
        budget_scope="residual",
        residual_slope=1 / 64,
        **options,
    )


def inputs(model, frames=12, level=-40.0):
    torch.manual_seed(29)
    x = torch.randn(2, 2, frames, 129) * 0.2
    x[:, 0] = level
    return x, torch.rand(2, frames, 1538) * 0.01, torch.ones(2, frames, 3), model.initial_state(2)


def activate(model):
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.02)
        model.budget_head.weight.normal_(0, 0.02)
        if model.phase_input:
            model.phase_projection.weight.normal_(0, 0.02)


def test_shared_initialization_old_checkpoint_and_zero_phase_are_exact():
    base = make()
    for flags in ({"decoder_normalization": True}, {"phase_input": True}):
        model = make(**flags)
        for key, value in shared_state(base).items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        args = inputs(model)
        torch.testing.assert_close(model(*args)[0], args[0], atol=0, rtol=0)
    phase = make(phase_input=True)
    activate(base)
    phase.load_state_dict(base.state_dict(), strict=False)
    args = inputs(base)
    for actual, expected in zip(phase(*args), base(*args), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    old = load_restoration_head(
        {
            "arm": "complex",
            "model_config": {
                "arm": "complex",
                "width": 8,
                "generation_budget": True,
                "budget_scope": "residual",
                "residual_slope": 1 / 64,
            },
            "model": base.state_dict(),
        }
    )
    torch.testing.assert_close(old(*args)[0], base(*args)[0], atol=0, rtol=0)


def test_detached_auxiliary_keeps_forward_and_main_gradients_exact():
    model = make()
    activate(model)
    args = inputs(model)
    a, _, budget = model.forward_details(*args)
    b, _, detached = model.forward_details(*args, detach_budget_auxiliary=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(budget, detached, atol=0, rtol=0)
    params = tuple(model.parameters())
    ga = torch.autograd.grad(a[:, 0].mean(), params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b[:, 0].mean(), params, retain_graph=True, allow_unused=True)
    for x, y in zip(ga, gb, strict=True):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    shared = model.backbone.encoder[0].weight
    head = model.budget_head.weight
    ga = torch.autograd.grad(budget.square().mean(), (shared, head))
    gb = torch.autograd.grad(detached.square().mean(), (shared, head), allow_unused=True)
    assert ga[0].norm() > 0 and gb[0] is None
    torch.testing.assert_close(ga[1], gb[1], atol=0, rtol=0)


@pytest.mark.parametrize("options", [{"decoder_normalization": True}, {"phase_input": True}])
@pytest.mark.parametrize("level", [-200.0, -140.0, -40.0])
def test_interventions_have_finite_silent_weak_and_regular_gradients(options, level):
    model = make(**options)
    activate(model)
    args = inputs(model, level=level)
    output, _ = model(*args)
    loss = output[:, 0].square().mean() + output[:, 1].square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_phase_input_can_change_residual_and_is_causal_with_independent_state():
    model = make(phase_input=True).eval()
    activate(model)
    x, context, condition, state = inputs(model, frames=90)
    with torch.no_grad():
        full, last = model(x, context, condition, state)
        frames, current = [], state
        for t in range(90):
            y, current = model(
                x[:, :, t : t + 1], context[:, t : t + 1], condition[:, t : t + 1], current
            )
            frames.append(y)
        torch.testing.assert_close(torch.cat(frames, 2), full, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(current, last, atol=2e-5, rtol=1e-5)
        changed = x.clone()
        changed[:, 1, 50:] += 0.9
        other, _ = model(changed, context, condition, state)
        torch.testing.assert_close(other[:, :, :50], full[:, :, :50], atol=0, rtol=0)
        assert (other[:, 0, 50:] - full[:, 0, 50:]).abs().max() > 1e-6
        mono, _ = model(x[1:], context[1:], condition[1:], state[1:])
        torch.testing.assert_close(mono, full[1:], atol=2e-5, rtol=1e-5)
        reset, _ = model(x, context, condition, model.initial_state(2))
        torch.testing.assert_close(reset, full, atol=0, rtol=0)


def test_decoder_normalization_preserves_gradient_under_negative_common_shift():
    plain, normalized = make(), make(decoder_normalization=True)
    activate(plain)
    normalized.load_state_dict(plain.state_dict(), strict=False)
    for model in (plain, normalized):
        with torch.no_grad():
            model.backbone.decoder[0].bias.sub_(20)
        model(*inputs(model))[0][:, 0].mean().backward()
    assert plain.backbone.decoder[0].weight.grad.norm() < 1e-15
    assert normalized.backbone.decoder[0].weight.grad.norm() > 1e-6


@pytest.mark.parametrize("options", [{"decoder_normalization": True}, {"phase_input": True}])
def test_nonzero_export_parity(options, tmp_path):
    model = make(**options).eval()
    activate(model)
    report = verify(model, export_graph(model, tmp_path / "model.onnx"))
    assert report["sequential_hops"] == 320 and report["reset"] and report["silence"]


def test_benchmark_uses_registered_endpoint_and_rejects_missing_candidate():
    config = {"primary_endpoint": 2048, "arms": ["control", "phase"], "seeds": [1, 2]}
    expected = {f"{arm}-2048-{seed}": {} for arm in config["arms"] for seed in config["seeds"]}
    assert final_labels(config, {**expected, "control-512-1": {}}) == sorted(expected)
    incomplete = copy.copy(expected)
    incomplete.pop("phase-2048-2")
    with pytest.raises(ValueError, match="incomplete"):
        final_labels(config, incomplete)


def test_protocol_changes_one_factor_and_does_not_scale_capacity():
    parent = {
        "seeds": [20261012, 20261013],
        "common": [],
        "weight": 0.13,
        "rate": 1e-4,
        "development": [],
        "shifted": [],
        "tolerance": 1e-6,
    }
    config = protocol(parent)
    assert config["steps"] == 2048 and len(config["seeds"]) == 2
    assert config["endpoints"] == [2048] and not config["smoke"]
    assert set(config["schedule_groups"].values()) == {"broad"}
    assert set(config["budget_weights"].values()) == {0.1}
    assert config["detached_budget_arms"] == ["budget_detach"]
    control = config["model_configs"]["control"]
    assert config["model_configs"]["budget_detach"] == control
    for arm, flag in (("decoder_norm", "decoder_normalization"), ("phase_input", "phase_input")):
        changed = {k for k in control if config["model_configs"][arm][k] != control[k]}
        assert changed == {flag}
    for value in config["model_configs"].values():
        assert value["width"] == 96
        assert sum(p.numel() for p in RestorationHead(**value).parameters()) < 500000
    smoke = protocol(parent, preflight=True)
    assert smoke["smoke"] and smoke["steps"] == 4 and len(smoke["seeds"]) == 1


@pytest.mark.parametrize(
    "options,parameter",
    [
        ({"phase_input": True}, "phase_projection.weight"),
        ({"decoder_normalization": True}, "decoder_norm.weight"),
    ],
)
def test_new_parameters_match_finite_difference(options, parameter):
    model = make(**options).double()
    activate(model)
    args = tuple(x.double() for x in inputs(model))
    p = dict(model.named_parameters())[parameter]
    index = (0, 0) if p.ndim == 2 else (0,)

    def score():
        return model(*args)[0][:, 0].mean()

    score().backward()
    analytical = p.grad[index].item()
    with torch.no_grad():
        value, epsilon = p[index].item(), 1e-5
        p[index] = value + epsilon
        plus = score().item()
        p[index] = value - epsilon
        minus = score().item()
        p[index] = value
    numerical = (plus - minus) / (2 * epsilon)
    assert abs(analytical - numerical) < max(1e-8, abs(numerical) * 0.001)
