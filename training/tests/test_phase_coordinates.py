"""P31 numerical/causal/export acceptance; no restoration quality claims."""

import pytest
import torch

from evaluation.head_export import export_graph, verify
from models.complex_synthesis import complex_features
from models.restoration_head import RestorationHead, load_restoration_head
from scripts.run_phase_coordinates_study import ARMS, protocol


def make(phase_input=False, coordinates="global_cartesian"):
    torch.manual_seed(31)
    return RestorationHead(
        "complex",
        width=8,
        generation_budget=True,
        budget_scope="residual",
        residual_slope=1 / 64,
        phase_input=phase_input,
        residual_coordinates=coordinates,
    )


def inputs(model, frames=8, level=-40):
    torch.manual_seed(32)
    x = torch.randn(2, 2, frames, 129)
    x[:, 0] = level
    x[:, 1, :, 0] = 0
    x[:, 1, :, -1] = torch.pi
    return x, torch.rand(2, frames, 1538) * 0.01, torch.ones(2, frames, 3), model.initial_state(2)


def activate(model):
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.02)
        if model.phase_input:
            model.phase_projection.weight.normal_(0, 0.02)


@pytest.mark.parametrize("phase_input", [False, True])
def test_matched_initialization_and_checkpoint_identity(phase_input):
    relative = make(False, "input_relative")
    model = make(phase_input)
    for key, value in relative.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    for level in (-200, -140, -40):
        args = inputs(model, level=level)
        torch.testing.assert_close(model(*args)[0], args[0], atol=0, rtol=0)
    activate(model)
    restored = load_restoration_head(
        {
            "arm": "complex",
            "model_config": dict(
                arm="complex",
                width=8,
                generation_budget=True,
                budget_scope="residual",
                residual_slope=1 / 64,
                phase_input=phase_input,
                residual_coordinates="global_cartesian",
            ),
            "model": model.state_dict(),
        }
    )
    torch.testing.assert_close(restored(*args)[0], model(*args)[0], atol=0, rtol=0)


def test_global_residual_direction_is_independent_of_observed_phase():
    model = make().double()
    with torch.no_grad():
        model.backbone.decoder[-1].bias[60] = 0.6
        model.backbone.decoder[-1].bias[189] = -0.4
    args = tuple(a.double() for a in inputs(model))
    x, context, condition, state = args
    changed = x.clone()
    changed[:, 1, :, 1:-1] += 1.1

    def residual(a):
        return complex_features(model(a, context, condition, state)[0]) - complex_features(a)

    a, b = residual(x), residual(changed)
    torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-10)
    assert a[..., 60].real.abs().min() > 1e-5
    assert a[..., 60].imag.abs().min() > 1e-5
    assert a[..., [0, 128]].imag.abs().max() < 1e-14


@pytest.mark.parametrize("phase_input", [False, True])
@pytest.mark.parametrize("level", [-200, -140, -40])
def test_finite_gradients_with_silence_weak_bins_and_phase_wrap(phase_input, level):
    model = make(phase_input).double()
    activate(model)
    args = tuple(a.double() for a in inputs(model, level=level))
    args[0][:, 1, :, 50:60] = torch.pi - 1e-8
    output, _ = model(*args)
    (output[:, 0].square().mean() + output[:, 1].sin().square().mean()).backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.backbone.decoder[-1].weight.grad.norm() > 0
    if phase_input and level != -200:
        assert model.phase_projection.weight.grad.norm() > 0


@pytest.mark.parametrize(
    "parameter,index", [("backbone.decoder.2.bias", (189,)), ("phase_projection.weight", (0, 60))]
)
def test_global_coordinate_gradients_match_finite_difference(parameter, index):
    model = make(True).double()
    activate(model)
    args = tuple(a.double() for a in inputs(model))
    p = dict(model.named_parameters())[parameter]

    def score():
        z = complex_features(model(*args)[0])
        return (z.real - 0.01).square().mean() + (z.imag + 0.02).square().mean()

    score().backward()
    analytical = p.grad[index].item()
    with torch.no_grad():
        original, epsilon = p[index].item(), 1e-5
        p[index] = original + epsilon
        plus = score().item()
        p[index] = original - epsilon
        minus = score().item()
        p[index] = original
    assert analytical == pytest.approx((plus - minus) / (2 * epsilon), rel=0.002, abs=1e-11)


def test_history_causality_chunk_equivalence_channel_isolation_and_reset():
    model = make(True).eval()
    activate(model)
    x, context, condition, state = inputs(model, frames=90)
    with torch.no_grad():
        full, final = model(x, context, condition, state)
        outputs, current = [], state
        for t in range(90):
            out, current = model(
                x[:, :, t : t + 1], context[:, t : t + 1], condition[:, t : t + 1], current
            )
            outputs.append(out)
        torch.testing.assert_close(torch.cat(outputs, 2), full, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(current, final, atol=2e-5, rtol=1e-5)
        other = x.clone()
        other[:, 1, 50:] += 0.9
        changed, _ = model(other, context, condition, state)
        torch.testing.assert_close(changed[:, :, :50], full[:, :, :50], atol=0, rtol=0)
        mono, _ = model(x[1:], context[1:], condition[1:], state[1:])
        torch.testing.assert_close(mono, full[1:], atol=2e-5, rtol=1e-5)
        reset, _ = model(x, context, condition, model.initial_state(2))
        torch.testing.assert_close(reset, full, atol=0, rtol=0)


@pytest.mark.parametrize("phase_input", [False, True])
def test_nonzero_global_coordinate_onnx(phase_input, tmp_path):
    model = make(phase_input).eval()
    activate(model)
    result = verify(model, export_graph(model, tmp_path / "model.onnx"))
    assert result["sequential_hops"] == 320 and result["reset"] and result["silence"]


def test_factorial_protocol_has_equal_objective_budget_and_fixed_endpoints():
    parent = dict(
        specs=[],
        weight=0.13,
        rate=1e-4,
        direct_loss={"version": 1},
        optimization="fixed",
        seeds=[20261012, 20261013],
    )
    config = protocol(parent)
    assert config["steps"] == config["primary_endpoint"] == 512
    assert len(config["seeds"]) == 2 and set(config["optimization_arms"].values()) == {"raw"}
    assert config["direct_loss"] == parent["direct_loss"]
    assert (
        len(
            {
                (v["phase_input"], v["residual_coordinates"])
                for v in config["model_configs"].values()
            }
        )
        == 4
    )
    assert all(config["model_configs"][arm]["width"] == 96 for arm in ARMS)
    assert protocol(parent, preflight=True)["steps"] == 4
    with pytest.raises(ValueError, match="coordinates"):
        make(coordinates="invalid")
