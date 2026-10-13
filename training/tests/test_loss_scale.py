import pytest
import torch

from evaluation.loss_scale import gradient_comparison, spectral_parts, summarize_parts
from models.linear_objective import linear_objective


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.3])
@pytest.mark.parametrize("rate", [24000, 44100, 48000])
def test_decomposition_matches_production_loss_and_gradient(scale, rate):
    generator = torch.Generator().manual_seed(20261013)
    target = torch.rand((100, 129), generator=generator, dtype=torch.float64) * scale
    prediction = (
        target + torch.rand(target.shape, generator=generator, dtype=target.dtype) * 1e-7
    ).requires_grad_()
    parts = spectral_parts(prediction, target, rate)
    actual = sum(x["total"] for x in parts)
    expected = linear_objective(prediction[None], target[None], torch.tensor([[rate]]), burn=89)[
        "total"
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    a = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    b = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert torch.isfinite(a).all()
    assert len(summarize_parts(parts, prediction)) == 5


def test_exact_reference_has_zero_loss_and_gradient_including_silence():
    for value in (0.0, 1e-9, 0.1):
        target = torch.full((100, 129), value, dtype=torch.float64)
        prediction = target.clone().requires_grad_()
        loss = sum(x["total"] for x in spectral_parts(prediction, target, 48000))
        assert float(loss.detach()) == 0
        assert torch.count_nonzero(torch.autograd.grad(loss, prediction)[0]) == 0


def test_finite_difference_and_unsupported_band():
    target = torch.full((100, 129), 1e-8, dtype=torch.float64)
    prediction = (target * 2).requires_grad_()
    parts = spectral_parts(prediction, target, 24000)
    assert parts[-1]["bins"] == 0 and parts[-1]["total"] == 0
    gradient = torch.autograd.grad(sum(p["total"] for p in parts), prediction)[0]
    direction = torch.linspace(-1, 1, prediction.numel(), dtype=prediction.dtype).reshape_as(
        prediction
    )
    eps = 1e-12

    def score(p):
        return sum(x["total"] for x in spectral_parts(p, target, 24000))

    numerical = (
        score(prediction.detach() + eps * direction) - score(prediction.detach() - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close(numerical, (gradient * direction).sum(), rtol=1e-5, atol=1e-5)


def test_parameter_gradient_conflict_and_zero_reference():
    report = gradient_comparison([torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 0.0])])
    assert report["cosine"] == -1 and report["equal_pair_l2"] == 0
    assert gradient_comparison([torch.zeros(2)], [torch.ones(2)])["cosine"] is None


def test_full_dsp_audit_is_read_only_and_separates_two_risks():
    from models.envelope_probe import CONTEXT_BINS
    from models.fit_diagnostic import FixedFitObjective
    from models.rendered_probe import make_rendered_model
    from scripts.audit_loss_scale import inspect
    from tests.test_fit_diagnostic import paired_batch

    amplitude, batch = paired_batch()
    batch["cutoff"] = torch.full((2, amplitude.shape[1]), 10000.0, dtype=amplitude.dtype)
    model = make_rendered_model("rendered_balanced").double().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    context = torch.zeros(2, amplitude.shape[1], CONTEXT_BINS, dtype=amplitude.dtype)
    result = inspect(model, FixedFitObjective(batch), context)
    assert [r["role"] for r in result["roles"]] == ["paired", "identity"]
    assert result["roles"][1]["loss"] < 1e-8
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())


def test_stage_audit_uses_common_frame_support_and_matches_render():
    from models.causal_objective import render
    from models.rendered_objective import amplitude_features, reanalyze
    from scripts.audit_render_stages import stages
    from tests.test_fit_diagnostic import paired_batch

    amplitude, batch = paired_batch()
    features = amplitude_features(amplitude + 0.002, batch["x"])
    result, prelimit = stages(features, batch)
    expected, expected_prelimit = render(
        features,
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )
    assert len({v.shape for v in result.values()}) == 1
    torch.testing.assert_close(result["limited"], reanalyze(expected), atol=0, rtol=0)
    torch.testing.assert_close(prelimit, expected_prelimit, atol=0, rtol=0)
