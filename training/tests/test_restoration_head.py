"""P20 identity, streaming clocks, reference semantics and learned phase gradients."""

import pytest
import torch

from models.restoration_head import ARMS, RestorationHead, shared_state
from models.spectral_consistency import consistency_loss, reference_features


def inputs(frames=12, batch=2):
    torch.manual_seed(53)
    x = torch.randn(batch, 2, frames, 129)
    x[:, 0] = x[:, 0] * 12 - 40
    context = torch.rand(batch, frames, 1538)
    condition = torch.ones(batch, frames, 3)
    condition[..., 1] = 0.6
    return x, context, condition


@pytest.mark.parametrize("arm", ARMS)
def test_identity_and_silent_finite_backward(arm):
    model = RestorationHead(arm, 8)
    x, context, condition = inputs()
    x[0, 0], x[0, 1], context[0] = -200, 0, 0
    out, state = model(x, context, condition, model.initial_state(2))
    torch.testing.assert_close(out, x, atol=2e-5, rtol=0)
    (out.square().mean() + state.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shared_backbone_and_real_residual_are_matched():
    models = []
    for arm in ARMS:
        torch.manual_seed(17)
        models.append(RestorationHead(arm, 8))
    for model in models[1:]:
        for key, tensor in shared_state(models[0]).items():
            torch.testing.assert_close(tensor, shared_state(model)[key], atol=0, rtol=0)
    amplitude, complex_model = models[1:]
    with torch.no_grad():
        amplitude.backbone.decoder[-1].weight.normal_(0, 0.001)
        complex_model.backbone.decoder[-1].weight[:129].copy_(amplitude.backbone.decoder[-1].weight)
    x, context, condition = inputs()
    a, _ = amplitude(x, context, condition, amplitude.initial_state(2))
    b, _ = complex_model(x, context, condition, complex_model.initial_state(2))
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)
    # A negative real coordinate carries a pi flip only in the complex head.
    torch.testing.assert_close(
        (a[:, 1] - b[:, 1]).sin(), torch.zeros_like(a[:, 1]), atol=1e-6, rtol=0
    )


def test_nonzero_complex_head_streaming_future_and_channel_isolation():
    model = RestorationHead("complex", 8)
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.01)
    x, context, condition = inputs()
    whole, end = model(x, context, condition, model.initial_state(2))
    state, parts = model.initial_state(2), []
    for t in range(x.shape[2]):
        out, state = model(
            x[:, :, t : t + 1], context[:, t : t + 1], condition[:, t : t + 1], state
        )
        parts.append(out)
    torch.testing.assert_close(torch.cat(parts, 2), whole, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state, end, atol=2e-5, rtol=2e-5)
    changed = x.clone()
    changed[:, :, 6:] += 5
    future, _ = model(changed, context, condition, model.initial_state(2))
    torch.testing.assert_close(future[:, :, :6], whole[:, :, :6], atol=0, rtol=0)
    single, _ = model(x[:1], context[:1], condition[:1], model.initial_state(1))
    torch.testing.assert_close(single, whole[:1], atol=2e-5, rtol=2e-5)
    assert (whole[:, 1, :, 1:-1] - x[:, 1, :, 1:-1]).abs().max() > 0.001
    torch.testing.assert_close(whole[:, 1, :, [0, -1]], x[:, 1, :, [0, -1]], atol=0, rtol=0)


def test_phase_head_learns_and_gradient_matches_finite_difference():
    model = RestorationHead("complex", 8).double()
    x, c, q = [v.double() for v in inputs(frames=2, batch=1)]
    state = model.initial_state(1).double()
    output, _ = model(x, c, q, state)
    loss = (output[:, 1, :, 50] - x[:, 1, :, 50] - 0.1).square().mean()
    loss.backward()
    head = model.backbone.decoder[-1].bias
    analytic = head.grad[179].item()
    values = []
    with torch.no_grad():
        for delta in [-1e-6, 1e-6]:
            head[179] = delta
            out, _ = model(x, c, q, state)
            values.append(float((out[:, 1, :, 50] - x[:, 1, :, 50] - 0.1).square().mean()))
        head[179] = 0
    assert abs(analytic) > 1e-5
    assert analytic == pytest.approx((values[1] - values[0]) / 2e-6, rel=1e-5, abs=1e-7)


def test_clean_complex_reference_is_zero_risk_and_identity_uses_input():
    x, _, _ = inputs(frames=128)
    y = x.clone()
    y[:, 1] += 0.4
    b = {
        "x": x,
        "y": y,
        "identity": torch.tensor([False, True]),
        "mask": torch.ones(2, 1, 128, 129),
        "active": torch.ones(2, 128),
        "rate": torch.full((2, 1), 48000.0),
    }
    target = reference_features(b, phase_reference="clean")
    torch.testing.assert_close(target[1, 1], x[1, 1], atol=0, rtol=0)
    torch.testing.assert_close(target[0, 1, 80:], y[0, 1, 80:], atol=0, rtol=0)
    assert consistency_loss(target, b, phase_reference="clean").item() == 0
    assert consistency_loss(target, b).item() > 1e-4


@pytest.mark.parametrize("arm", ("anchored", "amplitude", "complex"))
def test_nonzero_head_onnx_streaming_parity(arm, tmp_path):
    from evaluation.head_export import export_graph, verify

    torch.manual_seed(17)
    model = RestorationHead(arm, width=8).eval()
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.02)
        model.backbone.decoder[-1].bias.normal_(0, 0.02)
    session = export_graph(model, tmp_path / "head.onnx")
    result = verify(model, session)
    assert result["sequential_hops"] == 320


@pytest.mark.parametrize("arm", ("amplitude", "complex"))
def test_absent_high_band_can_receive_substantial_residual(arm):
    # A local-only scale would limit an entirely removed band to its tiny floor.
    model = RestorationHead(arm, width=8).eval()
    original = torch.zeros(1, 2, 1, 129)
    original[:, 0, :, :64] = -6
    original[:, 0, :, 64:] = -200
    with torch.no_grad():
        model.backbone.decoder[-1].bias[:129].fill_(0.1)
        output, _ = model(
            original, torch.zeros(1, 1, 1538), torch.ones(1, 1, 3), model.initial_state(1)
        )
    magnitude = torch.pow(10.0, original[:, 0] / 20)
    global_rms = magnitude.square().mean(-1, keepdim=True).sqrt()
    restored = torch.pow(10.0, output[:, 0] / 20)
    assert (restored[..., 80:] > global_rms * 0.15).all()
