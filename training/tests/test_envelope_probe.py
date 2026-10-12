import pytest
import torch

from models.envelope_objective import amplitude_metrics, envelope_objective
from models.envelope_probe import EnvelopeProbe, StreamingContext, context_features


@pytest.mark.parametrize("arm", ["short", "tiled", "past"])
def test_frontend_streaming_causality_channel_reset_polarity(arm):
    torch.manual_seed(4)
    pcm = torch.randn(2, 128 * 81)
    features = context_features(pcm, arm)
    stream = StreamingContext(2, arm)
    actual = torch.cat([stream.step(hop) for hop in pcm.split(128, dim=-1)], dim=1)
    torch.testing.assert_close(actual, features, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(context_features(-pcm, arm), features, atol=0, rtol=0)
    changed = pcm.clone()
    changed[:, 128 * 43 :] = torch.randn_like(changed[:, 128 * 43 :])
    torch.testing.assert_close(context_features(changed, arm)[:, :43], features[:, :43])
    torch.testing.assert_close(context_features(pcm[:1], arm), features[:1])
    reset = StreamingContext(2, arm)
    torch.testing.assert_close(reset.step(pcm[:, :128]), features[:, :1])
    torch.testing.assert_close(
        context_features(torch.zeros_like(pcm), arm), torch.zeros_like(features)
    )


def test_past_contains_information_tiling_cannot_see():
    pcm = torch.zeros(1, 128 * 24)
    pcm[0, 1000] = 1
    assert context_features(pcm, "past")[0, 16].sum() > 0
    assert context_features(pcm, "tiled")[0, 16].sum() == 0


def test_normalized_fft_amplitude():
    pcm = torch.cos(torch.arange(128 * 40) * (2 * torch.pi / 32))[None]
    values = context_features(pcm, "past")[0, 32]
    torch.testing.assert_close(values[32], torch.tensor(64.0), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(values[513 + 64], torch.tensor(64.0), atol=1e-4, rtol=1e-5)


def test_model_step_parity_identity_capacity_and_missing_bins():
    torch.manual_seed(8)
    model = EnvelopeProbe(width=16).eval()
    x = torch.rand(2, 70, 129)
    c = torch.rand(2, 70, 1538)
    condition = torch.rand(2, 70, 3)
    state = model.initial_state(2)
    with torch.no_grad():
        full, final = model(x, c, condition, state)
        chunks = []
        for start, stop in ((0, 1), (1, 7), (7, 44), (44, 70)):
            out, state = model(x[:, start:stop], c[:, start:stop], condition[:, start:stop], state)
            chunks.append(out)
        torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(state, final, atol=2e-6, rtol=2e-6)
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
        identity, _ = model(x, c, condition, state)
        torch.testing.assert_close(identity, x, rtol=0, atol=0)
        model.decoder[-1].bias.fill_(0.1)
        x[:, :, 100:] = 0
        filled, _ = model(x, c, condition, state)
        assert (filled[:, :, 100:] > 0).all()


@pytest.mark.parametrize("rate", [44100.0, 48000.0])
@pytest.mark.parametrize("level", [0.0, 1e-8, 1.0])
def test_objective_identity_silence_gradients(rate, level):
    target = torch.rand(1, 8, 129, dtype=torch.float64) * level
    prediction = target.clone().requires_grad_()
    rates = torch.tensor([[rate]], dtype=torch.float64)
    loss = envelope_objective(prediction, target, rates, burn=0)["total"]
    assert float(loss.detach()) == 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    prediction = (target + 1e-6).requires_grad_()
    loss = envelope_objective(prediction, target, rates, burn=0)["total"]
    loss.backward()
    assert loss > 0 and torch.isfinite(prediction.grad).all()
    metrics = amplitude_metrics(target, target, rates, burn=0)
    assert metrics["high_sc"] == metrics["high_flux"] == 0


def test_loss_finite_difference():
    torch.manual_seed(17)
    p = (torch.rand(1, 3, 129, dtype=torch.float64) + 0.2).requires_grad_()
    t = torch.rand_like(p)
    rate = torch.tensor([[48000.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: envelope_objective(value, t, rate, burn=0)["total"],
        (p,),
        eps=1e-6,
        atol=1e-5,
        fast_mode=True,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_loss_gradient_parity():
    torch.manual_seed(23)
    target = torch.rand(2, 70, 129)
    rate = torch.tensor([[44100.0], [48000.0]])
    cpu = (torch.rand_like(target) + 0.01).requires_grad_()
    mps = cpu.detach().to("mps").requires_grad_()
    a = envelope_objective(cpu, target, rate)["total"]
    b = envelope_objective(mps, target.to("mps"), rate.to("mps"))["total"]
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cpu.grad, mps.grad.cpu(), atol=2e-6, rtol=2e-5)
