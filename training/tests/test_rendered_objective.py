import pytest
import torch
from torch.nn import functional as F

from models.causal_objective import render
from models.rendered_objective import (
    RenderedObjective,
    amplitude_features,
    reanalyze,
    render_amplitude,
)
from scripts.audit_restoration_conflict import spatial_error_delta


def fixture(identity=True, level=0.1, dtype=torch.float64):
    time = torch.arange(12800, dtype=dtype) / 48000
    wave = (
        level * torch.sin(2 * torch.pi * 1100 * time)
        + level * 0.2 * torch.sin(2 * torch.pi * 17000 * time)
    )[None]
    padded = F.pad(wave, (128, 0))
    spectrum = torch.fft.rfft(padded.unfold(-1, 256, 128) * torch.hann_window(256, dtype=dtype))
    features = torch.stack((20 * spectrum.abs().clamp_min(1e-10).log10(), spectrum.angle()), dim=1)
    mask = torch.zeros(1, 1, 100, 129, dtype=dtype)
    mask[..., 65:] = 1
    batch = {
        "x": features,
        "mask": mask,
        "active": torch.ones(1, 100, dtype=torch.bool),
        "gate": torch.ones(1, 12800, dtype=dtype),
        "dry": padded[:, :12800],
        "clean_wave": padded[:, :12800],
        "rate": torch.tensor([[48000.0]], dtype=dtype),
        "identity": torch.tensor([identity]),
    }
    amplitude = torch.pow(10.0, features[:, 0] / 20)
    return amplitude, batch


def test_identity_preserves_existing_dsp_without_hiding_original_reference_error():
    p, batch = fixture(level=1.0)
    p.requires_grad_()
    result = RenderedObjective(7)(p, p.detach(), batch)
    assert result["total"] == 0 and result["dsp_preservation"] == 0
    assert result["rendered_waveform"] > 0.01  # limiter error remains explicitly reported
    result["total"].backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().max() == 0
    changed = (p.detach() + 0.002).requires_grad_()
    loss = RenderedObjective(7)(changed, p.detach(), batch)["total"]
    assert loss > 0
    loss.backward()
    assert torch.isfinite(changed.grad).all() and changed.grad.norm() > 0


def test_render_matches_existing_proxy_and_no_extra_delay():
    p, batch = fixture()
    p = p + 0.001
    actual = render_amplitude(p, batch)
    expected, _ = render(
        amplitude_features(p, batch["x"]),
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert reanalyze(actual).shape == (1, 98, 129)


def test_rendered_gradient_finite_difference_and_warmup():
    p, batch = fixture(identity=False)
    target = p.detach().clone()
    p = (p + 0.002).requires_grad_()
    fn = RenderedObjective(1)
    assert torch.autograd.gradcheck(
        lambda a: fn(a, target, batch)["total"], (p,), fast_mode=True, atol=2e-4, rtol=2e-3
    )
    baseline = fn(p, target, batch)
    changed = p.detach().clone()
    changed[:, :80] = 1e4
    for key, value in fn(changed, target, batch).items():
        torch.testing.assert_close(value, baseline[key], atol=0, rtol=0)


def test_identity_weight_is_expected_risk_not_batch_renormalization():
    p, batch = fixture()
    target = p.clone()
    p = p + 0.002
    a = RenderedObjective(1)(p, target, batch)["total"]
    b = RenderedObjective(7)(p, target, batch)["total"]
    torch.testing.assert_close(b, a * 4)
    batch["identity"][:] = False
    a = RenderedObjective(1)(p, target, batch)["total"]
    b = RenderedObjective(7)(p, target, batch)["total"]
    torch.testing.assert_close(b, a / 1.75)


@pytest.mark.parametrize("level", [0.0, 1e-9, 1e-5])
@pytest.mark.parametrize("identity", [False, True])
def test_silent_and_weak_rendered_gradients_remain_finite(level, identity):
    p, batch = fixture(level=level, identity=identity)
    target = p.clone()
    p = (p + 1e-10).requires_grad_()
    loss = RenderedObjective(7)(p, target, batch)["total"]
    loss.backward()
    assert torch.isfinite(loss) and loss >= 0
    assert torch.isfinite(p.grad).all()


def test_spatial_audit_partition_keeps_discarded_error():
    p = torch.full((1, 100, 129), 0.8)
    t = torch.ones_like(p)
    d = torch.full_like(p, 0.5)
    rate = torch.tensor([[48000.0]])
    a = spatial_error_delta(p, d, t, rate, torch.ones_like(p))
    b = spatial_error_delta(p, d, t, rate, torch.zeros_like(p))
    assert a["discarded_raw"] == 0 and b["applied_raw"] == 0 and b["blended"] == 0
    assert a["applied_raw"] == b["discarded_raw"] < 0
    assert a["blended"] == a["applied_raw"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_cpu_fft_bridge_preserves_mps_gradient():
    p, batch = fixture(identity=False, dtype=torch.float32)
    target = p.clone()
    a = (p + 0.002).requires_grad_()
    b = a.detach().to("mps").requires_grad_()
    fn = RenderedObjective(7)
    x = fn(a, target, batch)["total"]
    y = fn(b, target.to("mps"), {k: v.to("mps") for k, v in batch.items()})["total"]
    x.backward()
    y.backward()
    torch.testing.assert_close(x, y.cpu(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-5, rtol=1e-5)
