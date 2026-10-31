"""Original-reference, silence and derivative gates for P67 fused spectral loss."""

import pytest
import torch

from models.complex_synthesis import complex_features, synthesize
from models.fused_spectrum import fused_spectrum_loss, fused_stages, prepare_fusion
from train import spectral_features


def batch_for(scale):
    torch.manual_seed(67)
    pcm = torch.randn(2, 160 * 128, dtype=torch.float64) * scale
    features = spectral_features(torch.nn.functional.pad(pcm, (128, 0)), 256, 128)
    return {
        "x": features,
        "y": features.clone(),
        "clean_wave": synthesize(complex_features(features)).detach(),
        "dry": torch.nn.functional.pad(pcm, (128, 0))[:, : pcm.shape[-1]],
        "active": torch.ones(2, 160, dtype=torch.bool),
        "mask": torch.full((2, 1, 160, 129), 0.5, dtype=torch.float64),
        "rate": torch.tensor([[44100], [48000]]),
    }


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.01])
def test_correct_reference_silence_weak_gradients_and_warmup(scale):
    batch = batch_for(scale)
    plan = prepare_fusion(batch)
    assert not plan["weights"][:, :, :80].any()
    prediction = batch["x"].clone().requires_grad_()
    result = fused_spectrum_loss(prediction, batch, plan)["per_example"].mean()
    gradient = torch.autograd.grad(result, prediction)[0]
    assert abs(float(result.detach())) < 1e-9
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() < 1e-7
    changed = prediction.detach().clone()
    changed[:, 0, 100:, 60:] += 1
    changed[:, 1, 100:, 60:] += 0.1
    changed.requires_grad_()
    result = fused_spectrum_loss(changed, batch, plan)["per_example"].mean()
    result.backward()
    assert torch.isfinite(changed.grad).all()
    assert not changed.grad[:, :, :88].any()
    assert changed.grad[:, :, 100:, 60:].abs().sum() > 0


def test_phase_and_magnitude_finite_differences():
    batch = batch_for(0.01)
    plan = prepare_fusion(batch)
    prediction = batch["x"].clone()
    prediction[:, 0, 100:, 60:] += 0.5
    prediction[:, 1, 100:, 60:] += 0.1
    prediction.requires_grad_()
    loss = fused_spectrum_loss(prediction, batch, plan)["per_example"].mean()
    gradient = torch.autograd.grad(loss, prediction)[0]
    for axis in (0, 1):
        index = (0, axis, 110, 80)
        plus, minus = prediction.detach().clone(), prediction.detach().clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numeric = (
            fused_spectrum_loss(plus, batch, plan)["per_example"].mean()
            - fused_spectrum_loss(minus, batch, plan)["per_example"].mean()
        ) / 2e-5
        torch.testing.assert_close(gradient[index], numeric, rtol=1e-4, atol=1e-8)


def test_retained_bins_are_exact_and_limiter_never_changes_reference():
    batch = batch_for(0.6)
    original = batch["clean_wave"].clone()
    batch["mask"][..., :60] = 0
    prediction = batch["x"].clone().requires_grad_()
    stages = fused_stages(prediction, batch["x"], prepare_fusion(batch))
    assert torch.equal(stages["features"][..., :60], batch["x"][..., :60])
    assert stages["final"].abs().max() <= 0.95
    assert original.abs().max() > 0.95
    loss = fused_spectrum_loss(prediction, batch)["per_example"].mean()
    assert loss > 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(original, batch["clean_wave"])


def test_common_objective_matches_historical_path_and_fused_auxiliary_reaches_head():
    from types import SimpleNamespace

    from evaluation.spectral_training import objective_value
    from models.band_phase import BandPhaseCorrection
    from models.band_tcn import make_prototype
    from models.fused_spectrum import phase_objective
    from models.restoration_head import predict_budget

    batch = batch_for(0.01)
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0, dtype=torch.float64)
    context = torch.zeros(2, 160, 1538, dtype=torch.float64)
    model = BandPhaseCorrection(make_prototype("band", seed=67), phase_support="native").double()
    with torch.no_grad():
        model.phase_head[-1].bias.fill_(0.2)
    prediction, _ = predict_budget(model, batch, context)
    original = objective_value(model, (SimpleNamespace(batch=batch), context), "common")
    common = phase_objective(prediction, batch, include_fused=False)
    assert torch.equal(original, common)
    augmented = phase_objective(prediction, batch, include_fused=True)
    auxiliary = augmented - common
    assert auxiliary > 0
    auxiliary.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.phase_head.parameters()
    )
    assert model.phase_head[-1].weight.grad.abs().sum() > 0
