import pytest
import torch

from models.budget_objective import balanced_risk, balanced_spectrum, budget_risk, budget_target
from models.fit_diagnostic import FixedFitObjective
from models.restoration_head import RestorationHead, predict_budget, shared_state
from models.spectral_consistency import consistency_loss, reference_features
from scripts.run_budget_study import ARMS, ENDPOINTS, STEPS, protocol
from tests.test_fit_diagnostic import paired_batch
from tests.test_restoration_head import inputs


@pytest.mark.parametrize("level", [0.0, 1e-10, 0.1])
def test_balanced_exact_reference_zero_and_weak_finite_difference(level):
    torch.manual_seed(91)
    target = torch.rand(2, 6, 129, dtype=torch.double) * level
    rate = torch.tensor([[44100.0], [48000.0]], dtype=torch.double)
    p = target.clone().requires_grad_()
    zero = balanced_spectrum(p, target, rate, burn=0).sum()
    assert zero == 0
    zero.backward()
    assert p.grad.abs().max() == 0
    changed = (target + 1e-7).requires_grad_()
    loss = balanced_spectrum(changed, target, rate, burn=0).sum()
    grad = torch.autograd.grad(loss, changed)[0]
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    direction = torch.randn_like(changed)
    direction /= direction.norm()
    eps = 1e-9
    measured = (
        balanced_spectrum(changed + eps * direction, target, rate, burn=0).sum()
        - balanced_spectrum(changed - eps * direction, target, rate, burn=0).sum()
    ) / (2 * eps)
    torch.testing.assert_close(measured, (grad * direction).sum(), atol=2e-5, rtol=2e-4)


def test_balanced_tracks_reference_transients_and_does_not_drop_weak_band():
    target = torch.zeros(1, 6, 129, dtype=torch.double)
    target[:, :, :30] = 1
    target[:, 3, 70:] = 0.1
    rate = torch.tensor([[48000.0]])
    assert balanced_spectrum(target, target, rate, 0) == 0
    smoothed = target.mean(1, keepdim=True).expand_as(target)
    assert balanced_spectrum(smoothed, target, rate, 0) > 0
    changed = target.clone()
    changed[..., 100:] += 1e-5
    assert balanced_spectrum(changed, target, rate, 0) > 0


@pytest.mark.parametrize("level", [0.0, 1e-9, 0.1])
def test_rendered_balanced_gradient_and_supervised_region(level):
    _, batch = paired_batch(level)
    objective = FixedFitObjective(batch)
    prediction = batch["x"].clone().requires_grad_()
    risk = balanced_risk(objective, prediction)
    risk.backward()
    assert torch.isfinite(risk) and torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :, :80].abs().max() == 0
    reference = reference_features(batch, phase_reference="clean")
    assert (
        consistency_loss(reference, batch, phase_reference="clean", normalization="global_band")
        == 0
    )


def test_budget_identity_shared_initialization_and_missing_band_capacity():
    torch.manual_seed(17)
    control = RestorationHead("complex", 8, residual_slope=1 / 64)
    torch.manual_seed(17)
    model = RestorationHead("complex", 8, residual_slope=1 / 64, generation_budget=True)
    for key, tensor in shared_state(control).items():
        torch.testing.assert_close(tensor, shared_state(model)[key], atol=0, rtol=0)
    x, c, q = inputs()
    a, _ = control(x, c, q, control.initial_state(2))
    b, _ = model(x, c, q, model.initial_state(2))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    weights = model.budget_weights(q)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
    x[:, 0, :, :64], x[:, 0, :, 64:] = -6, -200
    with torch.no_grad():
        model.backbone.decoder[-1].bias[:129].fill_(6.4)
        model.budget_head.bias.fill_(4)
        out, _ = model(x, c, q, model.initial_state(2))
    mag = torch.pow(10, out[:, 0]) ** 0.05
    assert (mag[..., 80:] > 0.1).all()


def test_budget_target_identity_phase_and_no_warmup_supervision():
    _, batch = paired_batch()
    batch["y"][0, 1] += 0.4
    target = budget_target(batch)
    assert target[1].abs().max() == 0 and target[0].max() > 0
    budget = torch.ones_like(target, requires_grad=True)
    loss = budget_risk(budget, batch)
    loss.backward()
    assert budget.grad[:, :9].abs().max() == 0
    assert torch.isfinite(budget.grad).all() and budget.grad[:, 9:].abs().max() > 0
    altered = {k: v.clone() for k, v in batch.items()}
    altered["y"][:, :, :80] += 50
    torch.testing.assert_close(budget_target(altered), target, atol=0, rtol=0)


def test_budget_head_receives_supervision_and_uses_no_target_in_prediction():
    _, batch = paired_batch()
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    batch["cutoff"] = torch.full((2, 100), 8000.0)
    context = torch.rand(2, 100, 1538)
    model = RestorationHead("complex", 8, generation_budget=True)
    prediction, budget = predict_budget(model, batch, context)
    budget_risk(budget, batch).backward()
    assert model.budget_head.weight.grad.norm() > 0
    batch["y"] = batch["y"] + 10
    other, _ = predict_budget(model, batch, context)
    torch.testing.assert_close(prediction, other, atol=0, rtol=0)


def test_balanced_render_finite_difference():
    _, batch = paired_batch()
    objective = FixedFitObjective(batch)
    p = (batch["x"] + 0.002).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: balanced_risk(objective, x), (p,), fast_mode=True, atol=2e-4, rtol=2e-3
    )


def test_budget_auxiliary_finite_difference():
    _, batch = paired_batch()
    batch["y"][0, 0, 90:, 80:] += 3
    budget = torch.ones_like(budget_target(batch), requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: budget_risk(x, batch), (budget,), fast_mode=True)


def test_global_consistency_finite_difference():
    _, batch = paired_batch()
    prediction = (batch["x"] + 0.002).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: consistency_loss(x, batch, phase_reference="clean", normalization="global_band"),
        (prediction,),
        fast_mode=True,
        atol=2e-4,
        rtol=2e-3,
    )


def test_joint_summary_keeps_recording_weights_and_corpora_separate():
    from evaluation.joint_review import summarize

    rows = []
    for track, corpus, error in [
        ("a", "music", 1),
        ("a", "music", 3),
        ("b", "music", 8),
        ("c", "speech", 20),
    ]:
        values = {name: {"error": {"rms": error}} for name in ("clean", "identity_dsp")}
        rows.append({"track_id": track, "corpus": corpus, "roles": {"paired": values}})
    report = summarize(rows)
    assert report["music/paired"]["clean"]["clip_mean_error_rms"] == 4
    assert report["music/paired"]["clean"]["recording_mean_error_rms"] == 5
    assert report["speech/paired"]["clean"]["clip_mean_error_rms"] == 20


def test_protocol_fixes_full_factorial_budget_and_common_data():
    parent = {k: [] for k in ("bank", "common", "development", "shifted")}
    parent.update(seeds=[20261012, 20261013], rate=1e-4, weight=0.1365237430457469, tolerance=1e-6)
    config = protocol(parent)
    assert STEPS == 16192 and ENDPOINTS[-1] == STEPS
    assert len(ARMS) == 4 and len(config["seeds"]) == 2
    assert {c["width"] for c in config["model_configs"].values()} == {640}
    assert config["balanced_arms"] == ["balanced", "joint"]
    assert set(config["budget_weights"]) == {"budget", "joint"}
