"""P52 calibrated units, reference gradients, causal support and resumable training."""

import copy
import shutil
from types import SimpleNamespace

import pytest
import torch

from evaluation.adapter_continuation import update as control_update
from evaluation.physical_preservation import ARMS, band_errors, calibrate, penalties, update
from evaluation.reporting import sha256_file, write_json_report
from models.spectral_adapter import from_base
from scripts.run_adapter_pilot import optimizer_for


@pytest.mark.parametrize("size", [1024, 1025])
def test_parseval_dc_nyquist_and_physical_frequency(size):
    torch.manual_seed(52)
    x = torch.randn(2, size, dtype=torch.float64)
    target = torch.randn_like(x)
    rates = torch.tensor([[44100], [48000]])
    bands = band_errors(x, target, rates)
    torch.testing.assert_close(bands.sum(-1), (x - target).square().mean(-1))
    torch.testing.assert_close(
        band_errors(torch.ones_like(x), torch.zeros_like(x), rates)[:, 0], torch.ones(2).double()
    )
    t = torch.arange(size, dtype=torch.float64)
    # An exact-bin high-frequency sinusoid: RMS squared = amplitude squared / 2.
    tone = (0.2 * torch.sin(2 * torch.pi * 300 * t / size)).repeat(2, 1)
    result = band_errors(tone, torch.zeros_like(tone), rates)
    torch.testing.assert_close(result[:, 3], torch.full((2,), 0.02, dtype=torch.float64))
    assert result[:, [0, 1, 2, 4]].max() < 1e-26


def test_reference_zero_silence_weak_gradient_detachment_and_finite_difference():
    torch.manual_seed(53)
    target = torch.randn(2, 1024, dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([44100, 48000])
    exact = target.detach().clone().requires_grad_()
    loss = band_errors(exact, target, rates)[:, 1:].sum()
    gx, gy = torch.autograd.grad(loss, (exact, target), allow_unused=True)
    assert loss == 0 and torch.count_nonzero(gx) == 0 and gy is None
    for level in (0.0, 1e-10, 0.1):
        wave = (torch.randn_like(target) * level).requires_grad_()
        clean = torch.zeros_like(target)
        value = band_errors(wave, clean, rates)[:, 1:].sum()
        g = torch.autograd.grad(value, wave)[0]
        assert torch.isfinite(g).all()
        if not level:
            assert value == 0 and torch.count_nonzero(g) == 0
        else:
            direction = torch.randn_like(wave)
            direction /= direction.norm()
            eps = level * 1e-3
            a = band_errors(wave + eps * direction, clean, rates)[:, 1:].sum()
            b = band_errors(wave - eps * direction, clean, rates)[:, 1:].sum()
            torch.testing.assert_close(
                (a - b) / (2 * eps), (g * direction).sum(), atol=1e-25, rtol=1e-6
            )


def model_item():
    from test_adapter_continuation import real_item

    initial, item = real_item()
    model = from_base(initial, "high").eval()
    with torch.no_grad():
        model.adapter.weight.normal_(0, 1e-4)
    return initial, model, item


def test_full_model_derivative_supervision_crop_hooks_and_reference_target():
    _, model, item = model_item()
    model = model.double()
    batch = {k: v.double() if v.is_floating_point() else v for k, v in item[0].batch.items()}
    item = (SimpleNamespace(batch=batch), item[1].double())
    terms, _ = penalties(model, item)
    params = [model.adapter.weight, model.adapter.bias]
    saved = [p.detach().clone() for p in params]
    directions = [torch.randn_like(p) for p in params]
    norm = sum(d.square().sum() for d in directions).sqrt()
    directions = [d / norm for d in directions]
    for key in terms:
        gs = torch.autograd.grad(terms[key], params, retain_graph=True)
        analytic = sum((g * d).sum() for g, d in zip(gs, directions, strict=True))
        values = []
        with torch.no_grad():
            for sign in (-1, 1):
                for p, s, d in zip(params, saved, directions, strict=True):
                    p.copy_(s + sign * 1e-4 * d)
                values.append(penalties(model, item)[0][key])
            for p, s in zip(params, saved, strict=True):
                p.copy_(s)
        torch.testing.assert_close((values[1] - values[0]) / 2e-4, analytic, atol=1e-17, rtol=1e-4)
    assert not model.adapter._forward_hooks
    original = penalties(model, item)[0]
    # Unsupplied target regions cannot alter the new objective.
    changed = copy.deepcopy(batch)
    changed["clean_wave"][:, : 89 * 128] += 1
    changed["clean_wave"][:, -128:] += 1
    after = penalties(model, (SimpleNamespace(batch=changed), item[1]))[0]
    assert all(torch.equal(original[k], after[k]) for k in original)
    # Original clean PCM, not model/adapter output, is the physical target.
    changed["clean_wave"][:, 89 * 128 + 8] += 0.1
    after = penalties(model, (SimpleNamespace(batch=changed), item[1]))[0]
    assert after["physical"] > original["physical"] and after["logit"] == original["logit"]


def test_fixed_calibration_and_control_exactly_matches_existing_update():
    initial, model, item = model_item()
    old = copy.deepcopy(model)
    scales = calibrate(model, [item] * 4)
    assert all(
        scales["weights"][k] * scales["gradient_norms"][k]
        == pytest.approx(0.1 * scales["gradient_norms"]["common"])
        for k in scales["weights"]
    )
    result = update(model, [item] * 4, optimizer_for(model), "control", scales)
    control = control_update(old, [item] * 4, optimizer_for(old))
    assert result["fit_before"] == control["fit_before"]
    assert result["fit_after"] == control["fit_after"]
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in model.state_dict().items())
    from evaluation.adapter_pilot import assert_frozen

    assert_frozen(model, initial)


def test_real_training_resume_frozen_barrier_and_review_receipts(tmp_path, monkeypatch):
    from scripts import run_physical_preservation as runner
    from scripts.run_adapter_continuation import raw_review

    initial, model, item = model_item()
    base = tmp_path / "base.pth"
    torch.save(initial, base)
    parent = tmp_path / "parent.pth"
    torch.save(
        {
            "format": "soundex-spectral-adapter-v1",
            "model": model.state_dict(),
            "base_config": model.base_config,
            "support": "high",
        },
        parent,
    )
    base_entry = {"path": str(base), "sha256": sha256_file(base)}
    parent_entry = {"path": str(parent), "sha256": sha256_file(parent)}
    sample = {"row": {"corpus": "music_library"}, "spec": {"row_id": "fixture"}}
    config = {
        "starts": {"seed": base_entry},
        "parents": {"seed": parent_entry},
        "steps": 2,
        "calibration_indices": [0],
        "prior": str(tmp_path / "p51"),
        "p50": str(tmp_path / "p50"),
        "cases": {
            role: [{"index": i, "samples": {"calibration": [sample]}} for i in range(8)]
            for role in ("fit", "review")
        },
    }
    schedule = {"fit": [[0]] * 8, "review": [[0]]}
    pool = SimpleNamespace(select=lambda _: item)
    output = tmp_path / "current"
    scales = runner.calibration(config, pool, schedule, output)
    assert runner.calibration(config, pool, schedule, output) == scales
    whole = runner.train("seed", "physical", config, scales, pool, schedule, output)
    actual, calls = runner.update, 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interruption")
        return actual(*args)

    monkeypatch.setattr(runner, "update", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.train("seed", "physical", config, scales, pool, schedule, tmp_path / "resumed")
    monkeypatch.setattr(runner, "update", actual)
    resumed = runner.train("seed", "physical", config, scales, pool, schedule, tmp_path / "resumed")
    a, b = (torch.load(p, weights_only=False) for p in (whole, resumed))
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    for k, state in a["optimizer"]["state"].items():
        assert all(torch.equal(v, b["optimizer"]["state"][k][f]) for f, v in state.items())
    entries = {}
    for arm in ARMS:
        path = runner.train("seed", arm, config, scales, pool, schedule, output)
        entries["seed-" + arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "seed": "seed",
            "arm": arm,
        }
    parents = {"seed": base_entry, "seed-persistent": parent_entry, "seed-reset": parent_entry}
    raw_review(config, parents, pool, schedule, tmp_path / "p50")
    for role in ("fit", "review"):
        dest = tmp_path / "p51/review" / role
        dest.mkdir(parents=True)
        for arm in ("separated", "scalar"):
            shutil.copy2(
                tmp_path / "p50/review" / role / "seed-persistent.json",
                dest / f"seed-persistent-{arm}.json",
            )
    write_json_report(entries, output / "review-registration.json")
    runner.review(config, entries, pool, schedule, output)
    assert len(list((output / "review").rglob("*.receipt.json"))) == 6
    monkeypatch.setattr(
        runner, "review_case", lambda *_: pytest.fail("completed receipt must be reused")
    )
    runner.review(config, entries, pool, schedule, output)
    write_json_report({}, output / "review-registration.json")
    with pytest.raises(ValueError, match="barrier"):
        runner.review(config, entries, pool, schedule, output)
