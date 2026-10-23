"""Reference guards must allow recovery and reject genuine error growth.

SPDX-License-Identifier: Apache-2.0
"""

from types import SimpleNamespace

import pytest
import torch

from evaluation import reference_preservation as preservation
from evaluation import reference_training as pilot
from evaluation.anchored_training import pcm_guard as displacement_guard
from evaluation.fidelity_calibration import anchored_limits
from evaluation.spectral_priority_probe import GUARDS


def batch() -> dict:
    return {"identity": torch.tensor([False, True]), "rate": torch.tensor([[48000.0], [44100.0]])}


@pytest.mark.parametrize("bits", [16, 20, 24])
def test_beneficial_large_correction_passes_while_displacement_rule_rejects(bits: int) -> None:
    reference = torch.zeros(2, 512, dtype=torch.float64)
    reference[:, 128] = 0.2
    anchor = reference + 0.001
    result = preservation.reference_limits(reference, anchor, reference, bits)
    assert result["passed"].all()
    assert result["displacement"]["peak"].min() > 2.0**-bits
    assert not anchored_limits(reference, anchor, reference, half_step=2.0**-bits)[
        "within_budget"
    ].any()


def test_spike_smoothing_and_rolling_error_growth_are_not_allowed() -> None:
    q = 2.0**-20
    zero = torch.zeros(1, 512, dtype=torch.float64)
    spike = zero.clone()
    spike[:, 128] = 2 * q
    result = preservation.reference_limits(spike, zero, zero, 20)
    assert result["after"]["rms"].item() < q
    assert not result["passed"].any()
    target = spike * 1000
    assert preservation.reference_limits(target, target, target, 20)["passed"].all()
    assert not preservation.reference_limits(target * 0.5, target, target, 20)["passed"].any()
    assert preservation.reference_limits(zero + 0.75 * q, zero, zero, 20)["passed"].all()
    assert not preservation.reference_limits(zero + 1.5 * q, zero, zero, 20)["passed"].any()


def test_time_error_is_measured_against_reference_not_signal_variation() -> None:
    target = torch.zeros(1, 512, dtype=torch.float64)
    target[:, 128:] = 0.5
    exact = preservation.reference_limits(target, target, target, 24)
    assert exact["after"]["difference_rms"].item() == 0
    assert exact["after"]["boundary_difference_rms"].item() == 0
    assert exact["passed"].all()


@pytest.mark.parametrize("scale", [0.0, 1e-9, 0.01])
def test_mse_correct_reference_and_weak_gradients(scale: float) -> None:
    target = torch.zeros(2, 512, dtype=torch.float64)
    wave = torch.full_like(target, scale, requires_grad=True)
    losses = preservation.reference_mse(wave, target, batch())
    gradient = torch.autograd.grad(sum(losses.values()), wave)[0]
    assert torch.isfinite(gradient).all()
    if scale == 0:
        assert sum(losses.values()).item() == 0
        assert gradient.count_nonzero() == 0
    else:
        assert (gradient > 0).all()


def test_reference_mse_directional_finite_difference_and_target_detach() -> None:
    rng = torch.Generator().manual_seed(46)
    wave = torch.randn(2, 512, generator=rng, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 512, generator=rng, dtype=torch.float64, requires_grad=True)
    direction = torch.randn(2, 512, generator=rng, dtype=torch.float64)
    for key, loss in preservation.reference_mse(wave, target, batch()).items():
        grad, target_grad = torch.autograd.grad(
            loss, (wave, target), retain_graph=True, allow_unused=True
        )
        assert target_grad is None
        eps = 1e-5
        plus = preservation.reference_mse(wave + eps * direction, target, batch())[key]
        minus = preservation.reference_mse(wave - eps * direction, target, batch())[key]
        torch.testing.assert_close(
            (plus - minus) / (2 * eps), (grad * direction).sum(), atol=1e-9, rtol=1e-7
        )


def test_full_guard_allows_reference_recovery_with_column_rates(monkeypatch) -> None:
    target = torch.zeros(2, 512, dtype=torch.float64)
    data = {**batch(), "y": target}
    items = [(SimpleNamespace(batch=data), None)] * 5
    monkeypatch.setattr(preservation, "supervised_waves", lambda *_: (target, target))
    # P45's helper has its own binding; supply the same target there.
    monkeypatch.setattr(
        "evaluation.anchored_training.supervised_waves", lambda *_: (target, target)
    )
    anchors = [target + 0.001] * 5
    assert not preservation.pcm_guard([target] * 5, anchors, items, 24)["violations"]
    assert displacement_guard([target] * 5, anchors, items, 24)["violations"]


def test_absolute_constraint_changes_direction_without_replacing_primary_spectra() -> None:
    gradients = {k: torch.zeros(2) for k in GUARDS}
    gradients["paired_direct_amplitude"] = torch.tensor([1.0, 1.0])
    gradients["paired_mr"] = torch.tensor([1.0, 1.0])
    plain, _ = pilot.direction_for(gradients, None)
    direction, certificate = pilot.direction_for(
        gradients, {"identity": torch.tensor([-1.0, 0.0]), "paired_low4k": torch.zeros(2)}
    )
    assert plain[0] < -0.5
    torch.testing.assert_close(direction, torch.tensor([0.0, -1.0]), atol=1e-7, rtol=0)
    assert len(certificate["guard_names"]) == 8
    assert direction.dot(gradients["paired_mr"]) < 0


def test_nonfinite_pcm_is_not_a_passing_guard() -> None:
    zero = torch.zeros(1, 512)
    with pytest.raises(ValueError, match="finite"):
        preservation.reference_limits(zero + float("nan"), zero, zero, 16)
