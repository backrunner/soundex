"""Role isolation, spectral progress and calibrated PCM diagnostic checks.

SPDX-License-Identifier: Apache-2.0
"""

import copy

import pytest
import torch

from evaluation.batch_update_probe import (
    choose_step,
    error_rms,
    select_roles,
    step_policy,
)
from evaluation.spectral_priority_probe import spectral_directions
from scripts.summarize_pooled_updates import changes


def values() -> dict:
    metrics = {
        k: 1.0
        for k in (
            "total",
            "paired_direct_amplitude",
            "paired_direct_complex",
            "paired_mr",
            "paired_mr_dynamics",
            "identity_mr",
            "identity_risk",
            "budget",
        )
    }
    return {role: dict(metrics) for role in ("fit", "guard", "transfer")}


def test_noop_is_not_progress_and_transfer_never_selects() -> None:
    base = values()
    assert not step_policy(base, copy.deepcopy(base), 0)["accepted"]
    current = copy.deepcopy(base)
    for key in ("paired_direct_amplitude", "paired_mr"):
        current["fit"][key] -= 0.01
    current["transfer"] = {"arbitrary": float("nan")}
    assert step_policy(base, current, 0)["accepted"]
    current["guard"]["identity_mr"] += 0.0005
    assert not step_policy(base, current, 0)["accepted"]
    assert step_policy(base, current, 0.001)["accepted"]
    current["guard"]["identity_mr"] += 0.001
    assert not step_policy(base, current, 0.001)["accepted"]


def test_first_passing_radius_and_noop_are_distinct() -> None:
    points = [{"policies": {"strict": {"accepted": v}}} for v in (False, True, True)]
    assert choose_step(points, "strict") == 1
    assert choose_step(points[:1], "strict") is None


def test_report_retains_noops_without_counting_them_as_improvement() -> None:
    base = values()["fit"]
    improved = {k: v * 0.99 for k, v in base.items()}
    result = changes([base, base], [improved, base])
    assert result["count"] == 2
    assert result["all_five_improved"] == 1
    assert result["metrics"]["identity_mr"]["unchanged"] == 1
    assert changes([], [])["count"] == 0


def test_spectrum_priority_does_not_equalize_an_opposed_complex_gradient() -> None:
    gradients = {k: torch.zeros(2) for k in values()["fit"]}
    for key in ("paired_direct_amplitude", "paired_mr"):
        gradients[key] = torch.tensor([1.0, 0.0])
    gradients["paired_direct_complex"] = torch.tensor([-1.0, 0.0])
    directions, report = spectral_directions(gradients)
    torch.testing.assert_close(directions["pooled_spectrum_guarded"], torch.tensor([-1.0, 0.0]))
    assert (
        report["pooled_normalized_derivatives"]["pooled_spectrum_guarded"]["paired_direct_complex"]
        > 0
    )
    gradients["identity_mr"] = torch.tensor([-1.0, 0.0])
    directions, report = spectral_directions(gradients)
    assert report["projection"]["degenerate"]
    assert torch.equal(directions["pooled_spectrum_guarded"], torch.zeros(2))


@pytest.mark.parametrize("length", [48000, 48001])
def test_pcm_parseval_rms_and_band_calibration(length: int) -> None:
    t = torch.arange(length, dtype=torch.float64) / length
    error = (0.1 * torch.sin(2 * torch.pi * 1000 * t) + 0.02 * torch.sin(2 * torch.pi * 12000 * t))[
        None
    ]
    rate = torch.tensor([float(length)])
    torch.testing.assert_close(error_rms(error, rate), error.square().mean(-1).sqrt())
    assert float(error_rms(error, rate, 8000)) == pytest.approx(0.02 / 2**0.5, abs=1e-12)
    assert float(error_rms(torch.zeros_like(error), rate)) == 0
    assert float(error_rms(torch.ones_like(error), rate)) == pytest.approx(1.0)


def test_nyquist_is_not_doubled() -> None:
    wave = torch.tensor([[1.0, -1.0] * 128])
    assert float(error_rms(wave, torch.tensor([48000.0]), 8000)) == pytest.approx(1.0)


def test_roles_are_source_disjoint_and_skip_prior_indices() -> None:
    rows = {
        str(i): {"track_id": str(i), "source_checksum": str(i), "split": "train", "held_out": False}
        for i in range(12)
    }
    schedule = {"specs": [{"row_id": str(i)} for i in range(12)], "draws": [[i] for i in range(12)]}
    rows["2"]["source_checksum"] = "0"
    selected = select_roles(schedule, 0, rows, {4})
    assert len(selected) == len(set(selected)) == 6
    assert 2 not in selected and 4 not in selected
    assert len({rows[str(i)]["source_checksum"] for i in selected}) == 6
    for row in rows.values():
        row["source_checksum"] = "same"
    with pytest.raises(ValueError, match="insufficient"):
        select_roles(schedule, 0, rows, set())
