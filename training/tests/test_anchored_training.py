"""P45 fixed-anchor guards and rejected-update rollback, not quality tests.

SPDX-License-Identifier: Apache-2.0
"""

from types import SimpleNamespace

import torch

from evaluation import anchored_training as pilot


def test_metric_budget_is_relative_to_fixed_start_not_previous_step() -> None:
    keys = (
        "paired_direct_amplitude",
        "paired_mr",
        "paired_direct_complex",
        "paired_mr_dynamics",
        "budget",
    )
    anchor = {role: dict.fromkeys(keys, 1.0) for role in ("fit", "guard")}
    base = {role: dict(v) for role, v in anchor.items()}
    current = {role: dict(v) for role, v in anchor.items()}
    current["fit"].update(paired_direct_amplitude=0.9, paired_mr=0.9)
    assert not pilot.metric_guard(base, current, anchor)["violations"]
    base["guard"]["paired_direct_complex"] = 1.0009
    current["guard"]["paired_direct_complex"] = 1.0018
    assert pilot.metric_guard(base, current, anchor)["violations"]
    assert not pilot.metric_guard(anchor, anchor, anchor)["spectral_progress"]


def test_pcm_guard_uses_per_example_peaks_and_all_batches(monkeypatch) -> None:
    q = 2.0**-20
    zero = torch.zeros(2, 512, dtype=torch.float64)
    batch = {
        "y": zero,
        "rate": torch.tensor([[48000.0], [48000.0]]),
        "identity": torch.tensor([False, True]),
    }
    items = [(SimpleNamespace(batch=batch), None)] * 5
    monkeypatch.setattr(pilot, "supervised_waves", lambda *_: (zero, zero))
    waves = [zero.clone() for _ in range(5)]
    waves[4][1, 0] = 2 * q
    result = pilot.pcm_guard(waves, [zero] * 5, items, 20)
    assert len(result["violations"]) == 1
    assert result["violations"][0]["batch"] == 4
    assert result["violations"][0]["displacement_rms"] < q
    assert result["violations"][0]["displacement_peak"] > q
    # A constant paired low-band change must also be protected.
    waves[0][0] = 2 * q
    assert len(pilot.pcm_guard(waves, [zero] * 5, items, 20)["violations"]) == 2


def test_low_band_accepts_column_rates_without_cross_batch_broadcast() -> None:
    wave = torch.randn(5, 512, dtype=torch.float64)
    flat = torch.full((5,), 48000.0)
    projected = pilot.low_band(wave, flat[:, None])
    assert projected.shape == wave.shape
    torch.testing.assert_close(projected, pilot.low_band(wave, flat))


def test_noop_and_exception_restore_exact_parameters(monkeypatch) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    original = model.weight.detach().clone()
    value = {
        role: {
            "paired_direct_amplitude": 1.0,
            "paired_mr": 1.0,
            "paired_direct_complex": 1.0,
            "paired_mr_dynamics": 1.0,
            "budget": 1.0,
        }
        for role in ("fit", "guard")
    }
    observation = (value, [torch.zeros(1, 512)] * 5)
    monkeypatch.setattr(pilot, "observations", lambda *_: observation)
    monkeypatch.setattr(pilot, "gradients_for", lambda *_: ({"total": torch.ones(2)}, None, None))
    monkeypatch.setattr(
        pilot, "spectral_directions", lambda *_: ({"pooled_spectrum_guarded": torch.ones(2)}, {})
    )
    monkeypatch.setattr(pilot, "pcm_guard", lambda *_: {"violations": []})
    result = pilot.attempt(model, [None] * 5, observation, 16, None)
    assert not result["accepted"] and result["selected_radius"] is None
    assert len(result["points"]) == 2
    assert torch.equal(model.weight, original)

    def fail(*_):
        raise RuntimeError("measurement failure")

    monkeypatch.setattr(pilot, "pcm_guard", fail)
    import pytest

    with pytest.raises(RuntimeError, match="measurement failure"):
        pilot.attempt(model, [None] * 5, observation, 16, None)
    assert torch.equal(model.weight, original)
