"""Known PCM rulers, cumulative drift and per-example fidelity accounting.

SPDX-License-Identifier: Apache-2.0
"""

import pytest
import torch

from evaluation.fidelity_calibration import anchored_limits, lattice_round, pcm_statistics
from scripts.calibrate_pcm_fidelity import select_calibration


@pytest.mark.parametrize("bits", [16, 20, 24])
def test_precision_ruler_bound_silence_and_no_clipping(bits: int) -> None:
    wave = torch.linspace(-1.1, 1.1, 10001, dtype=torch.float64)
    rounded = lattice_round(wave, bits)
    assert (rounded - wave).abs().max() <= 2.0**-bits
    assert rounded[-1] > 1
    assert lattice_round(torch.zeros(2), bits).count_nonzero() == 0


def test_band_partition_and_hop_boundary_known_answer() -> None:
    n = 48000
    t = torch.arange(n, dtype=torch.float64) / n
    wave = (0.1 * torch.sin(2 * torch.pi * 1000 * t) + 0.02 * torch.cos(2 * torch.pi * 12000 * t))[
        None
    ]
    result = pcm_statistics(wave, torch.tensor([48000.0]))
    assert float(result["low8k_rms"]) == pytest.approx(0.1 / 2**0.5, abs=1e-12)
    assert float(result["hf8k_rms"]) == pytest.approx(0.02 / 2**0.5, abs=1e-12)
    torch.testing.assert_close(
        result["full_rms"].square(), result["low8k_rms"].square() + result["hf8k_rms"].square()
    )
    jumps = torch.zeros(1, 512)
    jumps[:, 128:] = 1
    result = pcm_statistics(jumps, torch.tensor([48000.0]))
    assert float(result["hop_boundary_difference_rms"]) == pytest.approx(1 / 3**0.5)


def test_fixed_anchor_prevents_rolling_budget_and_cancellation() -> None:
    q = 2.0**-20
    anchor = torch.zeros(2, 256, dtype=torch.float64)
    ref = anchor.clone()
    first, second = anchor + 0.75 * q, anchor + 1.5 * q
    assert anchored_limits(first, anchor, ref, half_step=q)["within_budget"].all()
    assert not anchored_limits(second, anchor, ref, half_step=q)["within_budget"].any()
    # Equal error norms can hide a complete sign reversal of the error waveform.
    anchor += 10 * q
    result = anchored_limits(-anchor, anchor, ref, half_step=q)
    assert not result["within_budget"].any()
    assert result["error_rms_increase"].count_nonzero() == 0
    torch.testing.assert_close(
        result["displacement_rms"], torch.full((2,), 20 * q, dtype=torch.float64)
    )


def test_bad_values_are_not_silently_accepted() -> None:
    zero = torch.zeros(1, 512)
    with pytest.raises(ValueError, match="finite"):
        anchored_limits(zero + float("nan"), zero, zero, half_step=2.0**-16)
    with pytest.raises(ValueError, match="finite"):
        pcm_statistics(zero + float("inf"), torch.tensor([48000.0]))


def test_calibration_excludes_source_time_across_views_and_keeps_new_draws_disjoint() -> None:
    rows = {
        name: {
            "source_checksum": "common",
            "split": "train",
            "held_out": False,
            "length_samples": 200000,
            "sample_rate": 48000,
            "start_sample": 0,
        }
        for name in ("left", "right")
    }
    specs = [{"row_id": "right", "start": start} for start in (0, 1000, 33000, 66000)]
    schedule = {"specs": specs, "draws": [[i] for i in range(4)]}
    protected = [{"row_id": "left", "start": 0}]
    selected = select_calibration(schedule, rows, protected, count=2)
    assert selected == [2, 3]
    with pytest.raises(ValueError, match="insufficient"):
        select_calibration(schedule, rows, protected, count=3)
