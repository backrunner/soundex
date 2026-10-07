"""Latency budgets must use total serial cost and raw maxima, including startup."""

import pytest

from evaluation.performance_latency import check_serial_latency


def _case(maximum_us: float) -> dict:
    return {
        "first_frame_us": 1000.0,
        "serial_latency_p99_ms": 3.9,
        "serial_latency_max_ms": 2.9 + max(maximum_us, 1000.0) / 1000.0,
        "latency_target_met": 2.9 + max(maximum_us, 1000.0) / 1000.0 < 8.0,
    }


def test_good_p99_cannot_hide_a_redline_outlier() -> None:
    with pytest.raises(ValueError, match="10 ms redline"):
        check_serial_latency(
            _case(7100.0), 2.9, {"p99_us": 1000.0, "max_us": 7100.0}, label="outlier"
        )


def test_eight_ms_target_is_distinct_from_ten_ms_redline() -> None:
    case = _case(6000.0)
    check_serial_latency(case, 2.9, {"p99_us": 1000.0, "max_us": 6000.0}, label="case")
    assert not case["latency_target_met"]
    case["latency_target_met"] = True
    with pytest.raises(ValueError, match="8 ms target"):
        check_serial_latency(case, 2.9, {"p99_us": 1000.0, "max_us": 6000.0}, label="case")


def test_first_frame_is_included_in_maximum_budget() -> None:
    case = _case(1000.0)
    case.update(first_frame_us=7100.0, serial_latency_max_ms=10.0, latency_target_met=False)
    with pytest.raises(ValueError, match="10 ms redline"):
        check_serial_latency(case, 2.9, {"p99_us": 1000.0, "max_us": 1000.0}, label="startup")


def test_budget_above_historical_five_ms_still_meets_current_target() -> None:
    case = _case(3000.0)
    check_serial_latency(case, 2.9, {"p99_us": 1000.0, "max_us": 3000.0}, label="current")
    assert case["latency_target_met"]


def test_exact_eight_ms_does_not_meet_strict_target() -> None:
    case = _case(5100.0)
    check_serial_latency(case, 2.9, {"p99_us": 1000.0, "max_us": 5100.0}, label="boundary")
    assert not case["latency_target_met"]
