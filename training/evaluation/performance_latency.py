"""Recompute the conservative processing budget, separate from device latency."""

from typing import Any

from evaluation.performance_numbers import _assert_close, _required_float


def check_serial_latency(
    case: dict[str, Any], algorithmic_ms: float, latency: dict[str, float], *, label: str
) -> None:
    """Require a 10ms redline and verify the optional 8ms target from raw samples."""
    first_us = _required_float(case, "first_frame_us", label=label, positive=True)
    p99_ms = algorithmic_ms + latency["p99_us"] / 1000.0
    maximum_ms = algorithmic_ms + max(latency["max_us"], first_us) / 1000.0
    for name, expected in (
        ("serial_latency_p99_ms", p99_ms),
        ("serial_latency_max_ms", maximum_ms),
    ):
        _assert_close(
            _required_float(case, name, label=label, positive=True),
            expected,
            label=f"{label}: {name}",
        )
    if maximum_ms >= 10.0:
        raise ValueError(f"{label}: serial processing latency reaches the 10 ms redline")
    if case.get("latency_target_met") is not (maximum_ms < 8.0):
        raise ValueError(f"{label}: 8 ms target does not match raw evidence")
