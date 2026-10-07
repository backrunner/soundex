"""Strict finite numeric validation for performance evidence."""

from __future__ import annotations

import math
from typing import Any


def _required_int(value: dict[str, Any], field: str, *, label: str, minimum: int) -> int:
    actual = value.get(field)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < minimum:
        raise ValueError(f"{label}: integer field {field!r} is invalid or missing")
    return actual


def _required_float(
    value: dict[str, Any],
    field: str,
    *,
    label: str,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    actual = value.get(field)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label}: numeric field {field!r} is invalid or missing")
    number = float(actual)
    if not math.isfinite(number):
        raise ValueError(f"{label}: numeric field {field!r} is not finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label}: numeric field {field!r} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}: numeric field {field!r} is below its minimum")
    return number


def _assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
    absolute_tolerance: float = 1e-9,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=absolute_tolerance):
        raise ValueError(f"{label} does not match raw evidence")


def _percentile(sorted_values: list[int], percentile: float) -> int:
    rank = math.ceil(percentile * len(sorted_values))
    return sorted_values[max(0, min(rank - 1, len(sorted_values) - 1))]
