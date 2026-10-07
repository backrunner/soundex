"""Historical timing reports must not silently inherit a different target."""

import pytest

from evaluation.performance import _check_report


def test_previous_schema_is_not_reinterpreted_with_new_budget() -> None:
    with pytest.raises(ValueError, match="schema"):
        _check_report(
            {"schema_version": 3, "report_type": "soundex-performance"},
            artifact_sha256="a" * 64,
            artifact_size=1_000_000,
            fft_size=256,
            hop_size=128,
            label="historical",
        )
