"""Source-level weighting and population matching for the P73 audit."""

from copy import deepcopy

import pytest

from scripts.audit_capacity_tradeoff import source_stages


def fixture():
    panels = {}
    for panel, deltas in (("raw_native_population", (1, 3, 10)), ("native", (2, 4, 8))):
        panels[panel] = {
            "cases": [
                {
                    "case": i,
                    "role": 0,
                    "source": "a" if i < 2 else "b",
                    "source_id": "source-a" if i < 2 else "source-b",
                    "codec": "codec",
                    "group": "music/paired",
                    "before": {"mr": 20.0},
                    "after": {"mr": 20.0 + delta},
                    "delta": {"mr": float(delta)},
                }
                for i, delta in enumerate(deltas)
            ]
        }
    return {"source_review": {"seed": panels}}


def test_stage_differences_use_equal_recordings_and_matching_roles():
    result = source_stages(fixture())["seed"]
    group = result["groups"]["music/paired/all"]
    assert group["sources"] == 2
    assert group["source_mean"]["raw_delta"] == 6.0
    assert group["source_mean"]["native_delta"] == 5.5
    assert group["source_mean"]["native_minus_raw_delta"] == -0.5
    assert len(result["records"]) == 3


@pytest.mark.parametrize("field", ["source", "codec", "group", "role"])
def test_stage_comparison_rejects_population_or_metadata_drift(field):
    report = deepcopy(fixture())
    report["source_review"]["seed"]["native"]["cases"][0][field] = "changed"
    with pytest.raises(ValueError, match="unmatched source review"):
        source_stages(report)
