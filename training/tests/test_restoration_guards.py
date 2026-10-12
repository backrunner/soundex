import copy

import pytest

from evaluation.restoration_guards import guard


def rows():
    return {
        "rows": [
            {
                "row_id": str(i),
                "track_id": str(i),
                "corpus": "music_library",
                "metrics": {
                    "total": 10.0,
                    "weak_band_count": 1 if i < 2 else 0,
                    "weak_reconstruction": 1.0,
                },
            }
            for i in range(5)
        ]
    }


def test_mean_gain_cannot_hide_weak_group_regression():
    control = rows()
    candidate = copy.deepcopy(control)
    for row in candidate["rows"]:
        row["metrics"]["total"] = 1.0
        row["metrics"]["weak_reconstruction"] = 2.0 if row["metrics"]["weak_band_count"] else 0.0
    assert guard(candidate, control, "music_library", "total")["status"] == "supported"
    assert (
        guard(candidate, control, "music_library", "weak_reconstruction", weak=True)["status"]
        == "regression"
    )


def test_unchanged_output_is_not_evidence_of_restoration():
    baseline = rows()
    assert (
        guard(baseline, baseline, "music_library", "total", improvement=True)["status"]
        != "supported"
    )
    assert guard(baseline, baseline, "music_library", "total")["status"] == "supported"


def test_guard_requires_pairs_and_reference_fixed_groups():
    baseline = rows()
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["metrics"]["weak_band_count"] = 0
    with pytest.raises(ValueError, match="weak group changed"):
        guard(changed, baseline, "music_library", "weak_reconstruction", weak=True)
    changed["rows"].pop()
    with pytest.raises(ValueError, match="unpaired"):
        guard(changed, baseline, "music_library", "total")
    assert guard({"rows": []}, {"rows": []}, "music_library", "total")["status"] == "unsupported"
