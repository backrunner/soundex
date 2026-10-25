import copy

import pytest

from evaluation.joint_attribution import BANDS, COMPONENTS, compare, index_records


def row(source, name, value):
    scores = {f"band{low}_{k}": value for low in BANDS for k in COMPONENTS}
    scores.update(mr=value * sum(COMPONENTS.values()), dynamics=value, full_rms=value)
    return {
        "source": source,
        "id": name,
        "rate": 48000,
        "codec": "test",
        "role": "paired",
        "score_start": 0,
        "score_length": 48000,
        "channels": 2,
        "path": "raw/test",
        "scores": scores,
    }


def test_source_weighting_and_exact_decomposition():
    left = {str(i): row("a" if i < 3 else "b", str(i), 2 if i < 3 else 4) for i in range(4)}
    right = {str(i): row(r["source"], str(i), 1) for i, r in left.items()}
    result = compare(left, right)
    assert result["source_mean_delta"]["full_rms"] == 2  # not clip-weighted 1.5
    assert result["decomposition"]["reconstructed_mr_delta"] == pytest.approx(3.2)
    assert result["cases"] == 4 and result["sources"] == 2
    assert len(result["per_case"]) == 4
    assert result["mr_worse_sources"] == ["a", "b"]


@pytest.mark.parametrize("field,value", [("source", "other"), ("rate", 44100), ("channels", 1)])
def test_mismatched_pair_rejected(field, value):
    a = row("a", "1", 1)
    b = copy.deepcopy(a)
    b[field] = value
    with pytest.raises(ValueError, match="metadata"):
        compare({"1": a}, {"1": b})


def test_missing_duplicate_and_corrupt_scores_rejected():
    a = row("a", "1", 1)
    with pytest.raises(ValueError, match="coverage"):
        compare({"1": a}, {})
    with pytest.raises(ValueError, match="duplicate"):
        index_records([a, a])
    b = copy.deepcopy(a)
    b["scores"]["mr"] = 4
    with pytest.raises(ValueError, match="reproduce"):
        compare({"1": a}, {"1": b})


def test_endpoint_roundoff_survives_near_zero_difference():
    a, b = row("a", "1", 1), row("a", "1", 1)
    a["scores"]["mr"] += 1.3e-7
    b["scores"]["mr"] -= 1.3e-7
    result = compare({"1": a}, {"1": b})
    assert result["decomposition"]["reconstructed_mr_delta"] == 0
    assert result["decomposition"]["residual"] == pytest.approx(2.6e-7)
