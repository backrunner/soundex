import numpy as np
import pytest
import torch

from data.causal_study import SAMPLES
from data.subset_study import make_draws, nested_specs, shifted_specs
from models.fit_diagnostic import FixedFitObjective
from models.rendered_objective import amplitude_features
from scripts.run_subset_study import aggregate
from tests.test_fit_diagnostic import paired_batch


def source_fixture():
    specs, rows = [], {}
    for i in range(40):
        key = f"{i:03}"
        specs.append({"row_id": key, "start": 40000, "gain_db": 0})
        rows[key] = {
            "track_id": key,
            "split": "train",
            "held_out": False,
            "length_samples": 150000,
            "corpus": "music_library" if i < 32 else "speech_library",
            "source_metadata": {"genre": f"genre-{i % 4}"},
        }
    return specs, rows


def test_nested_pools_preserve_ratio_and_prefix():
    specs, rows = source_fixture()
    pools = nested_specs(specs, rows)
    for size, pool in pools.items():
        assert len(pool) == size
        assert sum(rows[s["row_id"]]["corpus"] == "speech_library" for s in pool) == size // 5
    assert (
        {s["row_id"] for s in pools[5]}
        < {s["row_id"] for s in pools[10]}
        < {s["row_id"] for s in pools[40]}
    )
    rows["000"]["held_out"] = True
    with pytest.raises(ValueError, match="training-only"):
        nested_specs(specs, rows)


@pytest.mark.parametrize("size", [5, 10, 40])
def test_equal_exposure_and_fixed_batch_mix(size):
    draws = make_draws(size, 256 * size // 5, 20261012)
    assert draws == make_draws(size, 256 * size // 5, 20261012)
    assert np.array_equal(np.bincount(np.array(draws).flatten()), np.full(size, 256))
    assert all(len(set(d)) == 5 and sum(i >= size * 4 // 5 for i in d) == 1 for d in draws)


def test_shifted_scoring_windows_are_disjoint_even_with_context_overlap():
    specs, rows = source_fixture()
    specs = specs[:1]
    specs[0]["start"] = 30530
    rows["000"]["length_samples"] = SAMPLES + 61061
    shifted = shifted_specs(specs, rows)
    distance = abs(shifted[0]["start"] - specs[0]["start"])
    assert SAMPLES - 128 - 89 * 128 <= distance < SAMPLES
    rows["000"]["length_samples"] = SAMPLES
    specs[0]["start"] = 0
    with pytest.raises(ValueError, match="scoring window"):
        shifted_specs(specs, rows)


def test_cached_objective_slice_matches_recomputation_and_gradient():
    amplitude, batch = paired_batch()
    batch = {k: torch.cat((v, v), 0) for k, v in batch.items()}
    full = FixedFitObjective(batch)
    selected = full.select([2, 3])
    p = (amplitude + 0.002).requires_grad_()
    a = selected(amplitude_features(p, selected.batch["x"]))["total"]
    b = FixedFitObjective(selected.batch)(amplitude_features(p, selected.batch["x"]))["total"]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    ga = torch.autograd.grad(a, p, retain_graph=True)[0]
    gb = torch.autograd.grad(b, p)[0]
    torch.testing.assert_close(ga, gb, atol=0, rtol=0)


def test_common_corpus_weight_does_not_follow_development_row_count():
    rows = [
        {
            "corpus": "music_library",
            "metrics": {"total": 1, "paired_restoration": 2, "identity_preservation": 0},
        }
    ]
    rows += [
        {
            "corpus": "speech_library",
            "metrics": {"total": 0, "paired_restoration": 0, "identity_preservation": 0},
        }
    ] * 10
    assert aggregate(rows)["weighted"]["total"] == 0.8
    assert aggregate([]) is None
