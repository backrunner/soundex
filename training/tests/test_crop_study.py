import numpy as np
import pytest

from data.causal_study import SAMPLES
from data.crop_study import SUPERVISION_START, crop_bank, crop_draws, protected_probes
from data.subset_study import make_draws, shifted_specs
from tests.test_subset_study import source_fixture


def test_crop_bank_protects_full_probe_input_including_short_recordings():
    specs, rows = source_fixture()
    specs[0]["start"] = 30530
    rows["000"]["length_samples"] = SAMPLES + 61061
    reserved = protected_probes(specs, rows)
    assert reserved[0]["start"] == 0
    bank, audit = crop_bank(specs, reserved, rows, 8)
    assert len(bank) == 320
    for i, anchor in enumerate(specs):
        entries = bank[i * 8 : (i + 1) * 8]
        assert entries[0] == anchor
        assert len({s["start"] for s in entries}) == 8
        assert not any(audit[i]["supervision_overlaps_reserved_input"])
        begin, end = audit[i]["reserved_input"]
        for spec in entries:
            assert spec["row_id"] == anchor["row_id"]
            assert spec["gain_db"] == anchor["gain_db"]
            assert 0 <= spec["start"] <= rows[spec["row_id"]]["length_samples"] - SAMPLES
            assert spec["start"] + SAMPLES <= begin or spec["start"] + SUPERVISION_START >= end
    assert audit[0]["input_context_overlaps_reserved_input"][0]
    assert crop_bank(specs, reserved, rows, 1)[0] == specs


def test_bank_rejects_supervised_leak_and_mismatched_recordings():
    specs, rows = source_fixture()
    with pytest.raises(ValueError, match="supervision overlaps"):
        crop_bank(specs, specs, rows, 8)
    reserved = shifted_specs(specs, rows)
    reserved[0]["row_id"] = "001"
    with pytest.raises(ValueError, match="recording mismatch"):
        crop_bank(specs, reserved, rows, 8)


def test_sampling_keeps_recording_order_and_exact_balanced_exposures():
    draws = make_draws(40, 2048, 20261012)
    fixed = crop_draws(draws, 1, 20261012, 40)
    sampled = crop_draws(draws, 8, 20261012, 40)
    assert fixed == draws
    assert np.array_equal(np.array(sampled) // 8, draws)
    assert np.array_equal(np.bincount(np.array(sampled).ravel()), np.full(320, 32))
    assert np.array_equal(np.bincount(np.array(sampled[:256]).ravel()), np.full(320, 4))
    assert sampled == crop_draws(draws, 8, 20261012, 40)
    assert sampled != crop_draws(draws, 8, 20261013, 40)
