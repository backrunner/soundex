"""Checkpoint schema, compatibility, and reproducibility tests."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from checkpoint import CheckpointSchemaError, load_checkpoint, validate_checkpoint
from checkpoint_state import capture_rng_state, restore_rng_state


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_checkpoint_round_trip_preserves_every_field(checkpoint_factory) -> None:
    path, expected = checkpoint_factory()

    actual = load_checkpoint(path)

    _assert_nested_equal(expected, actual)


def test_legacy_2048_frame_contract_remains_loadable(
    checkpoint_factory,
    resolved_config: dict[str, Any],
) -> None:
    legacy_config = copy.deepcopy(resolved_config)
    legacy_config["audio"]["fft_size"] = 2048
    legacy_config["audio"]["hop_size"] = 512
    path, _ = checkpoint_factory("legacy-frame.pth", config=legacy_config)

    checkpoint = load_checkpoint(path, expected_config=legacy_config)

    assert checkpoint["feature_contract"]["input_shape"] == ["batch", 2, 1, 1025]
    assert checkpoint["feature_contract"]["hop_size"] == 512


def test_unsupported_frame_contract_is_rejected(
    checkpoint_factory,
    resolved_config: dict[str, Any],
) -> None:
    unsupported = copy.deepcopy(resolved_config)
    unsupported["audio"]["fft_size"] = 512

    with pytest.raises(CheckpointSchemaError, match="FFT/hop pair is unsupported"):
        checkpoint_factory("unsupported-frame.pth", config=unsupported)


def test_schema_zero_and_unknown_major_are_rejected(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.pth"
    torch.save({"generator": {}}, legacy_path)
    with pytest.raises(CheckpointSchemaError, match="schema-0"):
        load_checkpoint(legacy_path)

    unknown_path = tmp_path / "unknown.pth"
    torch.save({"schema": {"name": "soundex-checkpoint", "major": 99}}, unknown_path)
    with pytest.raises(CheckpointSchemaError, match="unsupported checkpoint schema major"):
        load_checkpoint(unknown_path)


@pytest.mark.parametrize("minor", [0, 1])
def test_checkpoint_schema_before_1_2_is_rejected(checkpoint_factory, minor: int) -> None:
    _, checkpoint = checkpoint_factory()
    checkpoint["schema"]["minor"] = minor

    with pytest.raises(CheckpointSchemaError, match=r"schema 1\.2 or newer"):
        validate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("audio", "fft_size", 4096, "fft_size"),
        ("audio", "sample_rate", 48000, "sample_rate"),
        ("training", "context_frames", 2, "context_frames"),
        ("training", "batch_size", 8, "batch_size"),
        ("training", "learning_rate", 1.0e-4, "learning_rate"),
        ("training", "warmup_steps", 500, "warmup_steps"),
        ("data.sampling", "strategy", "concat_subsample", "strategy"),
        ("model.generator", "expand_ratio", 3, "expand_ratio"),
    ],
)
def test_expected_config_mismatches_are_rejected(
    checkpoint_factory,
    resolved_config: dict[str, Any],
    section: str,
    key: str,
    value: Any,
    match: str,
) -> None:
    path, _ = checkpoint_factory()
    expected_config = copy.deepcopy(resolved_config)
    target = expected_config
    for component in section.split("."):
        target = target[component]
    target[key] = value

    with pytest.raises(CheckpointSchemaError, match=match):
        load_checkpoint(path, expected_config=expected_config)


def test_architecture_major_mismatch_is_rejected(checkpoint_factory) -> None:
    _, checkpoint = checkpoint_factory()
    checkpoint["model"]["architecture"]["major"] = 2

    with pytest.raises(CheckpointSchemaError, match="architecture major"):
        validate_checkpoint(checkpoint)


def test_internal_feature_contract_mismatch_is_rejected(checkpoint_factory) -> None:
    _, checkpoint = checkpoint_factory()
    checkpoint["feature_contract"]["fft_size"] = 2048

    with pytest.raises(CheckpointSchemaError, match=r"feature_contract\.fft_size"):
        validate_checkpoint(checkpoint)


def test_rng_and_sampler_state_restore_exactly() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    sampler = WeightedRandomSampler(
        torch.ones(8),
        num_samples=8,
        replacement=True,
        generator=torch.Generator().manual_seed(14),
    )
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, sampler=sampler)
    state = capture_rng_state(loader)

    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        list(iter(sampler)),
    )
    restore_rng_state(state, loader)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        list(iter(sampler)),
    )

    assert actual == expected


def test_current_schema_requires_matching_validation_state(checkpoint_factory) -> None:
    _, checkpoint = checkpoint_factory()
    missing = copy.deepcopy(checkpoint)
    del missing["training_state"]["validation"]
    with pytest.raises(CheckpointSchemaError, match=r"training_state\.validation"):
        validate_checkpoint(missing)

    mismatch = copy.deepcopy(checkpoint)
    mismatch["training_state"]["validation"]["manifest_sha256"] = "b" * 64
    with pytest.raises(CheckpointSchemaError, match="does not match checkpoint data"):
        validate_checkpoint(mismatch)
