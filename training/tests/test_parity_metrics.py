# SPDX-License-Identifier: Apache-2.0
"""Accept numerical rounding while rejecting feature and signal corruption."""

from __future__ import annotations

import copy
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from artifact_contract import ExportValidationError
from parity_metrics import (
    PARITY_POLICY,
    POLICY_SHA256,
    REQUIRED_PARITY_CASES,
    parity_evidence_valid,
    validate_features,
)


def features(db: float = -30.0) -> np.ndarray:
    value = np.zeros((1, 2, 1, 129), dtype=np.float32)
    value[:, 0] = db
    return value


def test_fp32_rounding_and_periodic_phase_are_allowed() -> None:
    expected = features()
    actual = expected.copy()
    actual[:, 0] += 5e-5
    actual[:, 1] += np.float32(2 * np.pi)
    metrics = validate_features(expected, actual, label="rounding")
    assert metrics["raw_max_absolute_error"] > 6
    assert metrics["max_phase_rad"] < 1e-6


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_and_overflow_fail_closed(invalid: float) -> None:
    actual = features()
    actual[0, 0, 0, 12] = invalid
    with pytest.raises(ExportValidationError, match="non-finite"):
        validate_features(features(), actual, label="corrupt")
    actual = features(1e20)
    with pytest.raises(ExportValidationError, match="non-finite"):
        validate_features(actual, actual, label="overflow")


@pytest.mark.parametrize("channel", [0, 1])
def test_large_local_feature_errors_cannot_hide_in_the_mean(channel: int) -> None:
    expected = features()
    actual = expected.copy()
    actual[0, channel, 0, 64] += 0.01
    with pytest.raises(ExportValidationError, match="mismatch"):
        validate_features(expected, actual, label="one-bin")


def test_mean_and_linear_signal_guards_are_independent() -> None:
    expected = features()
    actual = expected.copy()
    actual[:, 0] += 5e-4
    with pytest.raises(ExportValidationError, match="mean_magnitude_db"):
        validate_features(expected, actual, label="bias")
    expected = features(-200)
    expected[0, 0, 0, 64] = 0
    actual = expected.copy()
    actual[0, 1, 0, 64] = 5e-4
    with pytest.raises(ExportValidationError, match="max_complex_relative_rms"):
        validate_features(expected, actual, label="dominant-bin")
    expected[0, 0, 0, 64] = 80
    actual = expected.copy()
    actual[0, 0, 0, 64] += 5e-5
    with pytest.raises(ExportValidationError, match="max_inverse_fft_error_bound"):
        validate_features(expected, actual, label="large-linear-signal")


def test_loud_batch_item_cannot_mask_corruption_in_a_quiet_item() -> None:
    expected = np.concatenate((features(0), features(-200)))
    expected[1, 0, 0, 64] = -100
    actual = expected.copy()
    actual[1, 1, 0, 64] = 5e-4
    with pytest.raises(ExportValidationError, match="max_complex_relative_rms"):
        validate_features(expected, actual, label="asymmetric-stereo")


def passing_evidence() -> dict[str, object]:
    metrics = {key: 0.0 for key in PARITY_POLICY if key.startswith(("max_", "mean_"))}
    return {
        "schema_version": 2,
        "suite": "pytorch-ort-rust-v2",
        "passed": True,
        "policy": copy.deepcopy(PARITY_POLICY),
        "policy_sha256": POLICY_SHA256,
        "cases": [
            {"name": name, "pytorch_ort": metrics.copy(), "pytorch_rust": metrics.copy()}
            for name in sorted(REQUIRED_PARITY_CASES)
        ],
    }


def test_evidence_cannot_raise_budgets_omit_cases_or_fake_a_pass() -> None:
    evidence = passing_evidence()
    assert parity_evidence_valid(evidence)
    evidence["policy"]["max_phase_rad"] = 1.0
    assert not parity_evidence_valid(evidence)
    evidence = passing_evidence()
    evidence["cases"].pop()
    assert not parity_evidence_valid(evidence)
    for invalid in (True, float("nan"), float("inf"), 0.1):
        evidence = passing_evidence()
        evidence["cases"][0]["pytorch_rust"]["max_phase_rad"] = invalid
        assert not parity_evidence_valid(evidence)


@pytest.mark.parametrize("change,passes", [(5e-5, True), (0.01, False)])
def test_rust_and_python_enforce_the_same_policy(
    tmp_path: Path, change: float, passes: bool
) -> None:
    repository = Path(__file__).resolve().parents[2]
    input_value = features()
    expected = input_value.copy()
    expected[:, 0] += change
    expected[:, 1] += np.float32(2 * np.pi)
    paths = []
    for name, value in (("input", input_value), ("expected", expected)):
        path = tmp_path / f"{name}.sxt"
        path.write_bytes(b"SXT1" + struct.pack("<4I", *value.shape) + value.astype("<f4").tobytes())
        paths.append(path)
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "soundex-core",
            "--bin",
            "soundex-model-parity",
            "--",
            str(repository / "tests/fixtures/low-latency-identity.onnx"),
            *map(str, paths),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert (result.returncode == 0) is passes, result.stderr
    if passes:
        report = json.loads(result.stdout.splitlines()[-1])
        assert report["policy"] == PARITY_POLICY
        measured = validate_features(expected, input_value, label="rust")
        for field, value in report["metrics"].items():
            assert value == pytest.approx(measured[field], abs=1e-12)
