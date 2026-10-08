# SPDX-License-Identifier: Apache-2.0
"""Unit-aware FP32 feature parity with circular phase and linear signal guards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from artifact_contract import ExportValidationError

POLICY_PATH = Path(__file__).with_name("parity_policy.json")
PARITY_POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
METRIC_FIELDS = frozenset(
    {
        "max_magnitude_db",
        "mean_magnitude_db",
        "max_phase_rad",
        "mean_phase_rad",
        "max_complex_relative_rms",
        "max_inverse_fft_error",
    }
)
if set(PARITY_POLICY) != METRIC_FIELDS | {"schema_version", "policy_id", "complex_rms_floor"}:
    raise ExportValidationError("parity policy must specify every unit-aware budget")
for field in METRIC_FIELDS | {"complex_rms_floor"}:
    value = PARITY_POLICY[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ExportValidationError(f"invalid parity budget {field}")
POLICY_SHA256 = hashlib.sha256(
    json.dumps(PARITY_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
REQUIRED_PARITY_CASES = frozenset(
    ["silence", "near_full_scale_sine", "random_finite_spectra", "db_floor", "phase_edges"]
    + [
        f"{name}_{rate}"
        for rate in (44100, 48000)
        for name in ("music_tones", "quiet", "noise", "impulse")
    ]
)


def parity_evidence_valid(evidence: dict[str, object]) -> bool:
    """Validate policy identity, full case coverage and every reported budget."""
    if not isinstance(evidence, dict):
        return False
    if (
        evidence.get("schema_version") != 2
        or evidence.get("suite") != "pytorch-ort-rust-v2"
        or evidence.get("passed") is not True
        or evidence.get("policy") != PARITY_POLICY
        or evidence.get("policy_sha256") != POLICY_SHA256
    ):
        return False
    cases = evidence.get("cases")
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_PARITY_CASES):
        return False
    if any(not isinstance(case, dict) for case in cases):
        return False
    names = [case.get("name") for case in cases]
    if any(not isinstance(name, str) for name in names) or set(names) != REQUIRED_PARITY_CASES:
        return False
    for case in cases:
        for runtime in ("pytorch_ort", "pytorch_rust"):
            metrics = case.get(runtime)
            if not isinstance(metrics, dict):
                return False
            for field in METRIC_FIELDS:
                value = metrics.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= PARITY_POLICY[field]
                ):
                    return False
    return True


def compare_features(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Measure each independent spectrum in float64, without changing inference."""
    if actual.shape != expected.shape or expected.ndim != 4:
        raise ExportValidationError("parity tensor shape mismatch")
    if expected.shape[1:3] != (2, 1) or expected.shape[0] == 0 or expected.shape[-1] < 2:
        raise ExportValidationError("parity requires nonempty [B, 2, 1, F] tensors")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ExportValidationError("parity produced non-finite values")
    reference = expected.astype(np.float64)
    candidate = actual.astype(np.float64)
    delta = candidate - reference
    magnitude_error = np.abs(delta[:, 0])
    phase_error = np.abs((delta[:, 1] + np.pi) % (2 * np.pi) - np.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        reference_spectrum = 10 ** (reference[:, 0] / 20) * np.exp(1j * reference[:, 1])
        candidate_spectrum = 10 ** (candidate[:, 0] / 20) * np.exp(1j * candidate[:, 1])
        complex_error = np.abs(candidate_spectrum - reference_spectrum)
        weights = np.full(expected.shape[-1], 2.0)
        weights[[0, -1]] = 1.0
        reference_rms = np.sqrt(
            np.sum(weights * np.abs(reference_spectrum) ** 2, axis=-1) / weights.sum()
        )
        error_rms = np.sqrt(np.sum(weights * complex_error**2, axis=-1) / weights.sum())
        relative = error_rms / np.maximum(reference_rms, PARITY_POLICY["complex_rms_floor"])
        # Triangle inequality bounds every sample of the normalized real inverse FFT.
        bound = np.sum(weights * complex_error, axis=-1) / (2 * (expected.shape[-1] - 1))
        inverse_error = np.fft.irfft(
            candidate_spectrum - reference_spectrum, n=2 * (expected.shape[-1] - 1)
        )
    return {
        "max_magnitude_db": float(magnitude_error.max()),
        "mean_magnitude_db": float(magnitude_error.mean()),
        "max_phase_rad": float(phase_error.max()),
        "mean_phase_rad": float(phase_error.mean()),
        "max_complex_relative_rms": float(relative.max()),
        "max_inverse_fft_error": float(np.abs(inverse_error).max()),
        "raw_inverse_fft_error_bound": float(bound.max()),
        "raw_max_absolute_error": float(np.abs(delta).max()),
        "raw_mean_error": float(np.abs(delta).mean()),
    }


def validate_features(expected: np.ndarray, actual: np.ndarray, *, label: str) -> dict[str, float]:
    """Fail closed on every unit-specific or reconstructed-signal budget."""
    metrics = compare_features(expected, actual)
    for field, value in metrics.items():
        if not np.isfinite(value):
            raise ExportValidationError(f"{label}: non-finite parity metric {field}")
        if field in PARITY_POLICY and value > PARITY_POLICY[field]:
            raise ExportValidationError(
                f"{label} mismatch: {field}={value:.3e}, limit={PARITY_POLICY[field]:.3e}"
            )
    return metrics
