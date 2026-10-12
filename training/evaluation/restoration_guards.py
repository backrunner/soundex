"""Predeclared P8 recording-level subgroup guards, separate from mean selection.

Failure/uncertainty in one group cannot be compensated by another group's gain.
Passing is only raw-amplitude diagnostic evidence, never production approval.
"""

import math

import numpy as np

from evaluation.reporting import bootstrap_mean_ci

TOLERANCE = 1e-6


def guard(candidate, control, corpus, key, *, weak=False, improvement=False):
    left = {r["row_id"]: r for r in candidate["rows"] if r["corpus"] == corpus}
    right = {r["row_id"]: r for r in control["rows"] if r["corpus"] == corpus}
    if left.keys() != right.keys():
        raise ValueError("unpaired restoration guard")
    groups = {}
    for row_id, row in left.items():
        old = right[row_id]
        if row["track_id"] != old["track_id"]:
            raise ValueError("recording identity changed")
        if weak:
            count = row["metrics"]["weak_band_count"]
            if count != old["metrics"]["weak_band_count"]:
                raise ValueError("reference-defined weak group changed")
            if count == 0:
                continue
        delta = row["metrics"][key] - old["metrics"][key]
        if not math.isfinite(delta):
            raise ValueError("nonfinite restoration guard")
        groups.setdefault(row["track_id"], []).append(delta)
    values = [float(np.mean(v)) for v in groups.values()]
    if len(values) < 2:
        return {
            "status": "unsupported",
            "reason": "fewer than two recordings",
            "recordings": len(values),
        }
    lower, upper = bootstrap_mean_ci(values, seed=20261012)
    bound = -TOLERANCE if improvement else TOLERANCE
    status = (
        "supported" if upper <= bound else ("regression" if lower > TOLERANCE else "inconclusive")
    )
    return {
        "status": status,
        "candidate_minus_control": float(np.mean(values)),
        "recording_ci95": [lower, upper],
        "recordings": len(values),
        "require_improvement": improvement,
    }


def restoration_guards(candidate, control, compressed):
    checks = {}
    for reference, name in ((control["validation"], "control"), (compressed, "compressed")):
        for key in ("high_sc", "high_energy_error_db"):
            checks[f"music_{key}_vs_{name}"] = guard(
                candidate["validation"], reference, "music_library", key, improvement=True
            )
    for split in ("validation", "identity"):
        checks[f"{split}_music_weak"] = guard(
            candidate[split], control[split], "music_library", "weak_reconstruction", weak=True
        )
    for corpus in ("music_library", "speech_library"):
        checks[f"identity_{corpus}"] = guard(
            candidate["identity"], control["identity"], corpus, "linear_reconstruction"
        )
    for corpus in ("music_library", "speech_library"):
        for key in ("band_0_4000_sc", "high_flux"):
            checks[f"paired_{corpus}_{key}"] = guard(
                candidate["validation"], control["validation"], corpus, key
            )
    checks["paired_speech_reconstruction"] = guard(
        candidate["validation"], control["validation"], "speech_library", "linear_reconstruction"
    )
    return {
        "scope": "raw amplitude; reused development; no release qualification",
        "all_groups_supported": all(v["status"] == "supported" for v in checks.values()),
        "checks": checks,
    }
