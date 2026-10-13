#!/usr/bin/env python3
"""P18 endpoint review, retaining corpus and recording-level restoration checks."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import read, verify_receipt

METRICS = ("high_sc", "high_flux", "high_energy_error_db", "band_0_4000_sc")


def grouped(records):
    result = {"clip": {}, "recording": {}}
    for corpus in ("music_library", "speech_library"):
        subset = [r for r in records if r["corpus"] == corpus]
        if not subset:
            raise ValueError("both corpora required")
        tracks = defaultdict(list)
        for row in subset:
            tracks[row["track_id"]].append(row)
        keys = [k for k, v in subset[0]["metrics"].items() if isinstance(v, (int, float))]
        result["clip"][corpus] = {
            k: float(np.mean([r["metrics"][k] for r in subset])) for k in keys
        }
        result["recording"][corpus] = {
            k: float(
                np.mean([np.mean([r["metrics"][k] for r in rows]) for rows in tracks.values()])
            )
            for k in keys
        }
    return result


def support_checks(start, end, initial, tolerance=1e-6):
    checks = {}
    for weighting in ("clip", "recording"):
        a, b, z = [r["proxy"][weighting]["music_library"] for r in (start, end, initial)]
        for metric in ("total", "paired_restoration"):
            checks[f"{weighting}_music_{metric}_below_start_and_initial"] = (
                b[metric] < min(a[metric], z[metric]) - tolerance
            )
        checks[f"{weighting}_music_identity_not_worse"] = (
            b["identity_preservation"] <= a["identity_preservation"] + tolerance
        )
        a, b, z = [r["native"][weighting]["music_library"] for r in (start, end, initial)]
        for metric in ("high_sc", "high_flux"):
            key = f"output.paired.{metric}"
            checks[f"{weighting}_native_{metric}_below_start_and_initial"] = (
                b[key] < min(a[key], z[key]) - tolerance
            )
        for metric in ("high_energy_error_db", "band_0_4000_sc"):
            key = f"output.paired.{metric}"
            checks[f"{weighting}_native_{metric}_not_worse"] = b[key] <= a[key] + tolerance
    return checks


def run(root):
    registration, summary = read(root / "registration.json"), read(root / "summary.json")
    if (
        summary["status"] != "complete"
        or summary["registration_sha256"] != sha256_file(root / "registration.json")
        or summary["review_registration_sha256"] != sha256_file(root / "review-registration.json")
    ):
        raise ValueError("incomplete or changed study")
    for path, expected in registration["input_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("registered input drift")
    for path, expected in registration["source_hashes"].items():
        if sha256_file(root.parent / "source/training" / path) != expected:
            raise ValueError("registered source drift")
    for path, expected in summary["receipt_hashes"].items():
        receipt = root / path
        if sha256_file(receipt) != expected:
            raise ValueError("receipt drift")
        verify_receipt(receipt.parent, "files")
    checkpoints = read(root / "review-registration.json")
    for item in checkpoints.values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("review checkpoint drift")
    cases, pair_count, maximum_error = {}, 0, 0.0
    per_case_rows = {}
    expected_keys = None
    for label in ("initial", *checkpoints):
        directory = root / "review" / label
        proxy = read(directory / "proxy.json")
        if proxy["checkpoint_sha256"] != (
            None if label == "initial" else checkpoints[label]["sha256"]
        ):
            raise ValueError("proxy checkpoint mismatch")
        rows = proxy["groups"]["development"]["rows"]
        keys = [(r["spec"], r["track_id"], r["corpus"]) for r in rows]
        if expected_keys is None:
            expected_keys = keys
        if keys != expected_keys or len(keys) != 89:
            raise ValueError("review samples differ")
        native_rows = []
        for index, row in enumerate(rows):
            value = read(directory / "native" / f"{index:03d}.json")
            if (value["spec"], value["track_id"], value["corpus"]) != keys[index] or value[
                "checkpoint_sha256"
            ] != proxy["checkpoint_sha256"]:
                raise ValueError("native sample/checkpoint mismatch")
            native_rows.append(
                {
                    "track_id": row["track_id"],
                    "corpus": row["corpus"],
                    "metrics": {
                        f"{stage}.{role}.{metric}": value[role][f"{stage}_band_metrics"][metric]
                        for stage in ("raw", "output")
                        for role in ("paired", "identity")
                        for metric in METRICS
                    },
                }
            )
            pair_count += 1
            maximum_error = max(
                maximum_error,
                value["paired"]["proxy_max_error"],
                value["identity"]["proxy_max_error"],
            )
        cases[label] = {
            "proxy": grouped(rows),
            "weighted_proxy": proxy["groups"]["development"]["weighted"],
            "shifted": grouped(proxy["groups"]["shifted"]["rows"]),
            "native": grouped(native_rows),
        }
        per_case_rows[label] = (rows, native_rows)
    if pair_count * 2 != 1246:
        raise ValueError("incomplete native review")
    results = []
    for seed in registration["seeds"]:
        checks = support_checks(
            cases[f"start-{seed}"],
            cases[f"final-{seed}"],
            cases["initial"],
            registration["tolerance"],
        )
        training = []
        for step in registration["training_checks"]:
            directory = root / f"seed-{seed}"
            score = read(directory / f"training-{step}.json")
            if score["checkpoint_sha256"] != sha256_file(directory / f"step-{step}.pth"):
                raise ValueError("training score/checkpoint mismatch")
            training.append(
                {
                    "step": step,
                    **score["weighted"],
                    "auxiliary": score["auxiliary"],
                    "combined": score["combined"],
                }
            )
        exceptions = {}
        final_proxy, final_native = per_case_rows[f"final-{seed}"]
        for reference in ("initial", f"start-{seed}"):
            prior_proxy, prior_native = per_case_rows[reference]
            deltas = []
            for current, prior, current_native, old_native in zip(
                final_proxy, prior_proxy, final_native, prior_native, strict=True
            ):
                if current["corpus"] != "music_library":
                    continue
                deltas.append(
                    {
                        "spec": current["spec"],
                        "track_id": current["track_id"],
                        "paired_delta": current["metrics"]["paired_restoration"]
                        - prior["metrics"]["paired_restoration"],
                        "identity_delta": current["metrics"]["identity_preservation"]
                        - prior["metrics"]["identity_preservation"],
                        "native_high_sc_delta": current_native["metrics"]["output.paired.high_sc"]
                        - old_native["metrics"]["output.paired.high_sc"],
                    }
                )
            exceptions[reference] = {
                "music_clips": len(deltas),
                "worse_counts": {
                    key: sum(d[key] > registration["tolerance"] for d in deltas)
                    for key in ("paired_delta", "identity_delta", "native_high_sc_delta")
                },
                "largest_paired_regressions": sorted(
                    deltas, key=lambda d: d["paired_delta"], reverse=True
                )[:5],
                "per_clip": deltas,
            }
        results.append(
            {
                "seed": seed,
                "training": training,
                "checks": checks,
                "restoration_screen_passed": all(checks.values()),
                "music_exceptions": exceptions,
            }
        )
    report = {
        "scope": registration["scope"],
        "registration_sha256": sha256_file(root / "registration.json"),
        "summary_sha256": sha256_file(root / "summary.json"),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "cases": cases,
        "results": results,
        "both_seeds_passed_restoration_screen": all(
            r["restoration_screen_passed"] for r in results
        ),
        "native_count": pair_count * 2,
        "native_proxy_max_error": maximum_error,
    }
    write_json_report(report, root / "comparisons.json")
    print(
        {
            "both_seeds_passed_restoration_screen": report["both_seeds_passed_restoration_screen"],
            "native_count": report["native_count"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
