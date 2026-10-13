#!/usr/bin/env python3
"""P19 frozen coverage comparison at matched updates and matched music exposure."""

import argparse
from pathlib import Path

from data.causal_study import digest
from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_consistency_followup import METRICS, grouped, support_checks
from scripts.run_consistency_followup import read, verify_receipt


def run(root):
    config, summary = read(root / "registration.json"), read(root / "summary.json")
    if (
        summary["status"] != "complete"
        or summary["registration_sha256"] != sha256_file(root / "registration.json")
        or summary["review_registration_sha256"] != sha256_file(root / "review-registration.json")
    ):
        raise ValueError("incomplete or changed coverage study")
    for path, expected in config["input_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("input drift")
    for name, expected in config["source_hashes"].items():
        if sha256_file(root.parent / "source/training" / name) != expected:
            raise ValueError("source drift")
    for name, expected in summary["receipt_hashes"].items():
        path = root / name
        if sha256_file(path) != expected:
            raise ValueError("receipt drift")
        verify_receipt(path.parent, "files")
    for name, expected in summary["initialization_hashes"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("initialization receipt drift")
    for name, expected in config["schedule_hashes"].items():
        if digest(read(root / f"draws-{name}.json")["draws"]) != expected:
            raise ValueError("schedule drift")
    checkpoints = read(root / "review-registration.json")
    expected_labels = {
        f"{label}-{step}-{seed}"
        for label in config["endpoints"]
        for step in config["endpoints"][label]
        for seed in config["seeds"]
    }
    if set(checkpoints) != expected_labels or len(checkpoints) != 6:
        raise ValueError("registered endpoints differ")
    for value in checkpoints.values():
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("checkpoint drift")
    cases, source_rows, native_count, maximum = {}, {}, 0, 0.0
    expected = None
    for label in ("initial", *sorted(checkpoints)):
        directory = root / "review" / label
        proxy = read(directory / "proxy.json")
        checkpoint_hash = None if label == "initial" else checkpoints[label]["sha256"]
        if proxy["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("proxy checkpoint mismatch")
        rows = proxy["groups"]["development"]["rows"]
        keys = [(r["spec"], r["track_id"], r["corpus"]) for r in rows]
        if expected is None:
            expected = keys
        if keys != expected or len(keys) != 89:
            raise ValueError("development cases differ")
        native_rows = []
        for index, row in enumerate(rows):
            value = read(directory / "native" / f"{index:03d}.json")
            if (
                value["checkpoint_sha256"] != checkpoint_hash
                or (value["spec"], value["track_id"], value["corpus"]) != keys[index]
            ):
                raise ValueError("native case mismatch")
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
            native_count += 2
            maximum = max(
                maximum, *(value[role]["proxy_max_error"] for role in ("paired", "identity"))
            )
        cases[label] = {
            "proxy": grouped(rows),
            "weighted_proxy": proxy["groups"]["development"]["weighted"],
            "native": grouped(native_rows),
            "shifted": grouped(proxy["groups"]["shifted"]["rows"]),
        }
        source_rows[label] = rows
    if native_count != config["native_count"]:
        raise ValueError("incomplete native replay coverage")
    results = []
    for seed in config["seeds"]:
        small = f"small-2048-{seed}"
        for budget, step in (("common_updates", 2048), ("equal_music_exposure", 16192)):
            wide = f"wide-{step}-{seed}"
            checks = support_checks(
                cases[small], cases[wide], cases["initial"], config["tolerance"]
            )
            deltas = []
            for a, b in zip(source_rows[small], source_rows[wide], strict=True):
                if a["corpus"] == "music_library":
                    deltas.append(
                        {
                            "spec": a["spec"],
                            "track_id": a["track_id"],
                            **{
                                key: b["metrics"][key] - a["metrics"][key]
                                for key in ("total", "paired_restoration", "identity_preservation")
                            },
                        }
                    )
            results.append(
                {
                    "seed": seed,
                    "budget": budget,
                    "small": small,
                    "wide": wide,
                    "checks": checks,
                    "restoration_screen_passed": all(checks.values()),
                    "music_deltas": deltas,
                    "paired_worse_count": sum(
                        r["paired_restoration"] > config["tolerance"] for r in deltas
                    ),
                }
            )
    write_json_report(
        {
            "scope": config["scope"],
            "registration_sha256": sha256_file(root / "registration.json"),
            "summary_sha256": sha256_file(root / "summary.json"),
            "analyzer_sha256": sha256_file(Path(__file__)),
            "cases": cases,
            "results": results,
            "native_count": native_count,
            "native_proxy_max_error": maximum,
            "both_seeds_passed_by_budget": {
                budget: all(
                    r["restoration_screen_passed"] for r in results if r["budget"] == budget
                )
                for budget in ("common_updates", "equal_music_exposure")
            },
            "exposure": config["exposure"],
            "speech_caveat": config["speech_caveat"],
        },
        root / "comparisons.json",
    )
    print(
        {"native_count": native_count, "passed": [r["restoration_screen_passed"] for r in results]},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    run(parser.parse_args().study)
