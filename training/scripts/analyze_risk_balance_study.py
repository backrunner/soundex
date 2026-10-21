#!/usr/bin/env python3
"""Separate fitted and unfitted training-domain P34 evidence; never promote."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from data.causal_study import digest
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read


def recording_means(rows, value):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(value(row))
    return {k: statistics.mean(v) for k, v in grouped.items()}


def run(root):
    root = root.resolve()
    config = read(root / "registration.json")
    if config["format"] != "soundex-risk-balance-diagnostic-v1":
        raise ValueError("P34 protocol required")
    verify_inputs(config)
    labels = read(root / "review-registration.json")
    groups = dict(zip(map(digest, config["specs"]), config["review_groups"], strict=True))
    if len(groups) != len(config["specs"]):
        raise ValueError("duplicate review spec")
    cases = {}
    evidence = {}
    for label in ["initial", *labels]:
        path = root / "spectrum-review" / label / "report.json"
        data = read(path)
        evidence[str(path)] = sha256_file(path)
        if label != "initial" and data["checkpoint_sha256"] != labels[label]["sha256"]:
            raise ValueError("checkpoint report drift")
        rows_by_group = defaultdict(list)
        for row in data["rows"]:
            name = "/".join((groups[digest(row["spec"])], row["corpus"], row["role"], row["stage"]))
            rows_by_group[name].append(row)
        scores = {}
        for group, rows in rows_by_group.items():
            records = recording_means(rows, lambda r: r["score"]["total"])
            scores[group] = {
                "recordings": records,
                "recording_mean": statistics.mean(records.values()),
                "clip_mean": statistics.mean(r["score"]["total"] for r in rows),
                "components_recording_mean": {
                    k: statistics.mean(
                        recording_means(rows, lambda r, k=k: r["score"]["components"][k]).values()
                    )
                    for k in rows[0]["score"]["components"]
                },
            }
        cases[label] = {"spectral": scores}
        if label == "initial":
            continue
        path = root / "native-audit" / f"{label}.json"
        native = read(path)["rows"]
        evidence[str(path)] = sha256_file(path)
        pcm = defaultdict(list)
        for r in native:
            pcm["/".join((groups[digest(r["spec"])], r["corpus"], r["role"]))].append(r)
        cases[label]["native_pcm"] = {
            group: {
                name: statistics.mean(
                    recording_means(rows, lambda r, name=name: r[name]["error"]["rms"]).values()
                )
                for name in ("native_vs_clean", "native_vs_identity_dsp")
            }
            for group, rows in pcm.items()
        }
        cases[label]["synthesis_max_error"] = max(r["synthesis_max_error"] for r in native)
        cases[label]["synthesis_failures"] = sum(not r["synthesis_parity_passed"] for r in native)
        value = labels[label]
        checkpoint = Path(value["path"])
        if sha256_file(checkpoint) != value["sha256"]:
            raise ValueError("endpoint drift")
        history = read(checkpoint.parent / "updates.json")
        if len(history) != config["steps"] or [r["step"] for r in history] != list(
            range(1, config["steps"] + 1)
        ):
            raise ValueError("incomplete training")
        cases[label]["clipped_updates"] = sum(r["gradient_norm"] > 5 for r in history)
        cases[label]["gradient_max"] = max(r["gradient_norm"] for r in history)
    control = next(label for label in labels if label.startswith("control-"))
    for label in labels:
        for group, score in cases[label]["spectral"].items():
            original = group.rsplit("/", 1)[0] + "/input"
            baseline = cases["initial"]["spectral"][original]
            ref = cases[control]["spectral"][group]
            changes = [
                (v / baseline["recordings"][k] - 1) * 100
                for k, v in score["recordings"].items()
                if baseline["recordings"][k] > 0
            ]
            score.update(
                delta_input_pct=(score["recording_mean"] / baseline["recording_mean"] - 1) * 100
                if baseline["recording_mean"]
                else None,
                delta_control_pct=(score["recording_mean"] / ref["recording_mean"] - 1) * 100
                if ref["recording_mean"]
                else None,
                median_delta_input_pct=statistics.median(changes) if changes else None,
                wins_control=sum(v < ref["recordings"][k] for k, v in score["recordings"].items()),
                losses_control=sum(
                    v > ref["recordings"][k] for k, v in score["recordings"].items()
                ),
            )
    result = {
        "scope": config["review_scope"],
        "registration_sha256": sha256_file(root / "registration.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "evidence": evidence,
        "cases": cases,
    }
    if (root / "native-cost/summary.json").exists():
        timing = read(root / "native-cost/summary.json")["rows"]
        result["timing"] = {
            "hops": sum(r["measured_hops"] for r in timing),
            "over_5ms": sum(r["stages"]["total_ms"]["over_5ms"] for r in timing),
            "over_hop": sum(r["stages"]["total_ms"]["service_over_hop"] for r in timing),
            "max_ms": max(r["stages"]["total_ms"]["max_ms"] for r in timing),
            "p99_max_ms": max(r["stages"]["total_ms"]["p99_ms"] for r in timing),
            "paced_late_hops": sum(r["paced_late_hops"] for r in timing),
            "rss_max_bytes": max(r["process_peak_rss_bytes"] for r in timing),
        }
    write_json_report(result, root / "assessment.json")
    for label in labels:
        print(label, "clipped", cases[label]["clipped_updates"])
        for scope in ("fitted", "unfitted_train"):
            for stage in ("raw_ola", "native_output"):
                score = cases[label]["spectral"][f"{scope}/music_library/paired/{stage}"]
                print(
                    scope,
                    stage,
                    {
                        k: v
                        for k, v in score.items()
                        if k not in ("recordings", "components_recording_mean")
                    },
                )
    if "timing" in result:
        print("timing", result["timing"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    run(parser.parse_args().study)
