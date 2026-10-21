#!/usr/bin/env python3
"""P35 common-metric, matched-seed assessment after all endpoints are frozen."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_risk_balance_study import recording_means
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read


def run(root):
    root = root.resolve()
    config = read(root / "registration.json")
    if config["format"] != "soundex-broad-risk-v1":
        raise ValueError("P35 protocol required")
    verify_inputs(config)
    endpoints = read(root / "review-registration.json")
    expected = {f"{a}-{config['steps']}-{s}" for a in config["arms"] for s in config["seeds"]}
    if set(endpoints) != expected:
        raise ValueError("all endpoints required")
    cases, evidence = {}, {}
    for label in ["initial", *endpoints]:
        path = root / "spectrum-review" / label / "report.json"
        data = read(path)
        evidence[str(path)] = sha256_file(path)
        if label != "initial" and data["checkpoint_sha256"] != endpoints[label]["sha256"]:
            raise ValueError("report checkpoint drift")
        grouped = defaultdict(list)
        for row in data["rows"]:
            grouped["/".join((row["corpus"], row["role"], row["stage"]))].append(row)
        scores = {}
        for group, rows in grouped.items():
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
        checkpoint = Path(endpoints[label]["path"])
        if sha256_file(checkpoint) != endpoints[label]["sha256"]:
            raise ValueError("endpoint drift")
        history = read(checkpoint.parent / "updates.json")
        if [r["step"] for r in history] != list(range(1, config["steps"] + 1)):
            raise ValueError("incomplete training")
        cases[label]["clipped_updates"] = sum(r["gradient_norm"] > 5 for r in history)
        cases[label]["gradient_max"] = max(r["gradient_norm"] for r in history)
        path = root / "native-audit" / f"{label}.json"
        native = read(path)["rows"]
        evidence[str(path)] = sha256_file(path)
        cases[label]["synthesis_max_error"] = max(r["synthesis_max_error"] for r in native)
        cases[label]["synthesis_failures"] = sum(not r["synthesis_parity_passed"] for r in native)
        pcm = defaultdict(list)
        for row in native:
            pcm["/".join((row["corpus"], row["role"]))].append(row)
        cases[label]["native_pcm"] = {
            group: {
                name: statistics.mean(
                    recording_means(rows, lambda r, name=name: r[name]["error"]["rms"]).values()
                )
                for name in ("native_vs_clean", "native_vs_identity_dsp")
            }
            for group, rows in pcm.items()
        }
    comparisons = {}
    for seed in config["seeds"]:
        control = cases[f"control-{config['steps']}-{seed}"]
        candidate = cases[f"compressed_identity_mid-{config['steps']}-{seed}"]
        result = {}
        for group, score in candidate["spectral"].items():
            baseline = control["spectral"][group]
            original = cases["initial"]["spectral"][group.rsplit("/", 1)[0] + "/input"]
            if score["recordings"].keys() != baseline["recordings"].keys():
                raise ValueError("unmatched recording population")
            delta = [v - baseline["recordings"][k] for k, v in score["recordings"].items()]
            result[group] = {
                "delta_control_pct": (score["recording_mean"] / baseline["recording_mean"] - 1)
                * 100
                if baseline["recording_mean"]
                else None,
                "delta_input_pct": (score["recording_mean"] / original["recording_mean"] - 1) * 100
                if original["recording_mean"]
                else None,
                "median_recording_difference": statistics.median(delta),
                "wins": sum(d < 0 for d in delta),
                "losses": sum(d > 0 for d in delta),
                "ties": sum(d == 0 for d in delta),
            }
        music = result["music_library/paired/native_output"]
        gates = {
            "music_mean_improves": music["delta_control_pct"] is not None
            and music["delta_control_pct"] < 0,
            "music_median_improves": music["median_recording_difference"] < 0,
        }
        for corpus in ("music_library", "speech_library"):
            group = f"{corpus}/identity/native_output"
            gates[f"{corpus}_identity_preserved"] = (
                candidate["spectral"][group]["recording_mean"]
                <= control["spectral"][group]["recording_mean"]
            )
            for role in ("paired", "identity"):
                group = f"{corpus}/{role}/native_output"
                gates[f"{corpus}_{role}_dynamics_preserved"] = (
                    candidate["spectral"][group]["components_recording_mean"]["dynamics"]
                    <= control["spectral"][group]["components_recording_mean"]["dynamics"]
                )
        comparisons[str(seed)] = {
            "groups": result,
            "research_gates": gates,
            "all_pass": all(gates.values()),
        }
    result = {
        "scope": config["review_scope"],
        "registration_sha256": sha256_file(root / "registration.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "evidence": evidence,
        "cases": cases,
        "comparisons": comparisons,
        "research_screen_pass": all(v["all_pass"] for v in comparisons.values()),
        "promotion": False,
    }
    path = root / "native-cost/summary.json"
    if path.exists():
        evidence[str(path)] = sha256_file(path)
        timing = read(path)["rows"]
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
    print({"research_screen_pass": result["research_screen_pass"], "promotion": False})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    run(parser.parse_args().study)
