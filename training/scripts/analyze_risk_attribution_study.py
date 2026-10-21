#!/usr/bin/env python3
"""P36 matched-seed factorial contrasts; all frozen cells, no model promotion."""

import argparse
import math
import statistics
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_broad_risk_study import collect_cases
from scripts.run_consistency_followup import read


def compare(candidate, control):
    if candidate["recordings"].keys() != control["recordings"].keys():
        raise ValueError("unmatched recording populations")
    differences = [v - control["recordings"][k] for k, v in candidate["recordings"].items()]
    return {
        "difference": candidate["recording_mean"] - control["recording_mean"],
        "delta_pct": (candidate["recording_mean"] / control["recording_mean"] - 1) * 100
        if control["recording_mean"]
        else None,
        "median_recording_difference": statistics.median(differences),
        "wins": sum(d < 0 for d in differences),
        "losses": sum(d > 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "component_differences": {
            k: v - control["components_recording_mean"][k]
            for k, v in candidate["components_recording_mean"].items()
        },
    }


def factorial(cells):
    """Differences of scores, never differences of percentages or optimized loss."""
    groups = cells["raw_full"].keys()
    if any(set(cell) != set(groups) for cell in cells.values()):
        raise ValueError("unmatched factorial score groups")
    contrasts = {
        "compression_at_identity1": ("log_full", "raw_full"),
        "compression_at_identity03": ("log_mid", "raw_mid"),
        "identity_reduction_without_compression": ("raw_mid", "raw_full"),
        "identity_reduction_with_compression": ("log_mid", "log_full"),
    }
    result = {
        name: {group: compare(cells[a][group], cells[b][group]) for group in groups}
        for name, (a, b) in contrasts.items()
    }
    result["interaction"] = {
        group: (
            cells["log_mid"][group]["recording_mean"] - cells["log_full"][group]["recording_mean"]
        )
        - (cells["raw_mid"][group]["recording_mean"] - cells["raw_full"][group]["recording_mean"])
        for group in groups
    }
    return result


def screen(candidate, control):
    music = compare(
        candidate["music_library/paired/native_output"],
        control["music_library/paired/native_output"],
    )
    gates = {
        "music_mean_improves": music["difference"] < 0,
        "music_median_improves": music["median_recording_difference"] < 0,
    }
    for corpus in ("music_library", "speech_library"):
        group = f"{corpus}/identity/native_output"
        gates[f"{corpus}_identity_preserved"] = (
            candidate[group]["recording_mean"] <= control[group]["recording_mean"]
        )
        for role in ("paired", "identity"):
            group = f"{corpus}/{role}/native_output"
            gates[f"{corpus}_{role}_dynamics_preserved"] = (
                candidate[group]["components_recording_mean"]["dynamics"]
                <= control[group]["components_recording_mean"]["dynamics"]
            )
    return {"gates": gates, "all_pass": all(gates.values())}


def run(root):
    root = root.resolve()
    config, cases, evidence = collect_cases(root, study_format="soundex-risk-attribution-v1")
    result = {
        "scope": config["review_scope"],
        "smoke": config["smoke"],
        "registration_sha256": sha256_file(root / "registration.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "evidence": evidence,
        "cases": cases,
        "promotion": False,
    }
    if not config["smoke"]:
        reference = Path(config["reference_study"])
        parent = read(reference / "registration.json")
        if any(
            config[k] != parent[k]
            for k in (
                "steps",
                "seeds",
                "development",
                "direct_loss",
                "multires_loss",
                "rate",
                "schedule_hashes",
            )
        ):
            raise ValueError("unmatched reference design")
        old = read(reference / "assessment.json")
        # Same unprocessed/identity DSP populations and values, not a changed baseline.
        for group, score in cases["initial"]["spectral"].items():
            prior = old["cases"]["initial"]["spectral"][group]
            if score["recordings"].keys() != prior["recordings"].keys() or any(
                not math.isclose(v, prior["recordings"][k], rel_tol=1e-6, abs_tol=1e-10)
                for k, v in score["recordings"].items()
            ):
                raise ValueError("reference scoring population or metric drift")
        cases.update({label: case for label, case in old["cases"].items() if label != "initial"})
        result["reference_assessment_sha256"] = sha256_file(reference / "assessment.json")
        result["factorial"], result["screens"] = {}, {}
        for seed in config["seeds"]:
            cells = {
                name: cases[f"{arm}-{config['steps']}-{seed}"]["spectral"]
                for name, arm in {
                    "raw_full": "control",
                    "raw_mid": "identity_mid",
                    "log_full": "compressed",
                    "log_mid": "compressed_identity_mid",
                }.items()
            }
            result["factorial"][str(seed)] = factorial(cells)
            for name in ("raw_mid", "log_full"):
                result["screens"].setdefault(name, {})[str(seed)] = screen(
                    cells[name], cells["raw_full"]
                )
        result["research_screen_pass"] = {
            name: all(v["all_pass"] for v in seeds.values())
            for name, seeds in result["screens"].items()
        }
    timing = root / "native-cost/summary.json"
    if timing.exists():
        result["native_cost_summary"] = str(timing)
        evidence[str(timing)] = sha256_file(timing)
    write_json_report(result, root / "assessment.json")
    print(
        {
            "smoke": config["smoke"],
            "research_screen_pass": result.get("research_screen_pass"),
            "promotion": False,
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    run(parser.parse_args().study)
