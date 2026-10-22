#!/usr/bin/env python3
"""P38: matched CPU training with native versus fixed explicit detector conditions."""

import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace

from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_broad_risk_study import collect_cases
from scripts.audit_fusion_boundaries import run as audit_fusion
from scripts.run_broad_risk_study import protocol as broad_protocol
from scripts.run_broad_risk_study import register as broad_register
from scripts.run_broad_risk_study import run as broad_run
from scripts.run_consistency_followup import read
from scripts.run_head_study import progress
from scripts.run_risk_attribution_study import training_terms as parent_terms

ARMS = ("control", "fixed_detector")
FORMAT = "soundex-detector-condition-v1"


def training_terms(prediction, budget, objective, weight):
    terms = parent_terms(prediction, budget, objective, weight)
    return {**terms, "control": terms["compressed"], "fixed_detector": terms["compressed"]}


def protocol(parent, *, preflight=False):
    config = broad_protocol(parent, preflight=preflight)
    model = parent["model_configs"]["compressed"]
    config.update(
        format=FORMAT,
        arms=list(ARMS),
        model_configs={
            "control": dict(model),
            "fixed_detector": {**model, "detector_condition": "fixed"},
        },
        parameters=dict.fromkeys(ARMS, 431015),
        loss_routes={a: {a: 1.0} for a in ARMS},
        paced_arms=list(ARMS),
        training_device="cpu",
        training_threads=1,
        contrast="same P36 compressed objective and identity1; only explicit model cutoff/active are fixed to .5/1 inside forward and ONNX, including warmup. Rate, detector, gate, loss, optimizer, draws and parameter tensors unchanged",
        selection="all endpoints frozen before review; current and phase-only output paths fixed in advance. Within EACH path compare fixed_detector to same-seed control, report raw/native common MR and all recordings. Both seeds must improve music mean/median with no corpus identity or role dynamics regression for research continuation on that path; never automatic promotion or select best checkpoint/path",
        fusion_review_scope="P38 all frozen control/fixed_detector endpoints, current and phase-only output paths; reused recording-isolated validation, not independent test; no promotion",
        review_scope="P38 matched CPU endpoints on reused validation; original clean reference, raw/native separately; current and phase-only paths reported without selecting a winner",
    )
    return config


def register(args, root):
    parent_root = args.parent.resolve()
    parent = read(parent_root / "registration.json")
    if parent["format"] != "soundex-risk-attribution-v1":
        raise ValueError("P36 parent required")
    initials = {
        str(seed): read(parent_root / f"initialization-compressed-{seed}.json")["initial_sha256"]
        for seed in parent["seeds"]
    }

    def make(parent, *, preflight=False):
        config = protocol(parent, preflight=preflight)
        config["repository_root"] = str(Path(__file__).resolve().parents[2])
        config["reference_initializations"] = initials
        return config

    return broad_register(
        args,
        root,
        make_protocol=make,
        parent_format="soundex-risk-attribution-v1",
        study_format=FORMAT,
        reference_files=[
            parent_root / f"initialization-compressed-{s}.json" for s in parent["seeds"]
        ],
    )


def assess(root):
    config, legacy, evidence = collect_cases(root, study_format=FORMAT)
    binary = next(
        Path(p) for p in config["input_hashes"] if Path(p).name == "soundex-context-replay"
    )
    progress(root, stage="fusion_validation_review")
    output = root / "fusion-review"
    audit_fusion(
        SimpleNamespace(study=root, output=output, binary=binary, preflight=False),
        endpoint_arms=ARMS,
        cases=(("current", "native"), ("phase-only", "native")),
        expected_endpoints=len(ARMS) * len(config["seeds"]),
        require_complete=False,
    )
    report = read(output / "assessment.json")
    evidence[str(output / "assessment.json")] = sha256_file(output / "assessment.json")
    comparisons = compare_endpoints(report["cases"], config)
    write_json_report(
        dict(
            scope=config["review_scope"],
            registration_sha256=sha256_file(root / "registration.json"),
            evidence=evidence,
            current_native=legacy,
            comparisons=comparisons,
            promotion=False,
        ),
        root / "assessment.json",
    )


def compare_endpoints(cases, config):
    import statistics

    output = {}
    for fusion in ("current--native", "phase-only--native"):
        by_seed = {}
        for seed in config["seeds"]:
            control = cases[f"control-{config['steps']}-{seed}"][fusion]
            candidate = cases[f"fixed_detector-{config['steps']}-{seed}"][fusion]
            groups = {}
            for group, metrics in candidate.items():
                groups[group] = {}
                for metric, value in metrics.items():
                    old = control[group][metric]
                    if value["recordings"].keys() != old["recordings"].keys():
                        raise ValueError("recording population mismatch")
                    delta = [v - old["recordings"][k] for k, v in value["recordings"].items()]
                    groups[group][metric] = dict(
                        control=old["mean"],
                        candidate=value["mean"],
                        delta_pct=(value["mean"] / old["mean"] - 1) * 100 if old["mean"] else None,
                        median_difference=statistics.median(delta),
                        wins=sum(v < 0 for v in delta),
                        losses=sum(v > 0 for v in delta),
                    )
            music = groups["music_library/paired/native_output"]["total"]
            gates = dict(
                music_mean=music["candidate"] < music["control"],
                music_median=music["median_difference"] < 0,
            )
            for corpus in ("music_library", "speech_library"):
                identity = groups[f"{corpus}/identity/native_output"]["total"]
                gates[f"{corpus}_identity"] = identity["candidate"] <= identity["control"]
                for role in ("paired", "identity"):
                    dynamics = groups[f"{corpus}/{role}/native_output"]["dynamics"]
                    gates[f"{corpus}_{role}_dynamics"] = (
                        dynamics["candidate"] <= dynamics["control"]
                    )
            by_seed[str(seed)] = dict(
                groups=groups, research_gates=gates, all_pass=all(gates.values())
            )
        output[fusion] = dict(
            seeds=by_seed, research_screen_pass=all(s["all_pass"] for s in by_seed.values())
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "native-binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        broad_run(
            args, register_study=register, training_objective=training_terms, assessment=assess
        )
    except Exception:
        write_json_report(
            dict(stage="failed", error=traceback.format_exc()), args.output / "pipeline-error.json"
        )
        raise
