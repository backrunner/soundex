#!/usr/bin/env python3
"""P36: complete the compression x identity-weight factorial at matched budget."""

import argparse
import traceback
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss
from models.risk_balance import reconstruction_risk
from scripts.analyze_risk_attribution_study import run as assess
from scripts.run_broad_risk_study import SEEDS, STEPS
from scripts.run_broad_risk_study import protocol as broad_protocol
from scripts.run_broad_risk_study import register as broad_register
from scripts.run_broad_risk_study import run as broad_run
from scripts.run_consistency_followup import read
from scripts.run_direct_spectrum_study import MODEL

ARMS = ("compressed", "identity_mid")


def training_terms(prediction, budget, objective, weight):
    del weight
    batch = objective.batch
    direct = direct_spectrum_loss(prediction, batch)
    multires = multires_spectrum_loss(prediction, batch)
    args = (direct["per_example"], multires["per_example"], batch["identity"])
    auxiliary = 0.1 * budget_risk(budget, batch)
    return {
        "compressed": reconstruction_risk(*args, compress=True) + auxiliary,
        "identity_mid": reconstruction_risk(*args, identity_weight=0.3) + auxiliary,
        "raw": direct["total"],
        "multires": multires["total"],
        "budget": auxiliary,
    }


def protocol(parent, *, preflight=False):
    if parent["steps"] != STEPS or parent["seeds"] != list(SEEDS):
        raise ValueError("matched parent budget and seeds required")
    if parent["rate"] != 1e-4 or any(v != MODEL for v in parent["model_configs"].values()):
        raise ValueError("parent optimizer or architecture drift")
    config = broad_protocol(parent, preflight=preflight)
    config.update(
        format="soundex-risk-attribution-v1",
        arms=list(ARMS),
        model_configs={a: dict(MODEL) for a in ARMS},
        parameters=dict.fromkeys(ARMS, 431015),
        loss_routes={a: {a: 1.0} for a in ARMS},
        paced_arms=list(ARMS),
        contrast="complete the fixed 2x2 factorial: P35 uncompressed/identity1 and compressed/identity.3; P36 compressed/identity1 and uncompressed/identity.3. Same paired coefficient .5 and .1 budget risk; no new loss or DSP changes",
        selection="all four new endpoints frozen before any reused-validation scoring; report both seeds and every factorial contrast plus interaction in common uncompressed MR units. Research screen per new arm: both seeds improve native music mean and median recording difference versus P35 control, with no corpus identity MR or either-role dynamics regression. Failures remain diagnostic; no automatic promotion or best-checkpoint selection",
        review_scope="all fixed final endpoints, matched P35 reference cells; recording-isolated REUSED validation, not independent test; spectral primary, raw/native separate",
    )
    return config


def register(args, root):
    reference = args.parent.resolve()
    parent = read(reference / "registration.json")
    if parent["format"] != "soundex-broad-risk-v1":
        raise ValueError("P35 reference required")
    native = args.native_binary.resolve()
    if parent["input_hashes"].get(str(native)) != sha256_file(native):
        raise ValueError("reference native binary drift")
    source = Path(__file__).resolve().parents[1]
    # Objective, data construction, architecture and optimizer must be identical.
    for name, expected in parent["source_hashes"].items():
        if (
            name.startswith(("models/", "data/")) or name == "scripts/run_direct_spectrum_study.py"
        ) and sha256_file(source / name) != expected:
            raise ValueError(f"reference execution dependency drift: {name}")
    assessment = read(reference / "assessment.json")
    if assessment["registration_sha256"] != sha256_file(reference / "registration.json"):
        raise ValueError("reference assessment drift")
    files = [reference / "assessment.json", reference / "review-registration.json"]
    for name, expected in assessment["evidence"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("reference evidence drift")
        files.append(Path(name))
    endpoints = read(reference / "review-registration.json")
    if set(endpoints) != {f"{a}-{STEPS}-{s}" for a in parent["arms"] for s in SEEDS}:
        raise ValueError("all reference endpoints required")
    for value in endpoints.values():
        path = Path(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise ValueError("reference endpoint drift")
        files.append(path)
    initials = {}
    for seed in SEEDS:
        hashes = []
        for arm in parent["arms"]:
            path = reference / f"initialization-{arm}-{seed}.json"
            hashes.append(read(path)["initial_sha256"])
            files.append(path)
        if len(set(hashes)) != 1:
            raise ValueError("reference initialization mismatch")
        initials[str(seed)] = hashes[0]

    def make(parent, *, preflight=False):
        config = protocol(parent, preflight=preflight)
        config["reference_study"] = str(reference)
        config["reference_initializations"] = initials
        return config

    return broad_register(
        args,
        root,
        make_protocol=make,
        parent_format="soundex-broad-risk-v1",
        study_format="soundex-risk-attribution-v1",
        reference_files=files,
    )


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
            {"stage": "failed", "error": traceback.format_exc()},
            args.output / "pipeline-error.json",
        )
        raise
