"""Freeze P46 policy/direction controls without rewriting the completed P45 run.

SPDX-License-Identifier: Apache-2.0
"""

import shutil
from pathlib import Path

from evaluation.reporting import sha256_file
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered

STEPS = 128
ARMS = {
    f"reference_{kind}_pcm{bits}": {"bits": bits, "projected": kind == "projected"}
    for kind in ("policy", "projected")
    for bits in (16, 20, 24)
}


def prepare(prior: Path, output: Path) -> dict:
    prior, output = prior.resolve(), output.resolve()
    if (output / "registration.json").exists():
        result = read(output / "registration.json")
        if result["prior"] != str(prior):
            raise ValueError("resume prior changed")
        verify(result["hashes"])
        return result
    previous = read(prior / "registration.json")
    complete = read(prior / "summary.json")
    if complete["status"] != "complete_raw_review" or previous["steps"] != STEPS:
        raise ValueError("completed, matched P45 study required")
    verify(previous["hashes"])
    hashes = dict(previous["hashes"])
    for endpoint in complete["endpoints"].values():
        if sha256_file(Path(endpoint["path"])) != endpoint["sha256"]:
            raise ValueError("P45 endpoint changed")
        hashes[endpoint["path"]] = endpoint["sha256"]
    for label, expected in complete["review_files"].items():
        path = prior / "review" / f"{label}.json"
        if sha256_file(path) != expected:
            raise ValueError("P45 review changed")
        hashes[str(path)] = expected
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(
            part in {".venv", "tests", "outputs", "__pycache__"}
            for part in p.relative_to(root).parts
        )
    ]
    sources += sorted((root / "evaluation").glob("*.json"))
    for path in [
        *sources,
        prior / "registration.json",
        prior / "summary.json",
        prior / "assessment.json",
        prior / "gate-attribution.json",
    ]:
        hashes[str(path.resolve())] = sha256_file(path)
    verify(hashes)
    for path in sources:
        destination = output / "source" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    registration = {
        "format": "soundex-reference-pilot-v1",
        "steps": STEPS,
        "arms": ARMS,
        "prior": str(prior),
        "hashes": hashes,
        "endpoints": previous["endpoints"],
        "groups": previous["groups"],
        "schedule": previous["schedule"],
        "coverage": previous["coverage"],
        "calibration": previous["calibration"],
        "reused_endpoints": complete["endpoints"],
        "reused_review_files": complete["review_files"],
        "scope": "P46 post-hoc method pilot: two P38 small starts x two methods x all three fixed precision rulers x 128 attempts. P45 schedules and raw review reused for matched mechanism controls, not independent testing or full convergence/capacity qualification.",
        "control": "reuse exact P45 Adam and anchored-displacement endpoints, checkpoints, common exposure and verified raw reviews; no unchanged rerun. Reproduce each Adam first update on real data before reusing it as execution control.",
        "methods": "policy: original P43 six-guard direction; projected: same primary amplitude/MR bisector and six guards plus group-mean original-reference identity full-band MSE and paired below4k MSE gradients. Eight constraints use the existing certified QP. Group-mean first-order guards do not certify finite-step per-example peaks or temporal metrics.",
        "acceptance": "same radii .001 then .0003, both fit spectra must improve. Same P45 original-start relative checks for complex, MR dynamics, auxiliary budget and guard spectra. Replace PCM displacement caps with per-example original-reference RMS/peak error increases <=q=2^-bits, first-difference/boundary-difference RMS increases <=2q. Original start stays fixed; displacement is telemetry only. Retain all 16/20/24-bit sensitivity arms, no audibility or post-hoc threshold choice.",
        "history": "gradient warmup is detached; all finite-step checks recompute real warmup. New MSE derivative tests use fixed PCM coordinates, not recomputed model-state finite differences. Runtime/DSP and targets unchanged.",
        "controls_changed": "P46 versus P45 changes reference-based acceptance and adds reference temporal-error checks; policy versus projected isolates the added MSE-gradient constraints. Do not attribute the whole P45 contrast to a single removed check.",
        "selection": "finish all twelve new trajectories and freeze endpoints before any final P44 recheck. Retain rejected radii/no-ops and all seeds/budgets; do not select a winner or promote weights. Report spectrum, complex, original-target PCM and baseline/untreated comparisons. Supplement every endpoint with per-example full-identity/below4k original-reference and temporal error checks against the initial anchor at ALL three rulers. Actual Rust DSP, stereo, listening and runtime qualification remain separate.",
    }
    registered(registration, output / "registration.json")
    return registration
