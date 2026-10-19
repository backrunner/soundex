#!/usr/bin/env python3
"""P32 fixed small architecture: multi-resolution spectral supervision first."""

import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace

from evaluation.multires_review import review as spectrum_review
from evaluation.reporting import write_json_report
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import (
    ABSOLUTE_POWER_FLOOR,
    FFT_SIZES,
    RELATIVE_POWER_FLOOR,
    VERSION,
    WEIGHTS,
    multires_spectrum_loss,
)
from scripts.bench_context_deployment import run as bench_native
from scripts.run_direct_spectrum_study import MODEL, objective_terms
from scripts.run_direct_spectrum_study import run as run_fit
from scripts.run_head_study import progress
from scripts.run_phase_coordinates_study import register as register_parent

ARMS = ("control", "mr_add", "mr_primary")


def protocol(parent, *, preflight=False):
    steps = 4 if preflight else 1024
    return {
        **{k: parent[k] for k in ("specs", "weight", "rate", "direct_loss", "optimization")},
        "format": "soundex-spectrum-priority-fit-v1",
        "scope": "fixed P29/P31 TRAINING crops; objective learnability diagnosis, not full corpus training, convergence or generalization; no promotion",
        "review_scope": "all fixed endpoints on fitted training crops; spectrum-first descriptive assessment; PCM auxiliary, not sole veto",
        "arms": list(ARMS),
        "seeds": parent["seeds"][:1] if preflight else parent["seeds"],
        "steps": steps,
        "primary_endpoint": steps,
        "smoke": preflight,
        "model_configs": {arm: dict(MODEL) for arm in ARMS},
        "loss_routes": {
            "control": {"raw": 1.0, "budget": 1.0},
            "mr_add": {"raw": 1.0, "multires": 1.0, "budget": 1.0},
            "mr_primary": {"multires": 1.0, "complex_guard": 1.0, "budget": 1.0},
        },
        "multires_loss": {
            "version": VERSION,
            "fft_sizes": list(FFT_SIZES),
            "weights": WEIGHTS,
            "relative_power_floor": RELATIVE_POWER_FLOOR,
            "absolute_power_floor": ABSOLUTE_POWER_FLOOR,
            "low_band_weight": 0.25,
            "high_band_weight": 1.0,
            "complex_guard_weight": 0.02,
            "region": "[89*128, frames*128-128); center=False; no padding; raw OLA vs original clean PCM",
        },
        "development": parent["specs"],
        "shifted": [],
        "paced_arms": list(ARMS),
        "contrasts": "control->mr_add isolates addition of MR supervision; mr_add->mr_primary removes old direct magnitude/envelope/dynamics terms and reduces direct complex guard from .1 to .02, a bundled priority change not a pure weight-only ablation; all share architecture and initialization",
        "selection": "freeze all six endpoints before scoring; primary common MR spectrum score and all magnitude/energy/dynamics components on raw and actual native output, separately music/speech; auxiliary phase/PCM/low-band/identity/integrity disclosed; a small PCM increase alone is not a veto; no automatic promotion, subset exclusion or mid-run selection",
        "stop_policy": "nonfinite, integrity or runtime failure only; no performance stopping",
        "timing_policy": "all endpoints, same native compute benchmark after quality review; no new inference operations/buffers; measured service/deadlines/RSS retained independently of spectral merit",
    }


def register(args, root):
    return register_parent(
        args, root, make_protocol=protocol, parent_format="soundex-phase-coordinates-fit-v1"
    )


def training_terms(prediction, budget, objective, weight):
    terms = objective_terms(prediction, budget, objective, weight)
    direct = direct_spectrum_loss(prediction, objective.batch)
    identity = objective.batch["identity"].bool()
    values = direct["components"]["complex"]
    terms["complex_guard"] = 0.02 * 0.5 * (values[identity].mean() + values[~identity].mean())
    terms["multires"] = multires_spectrum_loss(prediction, objective.batch)["total"]
    return terms


def run(args):
    run_fit(
        args,
        register_study=register,
        training_terms=training_terms,
        completion_stage="native_review_complete",
    )
    if args.prepare_only:
        return
    progress(args.output, stage="multires_native_review")
    spectrum_review(args.output, args.native_binary)
    if not args.preflight:
        progress(args.output, stage="native_compute_benchmark")
        bench_native(
            SimpleNamespace(
                study=args.output, output=args.output / "native-cost", binary=args.native_binary
            )
        )
    progress(args.output, stage="complete", preflight=args.preflight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        write_json_report(
            {"stage": "failed", "error": traceback.format_exc()},
            args.output / "pipeline-error.json",
        )
        raise
