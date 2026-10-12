#!/usr/bin/env python3
"""P9 matched shape-coordinate and exact-identity initialization study."""

import argparse
from pathlib import Path

from models.conditioned_shape import ARMS, make_shape_model
from models.linear_objective import SMOOTHING, linear_objective
from scripts.run_factorized_study import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--validation-interval", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261012, 20261013])
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.validation_interval < 1:
        parser.error("positive steps and interval required")
    run(
        args,
        arms=ARMS,
        model_factory=make_shape_model,
        study_format="soundex-shape-study-v1",
        checkpoint_format="shape-envelope-v1",
        schedule_offset=70000,
        candidate="anchored_band",
        training_objectives={arm: linear_objective for arm in ARMS},
        selection_objective=linear_objective,
        objective_metadata={
            "selection_objective": "linear-amplitude-dynamics-v1",
            "training_objectives": dict.fromkeys(ARMS, "linear-amplitude-dynamics-v1"),
            "guard_controls": ["energy_conditioned", "band_conditioned"],
            "dsp_identity_controls": True,
            "weights": {"linear_reconstruction": 1.0, "linear_dynamics": 0.25},
            "norm_smoothing": SMOOTHING,
            "diagnostic_group_rule": "recording-paired upper CI <= 1e-6 for every guard; otherwise unsupported; never production qualification",
            "weak_group": "reference clip-band RMS <= existing reference floor; no input oracle",
        },
        experiment={
            "hypothesis": "band-relative shape steps and exact identity initialization reduce optimizer-induced damage while retaining restoration",
            "contrasts": [
                ["band_conditioned", "energy_conditioned"],
                ["anchored_band", "band_conditioned"],
                ["anchored_band", "energy_conditioned"],
            ],
            "changed": "shape update units; separate third arm subtracts zero-head mapping for exact identity",
            "unchanged": "input-relative energy head, parameter tensors, features, linear loss, 80-hop warmup, optimizer and budget",
        },
    )
