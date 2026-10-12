#!/usr/bin/env python3
"""P8 matched objective study; both arms use the exact P7 additive energy model."""

import argparse
from pathlib import Path

from models.factorized_envelope import factor_objective
from models.linear_objective import ARMS, SMOOTHING, linear_objective, make_linear_model
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
        model_factory=make_linear_model,
        study_format="soundex-linear-study-v1",
        checkpoint_format="linear-envelope-v1",
        schedule_offset=60000,
        candidate="linear_reconstruction",
        training_objectives={
            "energy_control": factor_objective,
            "linear_reconstruction": linear_objective,
        },
        selection_objective=linear_objective,
        objective_metadata={
            "selection_objective": "linear-amplitude-dynamics-v1",
            "training_objectives": {
                "energy_control": "band-envelope-shape-v1",
                "linear_reconstruction": "linear-amplitude-dynamics-v1",
            },
            "weights": {"linear_reconstruction": 1.0, "linear_dynamics": 0.25},
            "norm_smoothing": SMOOTHING,
            "diagnostic_group_rule": "recording-paired upper CI <= 1e-6 for every guard; otherwise unsupported; never production qualification",
            "weak_group": "reference clip-band RMS <= existing reference floor; no input oracle",
        },
        experiment={
            "hypothesis": "linear reconstruction ranking reduces excess energy without sacrificing restoration",
            "contrasts": [["linear_reconstruction", "energy_control"]],
            "changed": "training objective only; common new selection metric and separate diagnostic group guards",
            "unchanged": "P7 additive energy model, initialization, features, 80-hop warmup, optimizer and budget",
        },
    )
