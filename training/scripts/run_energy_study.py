#!/usr/bin/env python3
"""P7 matched energy-coordinate experiment, unchanged P6 objective and capacity."""

import argparse
from pathlib import Path

from models.residual_energy import ARMS, make_energy_model
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
        model_factory=make_energy_model,
        study_format="soundex-energy-study-v1",
        checkpoint_format="energy-envelope-v1",
        schedule_offset=50000,
        candidate="rms_residual",
        experiment={
            "hypothesis": "input-anchored multiplicative energy impedes missing-band optimization",
            "contrasts": [["rms_residual", "factorized"]],
            "changed": "band-energy coordinates only; same shared initialization and parameter count",
            "unchanged": "P6 objective, shape head, features, optimizer and 2048-update budget",
        },
    )
