#!/usr/bin/env python3
"""P10 matched raw/rendered supervision and identity-risk weighting."""

import argparse
from pathlib import Path

from models.linear_objective import SMOOTHING, linear_objective
from models.rendered_objective import RenderedObjective
from models.rendered_probe import ARMS, make_rendered_model
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
        model_factory=make_rendered_model,
        study_format="soundex-rendered-study-v1",
        checkpoint_format="rendered-envelope-v1",
        schedule_offset=80000,
        candidate="rendered_balanced",
        training_objectives={
            "raw_control": linear_objective,
            "rendered_control": RenderedObjective(1),
            "rendered_balanced": RenderedObjective(7),
        },
        selection_objective=RenderedObjective(7),
        objective_metadata={
            "summary_scope": "rendered training proxy and raw amplitude diagnostics; native DSP, listening and realtime qualification remain separate",
            "selection_objective": "rendered-amplitude-waveform-identity-balanced-v1",
            "deployment_phase": "input",
            "training_objectives": {
                "raw_control": "linear-amplitude-dynamics-v1",
                "rendered_control": "rendered-amplitude-waveform-v1",
                "rendered_balanced": "rendered-amplitude-waveform-identity-balanced-v1",
            },
            "guard_controls": ["raw_control", "rendered_control"],
            "dsp_identity_controls": True,
            "weights": {"rendered_linear": 1.0, "rendered_waveform": 0.25, "identity_risk": 7.0},
            "norm_smoothing": SMOOTHING,
            "diagnostic_group_rule": "recording-paired upper CI <= 1e-6 for every guard; otherwise unsupported; never production qualification",
            "weak_group": "reference clip-band RMS <= existing reference floor; no input oracle",
        },
        experiment={
            "hypothesis": "supervising rendered output and balanced preservation risk improve actual audio without rewarding bypass",
            "contrasts": [
                ["rendered_control", "raw_control"],
                ["rendered_balanced", "rendered_control"],
                ["rendered_balanced", "raw_control"],
            ],
            "changed": "rendered versus raw objective, then identity risk weight 1 versus 7; all evaluated with common balanced rendered score",
            "unchanged": "P9 anchored model, initialization, past-only features, input phase, draw ratios, optimizer, 80-hop warmup and budget",
        },
    )
