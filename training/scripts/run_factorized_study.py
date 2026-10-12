#!/usr/bin/env python3
"""P6: separate architecture from objective, with complete 80-hop warmup."""

import argparse
import fcntl
import json
from functools import partial
from pathlib import Path

import torch

from data.causal_study import digest, load_sources, make_schedule
from data.envelope_study import EnvelopeClips
from data.weak_band_study import add_identity_controls, audit_groups
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_objective import BANDS, envelope_objective
from models.factorized_envelope import (
    ABSOLUTE_FLOOR,
    ARMS,
    BURN,
    RELATIVE_FLOOR,
    factor_objective,
    make_factor_model,
)
from scripts.run_envelope_study import evaluate, run_arm
from scripts.run_weak_band_study import freeze_binary


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads(args.parent.read_text())
        rows, manifests = load_sources([Path(p) for p in parent["manifests"]])
        if manifests != parent["manifests"]:
            raise ValueError("source manifests changed")
        detector = freeze_binary(
            args.parent.parent / "binaries/soundex-gain-data",
            root / "binaries/soundex-gain-data",
            parent["detector_sha256"],
        )
        source = Path(__file__).resolve().parents[1]
        hashes = {
            str(p.relative_to(source)): sha256_file(p)
            for p in sorted(source.rglob("*.py"))
            if not any(
                x in {".venv", "tests", "outputs", "__pycache__"}
                for x in p.relative_to(source).parts
            )
        }
        specs, train_specs = parent["validation_specs"], parent["training_probe_specs"]
        if args.smoke:

            def subset(values):
                return [
                    next(s for s in values if rows[s["row_id"]]["corpus"] == corpus)
                    for corpus in ("music_library", "speech_library")
                ]

            specs, train_specs = subset(specs), subset(train_specs)
        config = {
            "format": "soundex-factor-study-v1",
            "checkpoint_format": "factor-envelope-v1",
            "steps": args.steps,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "validation_interval": args.validation_interval,
            "seeds": args.seeds,
            "arms": list(ARMS),
            "history_ablation_arms": ["factorized"],
            "context": "past",
            "burn_hops": BURN,
            "context_stride": 4,
            "selection_objective": "band-envelope-shape-v1",
            "initial_checkpoint_eligible": True,
            "training_objectives": {
                arm: "amplitude-log1p-v1" if arm == "direct_legacy" else "band-envelope-shape-v1"
                for arm in ARMS
            },
            "weights": {
                "band_envelope": 1.0,
                "band_shape": 0.5,
                "envelope_temporal": 0.25,
                "amplitude": 0.25,
            },
            "relative_floor": RELATIVE_FLOOR,
            "absolute_floor": ABSOLUTE_FLOOR,
            "bands_hz": list(map(list, BANDS)),
            "identity_selection_weight": 0.125,
            "device": args.device,
            "smoke": args.smoke,
            "validation_specs": specs,
            "training_probe_specs": train_specs,
            "manifests": manifests,
            "coverage": audit_groups(rows),
            "parent_config_sha256": sha256_file(args.parent),
            "parent_weights_used": False,
            "source_sha256": digest(hashes),
            "detector_sha256": sha256_file(detector),
            "torch_version": str(torch.__version__),
            "independent_test": False,
            "test_scored": False,
            "deployable_audio_model": False,
            "cache_path": str((args.cache or root / "cache").resolve()),
        }
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("registered study changed")
        write_json_report(config, config_path)
        write_json_report(hashes, root / "source-files.json")
        clips = EnvelopeClips(rows, detector, args.cache or root / "cache")
        if not (root / "compressed-reference.json").exists():
            write_json_report(
                evaluate(
                    None,
                    clips,
                    specs,
                    "past",
                    args.device,
                    detailed=True,
                    objective_fn=factor_objective,
                    burn=BURN,
                ),
                root / "compressed-reference.json",
            )
        completed = []
        for seed in args.seeds:
            schedule, sampling = make_schedule(rows, seed + 40000, args.steps * 4)
            draws = add_identity_controls(schedule, 4)
            write_json_report(
                {"draws": draws, "sampling_before_identity": sampling, "sha256": digest(draws)},
                root / f"draws-{seed}.json",
            )
            for arm in ARMS:
                completed.append(
                    run_arm(
                        config,
                        seed,
                        arm,
                        draws,
                        clips,
                        root,
                        model_factory=partial(make_factor_model, arm),
                        training_objective=envelope_objective
                        if arm == "direct_legacy"
                        else factor_objective,
                        selection_objective=factor_objective,
                        context_arm="past",
                        burn=BURN,
                        include_initial=True,
                    )
                )
        write_json_report(
            {
                "status": "complete",
                "arms": completed,
                "scope": "raw amplitude only; no phase/DSP/audio-quality release qualification",
            },
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", required=True, type=Path, help="P5 config supplying audited splits"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="optional shared audited input-feature cache")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--validation-interval", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261012, 20261013])
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.steps < 1 or options.validation_interval < 1:
        parser.error("positive steps and interval required")
    run(options)
