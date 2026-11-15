"""Read-only physical P77 displacement attribution; no training or rendering.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.summarize_phase_routing import phase_delta


def summarize(root, output):
    if output.exists():
        raise ValueError("fresh read-only review required")
    output.mkdir(parents=True)
    source = output / Path(__file__).name
    source.write_bytes(Path(__file__).read_bytes())
    config = read(root / "registration.json")
    prior = Path(config["prior"])
    checked_receipt(root / "complete.json")
    checked_receipt(prior / "complete.json")
    frozen = read(root / "frozen.json")["endpoints"]
    results = {}
    for origin in config["endpoints"]:
        for candidate, control in (
            ("gain64_realized_phase", "gain64_old"),
            ("gain64_old", "unit_old"),
            ("gain64_realized_phase", "unit_realized_phase"),
        ):
            states = [
                torch.load(
                    frozen[origin + "/" + variant]["path"],
                    map_location="cpu",
                    weights_only=False,
                )["model"]
                for variant in (control, candidate)
            ]
            parameters = {}
            for prefix in ("base.", "phase_head.0.", "phase_head.2."):
                difference = torch.cat(
                    [
                        (v.double() - states[0][key].double()).flatten()
                        for key, v in states[1].items()
                        if key.startswith(prefix)
                    ]
                )
                parameters[prefix] = dict(
                    l2=float(difference.norm()),
                    peak=float(difference.abs().max()),
                    changed=int(torch.count_nonzero(difference)),
                )
            cases = []
            for draw in config["native_draws"]:
                for role in range(10):
                    paths = [
                        (prior if variant.startswith("unit_") else root)
                        / "native"
                        / origin
                        / variant.removeprefix("unit_")
                        / str(draw)
                        / str(role)
                        for variant in (control, candidate)
                    ]
                    (a, rate), (b, other_rate) = [read_sxa(p / "output.sxa") for p in paths]
                    if rate != other_rate or a.shape != b.shape:
                        raise ValueError("native timeline mismatch")
                    delta = (b.astype(np.float64) - a.astype(np.float64))[89 * 128 : -128]
                    case = dict(
                        draw=draw,
                        role=role,
                        pcm_rms=float(np.sqrt(np.mean(delta**2))),
                        pcm_peak=float(np.max(np.abs(delta))),
                        changed=int(np.count_nonzero(delta)),
                        samples=delta.size,
                    )
                    if draw == config["fit"][0] and role == 0:
                        case["phase_radians"] = phase_delta(
                            [read(p / "runtime.json")["traces"] for p in paths], rate
                        )
                    cases.append(case)
            results[origin + "/" + candidate + "-vs-" + control] = dict(
                canonical_parameters=parameters, native_cases=cases
            )
    registered(
        dict(
            results=results,
            scope="Post-completion physical canonical parameter, retained native PCM and one fit trace phase differences. Not restoration error, audible quality, a new selection rule or new rendering. All origins/cases retained.",
        ),
        output / "assessment.json",
    )
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in (
                    root / "complete.json",
                    prior / "complete.json",
                    source,
                    output / "assessment.json",
                )
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh = not args.output.exists()
    try:
        summarize(args.root.resolve(), args.output.resolve())
    except BaseException:
        if fresh and args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
