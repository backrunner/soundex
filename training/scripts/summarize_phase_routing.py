"""Read-only P76 finite-displacement attribution, using retained native PCM.

No new update, render, score threshold, checkpoint selection or runtime retry.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.reporting import sha256_file
from evaluation.rust_bridge import read_sxa
from models.losses import build_missing_band_mask
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def phase_delta(traces, rate):
    """Ordinary native traces have condition/prediction, not optional DSP stages."""
    conditions = [torch.tensor([t["condition"] for t in ts])[89:] for ts in traces]
    if not torch.equal(*conditions):
        raise ValueError("input-derived DSP policy changed")
    condition = conditions[0]
    cutoff = condition[:, 1] * rate / 2
    weights = build_missing_band_mask(
        cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
    ).reshape(-1, 129)
    support = ((weights > 0) & condition[:, 2:].bool()).numpy()
    predictions = [
        np.asarray([t["prediction"] for t in ts]).reshape(-1, 2, 129)[89:] for ts in traces
    ]
    phase = predictions[1][:, 1] - predictions[0][:, 1]
    phase = np.arctan2(np.sin(phase), np.cos(phase))[support]
    return {
        "max": float(np.max(np.abs(phase))) if phase.size else None,
        "rms": float(np.sqrt(np.mean(phase**2))) if phase.size else None,
        "bins": int(phase.size),
        "scope": "One fixed TRAIN mono role per endpoint; support reconstructed from actual native input condition using the unchanged 1kHz crossover, after89 hops. Not reference phase error or audibility.",
    }


def summarize(root, output, previous=None):
    if output.exists():
        raise ValueError("fresh read-only review directory required")
    output.mkdir(parents=True)
    source = output / Path(__file__).name
    source.write_bytes(Path(__file__).read_bytes())
    evidence = [root / "complete.json", source, output / "assessment.json"]
    if previous is not None:
        if not (previous / "failure.json").is_file():
            raise ValueError("previous failed review must remain available")
        evidence.extend(p for p in previous.rglob("*") if p.is_file())
    checked_receipt(root / "complete.json")
    config, frozen = read(root / "registration.json"), read(root / "frozen.json")
    results = {}
    for endpoint in config["endpoints"]:
        before = torch.load(
            frozen["endpoints"][endpoint + "/old"]["path"], map_location="cpu", weights_only=False
        )["model"]
        for arm in ("realized_phase", "direct_phase"):
            label = endpoint + "/" + arm
            after = torch.load(
                frozen["endpoints"][label]["path"], map_location="cpu", weights_only=False
            )["model"]
            groups = {}
            for prefix in ("base.", "phase_head."):
                delta = torch.cat(
                    [
                        (v.double() - before[k].double()).flatten()
                        for k, v in after.items()
                        if k.startswith(prefix)
                    ]
                )
                groups[prefix] = {
                    "l2": float(delta.norm()),
                    "max": float(delta.abs().max()),
                    "changed_entries": int(torch.count_nonzero(delta)),
                }
            cases = []
            for draw in config["native_draws"]:
                for role in range(10):
                    paths = [
                        root / "native" / endpoint / variant / str(draw) / str(role)
                        for variant in ("old", arm)
                    ]
                    (a, rate_a), (b, rate_b) = [read_sxa(p / "output.sxa") for p in paths]
                    if rate_a != rate_b or a.shape != b.shape:
                        raise ValueError("native comparison timeline mismatch")
                    difference = (b.astype(np.float64) - a.astype(np.float64))[89 * 128 : -128]
                    case = {
                        "draw": draw,
                        "role": role,
                        "samples": difference.size,
                        "pcm_rms": float(np.sqrt(np.mean(difference**2))),
                        "pcm_max": float(np.max(np.abs(difference))),
                        "changed_samples": int(np.count_nonzero(difference)),
                    }
                    if draw == config["fit"][0] and role == 0:
                        traces = [read(p / "runtime.json")["traces"] for p in paths]
                        case["active_prediction_phase_delta_radians"] = phase_delta(traces, rate_a)
                    cases.append(case)
            results[label] = {
                "parameters_vs_matched_old": groups,
                "native_pcm_vs_matched_old": cases,
            }
    registered(
        {
            "results": results,
            "scope": "Post-completion attribution of frozen P76 proposals against their matched old updates. No new training or rendering; all case differences retained. Tiny output changes cannot reject a phase architecture or establish listening quality.",
        },
        output / "assessment.json",
    )
    registered(
        {"files": {str(p): sha256_file(p) for p in evidence}},
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh = not args.output.exists()
    try:
        summarize(
            args.root.resolve(),
            args.output.resolve(),
            args.previous_attempt.resolve() if args.previous_attempt else None,
        )
    except BaseException:
        if fresh and args.output.exists():
            registered({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
