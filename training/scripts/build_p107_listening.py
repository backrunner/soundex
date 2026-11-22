"""P107 blinded listening pack: current models versus parametric envelope oracles.

Conditions and codec rotation are fixed here before any audio is written; the
answer key stays outside the listener folder. Oracles use the lossless reference
and are not deployable models. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import envelope_oracle  # noqa: E402
from evaluation.longform_listening import write_listener_files, write_trial  # noqa: E402
from evaluation.reporting import write_json_report  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from scripts.score_missing_band import BURN, HOP, NATIVE_LAG, SECONDS  # noqa: E402

SEED = 107
CODECS = ("mp3-cbr-64", "aac-lc-cbr-96", "vorbis-vbr-q3")
NATIVE = ("20261013-magnitude", "852/shape/20261013-magnitude", "legacy")
ORACLES = ("oracle_patch_erb", "oracle_patch_level")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="P107 v3 run (spectral renders)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = {c["id"]: c for c in json.loads((args.panel / "cases.json").read_text())}
    registration = json.loads((args.panel / "registration.json").read_text())
    sources = sorted({c["source"] for c in cases.values() if c["role"] == "paired"})
    rng = np.random.default_rng(SEED)
    order = [sources[int(i)] for i in rng.permutation(len(sources))]
    plan = []
    for i, source in enumerate(order):
        codec = CODECS[i % len(CODECS)]
        match = [c for c in cases.values() if c["source"] == source and c["codec"] == codec]
        if len(match) != 1:
            raise ValueError("case plan must select exactly one case per work")
        plan.append(match[0]["id"])
    frozen = {
        "study": "p107-listening",
        "seed": SEED,
        "conditions": ["untreated", *NATIVE, "spectral_dsp", *ORACLES],
        "trials": plan,
        "note": "One trial per work, codec rotated by seeded work order; no case selected by score.",
    }
    write_json_report(frozen, args.out / "frozen-plan.json")
    begin, n = BURN * HOP - HOP, None
    trials, keys = [], {}
    directory = args.out / "listening"
    for case_id in plan:
        case = cases[case_id]
        dry, rate = read_sxa(case["input"])
        ref, _ = read_sxa(case["ref"])
        n = rate * SECONDS
        conditions = {"untreated": dry[begin : begin + n]}
        for label in NATIVE:
            wave, _ = read_sxa(args.panel / "renders" / case_id / label / "output.sxa")
            conditions[label] = wave[begin + NATIVE_LAG : begin + NATIVE_LAG + n]
        spectral, _ = read_sxa(args.scores / "spectral" / case_id / "output.sxa")
        conditions["spectral_dsp"] = spectral[begin : begin + n]
        for kind in ORACLES:
            conditions[kind] = envelope_oracle.render(dry, ref, rate, kind)[begin : begin + n]
        conditions = {k: v.astype(np.float32) for k, v in conditions.items()}
        trial, key = write_trial(
            directory / f"trial-{len(trials) + 1:02d}",
            ref[begin : begin + n].astype(np.float32),
            conditions,
            rate,
            rng,
        )
        trials.append(trial)
        keys[trial["id"]] = {**key, "case": case_id, "source": case["source"]}
    write_json_report(keys, args.out / "listening-key.json")
    write_listener_files(directory, trials, registration["sources"], "soundex-p107-envelope-v1")
    print(json.dumps({"trials": len(trials), "pack": str(directory)}))


if __name__ == "__main__":
    main()
