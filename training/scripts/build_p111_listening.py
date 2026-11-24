"""P111 blinded listening pack from native renders (compressed and clean-input trials).

The plan (conditions, seed rule, trial/case allocation) is written before any
audio. Seed 1 is used by the fixed lowest-registered-seed rule; it is also the
seed that failed P111's relative clean-preservation criterion. The answer key
stays outside the listener folder. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.longform_listening import write_listener_files, write_trial  # noqa: E402
from evaluation.reporting import write_json_report  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from scripts import run_envelope_p108 as p108  # noqa: E402

SEED = 111
CODECS = ("mp3-cbr-64", "aac-lc-cbr-96", "vorbis-vbr-q3")
CLEAN_TRIALS = 4
P110 = p108.REPO / "data/runs/envelope-native-p110-20261124-v2"
P111 = p108.REPO / "data/runs/envelope-gate-p111-20261124"
# label -> (render path template, lag); {case} is the panel case id.
NATIVE = {
    "p74_parent": (str(p108.PANEL / "renders/{case}/20261013-magnitude/output.sxa"), 128),
    "envelope_ungated": (str(P110 / "renders/{case}/seed1/output.sxa"), 0),
    "envelope_gated": (str(P111 / "renders/{case}/seed1/output.sxa"), 0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = {c["id"]: c for c in json.loads((p108.PANEL / "cases.json").read_text())}
    sources = sorted({c["source"] for c in cases.values() if c["role"] == "paired"})
    rng = np.random.default_rng(SEED)
    order = [sources[int(i)] for i in rng.permutation(len(sources))]
    plan = []
    for i, source in enumerate(order):
        codec = CODECS[i % len(CODECS)]
        (case,) = [c for c in cases.values() if c["source"] == source and c["codec"] == codec]
        plan.append({"case": case["id"], "kind": "compressed"})
    for source in [order[int(i)] for i in rng.choice(len(order), CLEAN_TRIALS, replace=False)]:
        (case,) = [c for c in cases.values() if c["source"] == source and c["role"] == "identity"]
        plan.append({"case": case["id"], "kind": "clean"})
    trial_order = [plan[int(i)] for i in rng.permutation(len(plan))]
    write_json_report(
        {
            "study": "p111-listening",
            "seed": SEED,
            "envelope_seed": 1,
            "conditions": {"compressed": ["untreated", *NATIVE], "clean": list(NATIVE)},
            "always": ["hidden_reference", "anchor_3500hz"],
            "trials": trial_order,
            "note": "Allocation by seeded work order; no case or seed chosen by score.",
        },
        a.out / "frozen-plan.json",
    )
    begin = 89 * 128 - 128
    trials, keys = [], {}
    directory = a.out / "listening"
    for item in trial_order:
        case = cases[item["case"]]
        dry, rate = read_sxa(case["input"])
        ref, _ = read_sxa(case["ref"])
        n = rate * 10
        conditions = {} if item["kind"] == "clean" else {"untreated": dry[begin : begin + n]}
        for label, (template, lag) in NATIVE.items():
            wave, _ = read_sxa(template.format(case=case["id"]))
            conditions[label] = wave[begin + lag : begin + lag + n]
        conditions = {k: v.astype(np.float32) for k, v in conditions.items()}
        trial, key = write_trial(
            directory / f"trial-{len(trials) + 1:02d}",
            ref[begin : begin + n].astype(np.float32),
            conditions,
            rate,
            rng,
        )
        trials.append(trial)
        keys[trial["id"]] = {
            **key,
            "case": case["id"],
            "kind": item["kind"],
            "source": case["source"],
        }
    write_json_report(keys, a.out / "listening-key.json")
    registration = json.loads((p108.PANEL / "registration.json").read_text())
    write_listener_files(directory, trials, registration["sources"], "soundex-p111-blinded-v2")
    print(json.dumps({"trials": len(trials), "pack": str(directory / "index.html")}))


if __name__ == "__main__":
    main()
