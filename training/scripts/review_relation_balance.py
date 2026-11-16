"""P82 follow-up: actual balanced routed gradient against each corpus/role.

The original stratum-only directions are hypothetical isolated training fields.
Keep them, then measure the actual mixed objective without a model/optimizer step.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import itertools
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_routing import exact, samples
from scripts.audit_realized_relations import probe
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import load
from scripts.summarize_realized_relations import stats

FIELDS = ("balanced_head_direction_cosines", "balanced_routed_full_direction_cosines")


def assess(probes):
    out = {}
    for field, stratum in itertools.product(FIELDS, probes[0]["strata"]):
        for arm, targets in probes[0]["strata"][stratum][field].items():
            for target in targets:
                values = [p["strata"][stratum][field][arm][target] for p in probes]
                controls = [p["strata"][stratum][field]["old"][target] for p in probes]
                new_adverse, resolved = [], []
                for p, value, before in zip(probes, values, controls, strict=True):
                    if value is None or before is None:
                        continue
                    entry = {
                        "origin": p["origin"],
                        "draw": p["draw"],
                        "control_cosine": before,
                        "cosine": value,
                        "sources": [
                            s["row"]["source_checksum"]
                            for s in p["samples"]
                            if s["row"]["corpus"] == stratum.split("/")[0]
                        ],
                    }
                    if before >= 0 and value < 0:
                        new_adverse.append(entry)
                    if before < 0 and value >= 0:
                        resolved.append(entry)
                out[f"{field}/{stratum}/{arm}/{target}"] = {
                    **stats(values),
                    "new_adverse": new_adverse,
                    "resolved": resolved,
                }
    return out


def run(root, output):
    if output.exists():
        raise ValueError("fresh balance review required")
    checked_receipt(root / "complete.json")
    config = read(root / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = {**config["hashes"], **read(root / "complete.json")["files"]}
    hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "hashes": hashes,
            "changed_question": "P82 reports isolated-stratum old+auxiliary directions. Those are not the actual balanced training field. Measure balanced head and full routed (unchanged old base, auxiliary only in head) directions against each role/corpus target before recommending a finite recipe.",
            "budget": "Replay all32 original frozen fit probes exactly, adding only cross-stratum gradient dot products. Zero optimizer updates, no new native render/score, no held-out sources. Retain all40 completed native checks from the original receipt. Review every origin, domain and kernel together.",
            "gates": "Every original probe field must reproduce exactly after removing the two additive strata fields. All endpoint weights/hashes stay fixed. Counts of negative cosine are descriptive first-order Euclidean conflicts, not audible thresholds, optimizer movements or architecture verdicts.",
        },
        output / "registration.json",
    )
    verify(hashes)
    p74 = Path(config["p74"])
    old = read(p74 / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        config["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), p74 / "cache", capacity=100),
    )
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"]}
    probes = []
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = load(saved).double().eval()
        for draw in config["fit"]:
            sample = samples(schedule, rows, draw)
            value = {
                "origin": origin,
                "draw": draw,
                "samples": sample,
                **probe(model, items[draw], [s["row"]["corpus"] for s in sample] * 2),
            }
            historical = copy.deepcopy(value)
            for stratum in historical["strata"].values():
                for field in FIELDS:
                    del stratum[field]
            if historical != read(root / "probes" / origin / f"{draw}.json"):
                raise ValueError("original frozen probe did not reproduce exactly")
            registered(value, output / "probes" / origin / f"{draw}.json")
            probes.append(value)
            print("balance review", origin, draw, flush=True)
        exact(model.state_dict(), {k: v.double() for k, v in saved["model"].items()})
    registered(
        {
            "strata": assess(probes),
            "original_probes_exact": len(probes),
            "qualified": False,
            "scope": "Actual balanced Euclidean field versus isolated corpus/role losses. Persistent Adam and finite/native changes remain untested for this new recipe.",
        },
        output / "assessment.json",
    )
    verify(hashes)
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
        },
        output / "complete.json",
    )
    print("complete:32 exact replays and additive mixed-objective projections", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh = not args.output.exists()
    try:
        run(args.root.resolve(), args.output.resolve())
    except BaseException:
        if fresh and args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
