"""Post-completion P80 fixed-fit objective review, never a new selection gate.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import balanced_risk, fused_spectrum_loss, phase_objective
from models.multires_spectrum import multires_spectrum_loss
from models.phase_relations import phase_relations
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_relations import ARMS
from scripts.audit_phase_routing import add_records
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import fresh


@torch.no_grad()
def run(root, output):
    if output.exists():
        raise ValueError("fresh descriptive review required")
    checked_receipt(root / "complete.json")
    output.mkdir(parents=True)
    snapshot(output)
    config = read(root / "registration.json")
    hashes = {**config["hashes"], **read(root / "complete.json")["files"]}
    hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "hashes": hashes,
            "protocol": "Post-completion descriptive fixed-fit review of all16 P80 endpoints and4 original parents on all8 fit draws. No optimizer, selection, extra native render or new held-out scoring. Evaluate both phase-relation definitions and the unchanged common objective in FP64 on lifted actual FP32 predictions and original references. Compare each changed endpoint against matched old and parent; this cannot alter P80 gates or establish audible gains.",
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
    endpoints = {
        **read(root / "frozen.json")["endpoints"],
        **{k + "/parent": v for k, v in config["endpoints"].items()},
    }
    records = []
    for name, entry in endpoints.items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = fresh(saved["kind"], saved["seed"]).eval()
        model.load_state_dict(saved["model"], strict=True)
        for draw, (objective, context) in items.items():
            prediction, _ = predict_budget(model, objective.batch, context)
            prediction = prediction.double()
            batch = {
                k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in objective.batch.items()
            }
            values = {}
            for kind in ("circular", "wrapped"):
                terms = phase_relations(prediction, batch, kind)
                values.update({kind + "/" + k: v for k, v in terms["components"].items()})
                values[kind + "/mean"] = terms["per_example"]
                values[kind + "/weighted_auxiliary"] = 0.1 * terms["per_example"].log1p()
            values["common_risk"] = sum(
                loss(prediction, batch)["per_example"].log1p()
                for loss in (direct_spectrum_loss, multires_spectrum_loss, fused_spectrum_loss)
            )
            torch.testing.assert_close(
                balanced_risk(values["common_risk"], batch["identity"]),
                phase_objective(prediction, batch, include_fused=True),
                rtol=1e-12,
                atol=1e-14,
            )
            add_records(
                records,
                config,
                schedule,
                rows,
                draw,
                name,
                [{k: float(v[i]) for k, v in values.items()} for i in range(len(prediction))],
            )
        print("reviewed", name, flush=True)
    registered(records, output / "records.json")
    comparisons = {}
    for origin, role, corpus in itertools.product(
        config["endpoints"], ("paired", "identity"), ("all", "music_library", "speech_library")
    ):
        selected = [
            r for r in records if r["role"] == role and (corpus == "all" or r["corpus"] == corpus)
        ]
        for candidate, control in [(a, "old") for a in ARMS[1:]] + [(a, "parent") for a in ARMS]:
            comparisons[f"{origin}/{role}/{corpus}/{candidate}-vs-{control}"] = paired_recordings(
                selected, origin + "/" + candidate, origin + "/" + control, seed=80
            )
    registered(
        {
            "comparisons": comparisons,
            "qualified": False,
            "scope": "Descriptive fixed-fit risk, FP64 analysis of FP32 predictions. Different definition totals are not compared against one another; no new gate, unseen test, listening or inference result.",
        },
        output / "assessment.json",
    )
    verify(hashes)
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.log" and "__pycache__" not in p.parts
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh_directory = not args.output.exists()
    try:
        run(args.root.resolve(), args.output.resolve())
    except BaseException:
        if fresh_directory and args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
