#!/usr/bin/env python3
"""P47 supplemental input-band control; no second model needed for the transform.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.fidelity_calibration import assess_wave
from evaluation.output_coupling import compare_outputs
from evaluation.reporting import sha256_file, write_json_report
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import supervised_waves
from models.restoration_head import load_restoration_head, predict_budget
from scripts.audit_output_coupling import aggregate
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    prior, output = args.prior.resolve(), args.output.resolve()
    config = read(prior / "registration.json")
    if output.exists():
        raise ValueError("fresh supplement directory required")
    verify(config["hashes"])
    verify(config["prior_input_hashes"])
    original_receipt = read(prior / "complete.json")
    verify({str(prior / "summary.json"): original_receipt["summary_sha256"]})
    source = Path(__file__).resolve()
    hashes = {
        str(source): sha256_file(source),
        str(prior / "registration.json"): sha256_file(prior / "registration.json"),
        str(prior / "complete.json"): sha256_file(prior / "complete.json"),
    }
    output.mkdir(parents=True)
    shutil.copy2(source, output / source.name)
    registered(
        {
            "hashes": hashes,
            "prior": str(prior),
            "scope": "post-hoc extension prompted by P47 band attribution; same two starts, all twelve candidates and both Adam controls, same 32 reused TRAIN batches; no training, tuning, selection or independent test",
            "paths": "same fixed 4k/8k boundaries, replacing candidate low bins with observed compressed input instead of the initial model; apply each transform to starts as well, retaining matched transformed-start comparisons",
            "preservation_anchor": "original compressed input for supplemental ruler counts, explicitly different from main P47 initial-model ruler; full path spectral/PCM scores must still reproduce P46 exactly",
            "deployment": "transform needs only current input/model output, but hard boundaries and native DSP/continuity/runtime have NOT been qualified",
        },
        output / "registration.json",
    )
    paths = [Path(p) for p in config["prior_input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    original = next(iter(read(Path(config["calibration"]))["cases"].values()))
    specs = [s["spec"] for c in original for s in c["samples"]["calibration"]]
    pool = LazyPool(specs, clips)
    items = [pool.select(list(range(i * 5, i * 5 + 5))) for i in range(len(original))]
    baselines = []
    for item in items:
        batch = item[0].batch
        wave, reference = supervised_waves(batch["x"], batch)
        direct = direct_spectrum_loss(batch["x"], batch)
        baselines.append(
            {
                "metrics": {
                    **assess_wave(wave, reference, batch["rate"]),
                    **{"direct_" + k: v.tolist() for k, v in direct["components"].items()},
                    **{
                        f"band_{b['hz'][0]}_{k}": b[k].tolist()
                        for b in direct["bands"]
                        for k in ("amplitude", "complex")
                    },
                }
            }
        )
    reports, raw_hashes = {}, {}
    for label, entry in config["endpoints"].items():
        model = load_restoration_head(
            torch.load(entry["path"], map_location="cpu", weights_only=False)
        ).eval()
        old = read(Path(config["prior"]) / "review" / f"{label}.json")["cases"]
        cases = []
        for i, item in enumerate(items):
            predicted, _ = predict_budget(model, item[0].batch, item[1])
            result = compare_outputs(item[0].batch["x"], predicted, item)
            if result["paths"]["full"]["metrics"] != old[i]["paths"]["frozen_model"]:
                raise ValueError("full endpoint score changed")
            cases.append(
                {"index": original[i]["index"], "samples": original[i]["samples"], **result}
            )
        path = output / "cases" / f"{label}.json"
        write_json_report(cases, path)
        raw_hashes[str(path)] = sha256_file(path)
        reports[label] = aggregate(cases, baselines)
        write_json_report(
            {"completed": len(reports), "total": len(config["endpoints"])}, output / "progress.json"
        )
        print(label, "input-band controls complete", flush=True)
    verify(hashes)
    verify(config["hashes"])
    verify(config["prior_input_hashes"])
    write_json_report(
        {
            "reports": reports,
            "raw_files": raw_hashes,
            "comparison": "all change statistics are against untreated input; compare candidate/start current_mean under same path for matched training effect",
        },
        output / "summary.json",
    )
    registered(
        {
            "summary_sha256": sha256_file(output / "summary.json"),
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
