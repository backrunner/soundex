"""P68 equal-width cross-frequency phase feasibility; no optimization or ranking.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
import platform
import shutil
import traceback
from pathlib import Path

import torch

from evaluation.adapter_runtime import cost, export
from evaluation.reporting import sha256_file, write_json_report
from models.band_phase import BandPhaseCorrection
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_band_phase_pilot import model_for
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def prepare(prior, audit, binary, inputs, output):
    if output.exists():
        raise ValueError("fresh directory required; retain every attempt")
    checked_receipt(prior / "complete.json")
    checked_receipt(audit / "complete.json")
    old = read(prior / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    assets = output / "assets"
    assets.mkdir()
    shutil.copy2(binary, assets / "soundex-context-replay")
    for rate in (44100, 48000):
        for channels in (1, 2):
            name = f"input-{rate}-{channels}.sxa"
            shutil.copy2(inputs / name, assets / name)
    hashes = dict(old["hashes"])
    hashes.update(read(audit / "complete.json")["files"])
    for directory in (output / "source", assets):
        hashes.update({str(p): sha256_file(p) for p in directory.rglob("*") if p.is_file()})
    registered(
        {
            "study": "p68-cross-frequency-phase-feasibility",
            "parents": old["parents"],
            "seeds": old["seeds"],
            "observations": ["temporal_repeat", "adjacent"],
            "training_updates": 0,
            "fixture": "Frozen P63 magnitude base; matched phase-head initialization at seed+68; final weight normal(0,0.05), bias 0.2. Native output support. Untrained stress, not quality evidence.",
            "controls": "Repeated temporal features add the same 8256 parameters without new information; may change optimization conditioning. Adjacent current-bin Cartesian products add information, no extra history/lookahead.",
            "compute_limit_ms": 5,
            "hop_frames": 128,
            "fixed_buffer_frames": 256,
            "warmup_hops": 100,
            "unpaced_scored_hops": 3000,
            "paced_scored_hops": 1000,
            "cases": "Each of four graphs: 44.1/48 kHz x mono/stereo unpaced; 48 kHz stereo paced. 20 cases, 52000 scored hops. No retry-to-pass. Synthetic pacing, not physical-device certification.",
            "machine": platform.platform(),
            "processor": platform.machine(),
            "binary_origin": str(binary),
            "inputs_origin": str(inputs),
            "hashes": hashes,
        },
        output / "registration.json",
    )
    print("registered four nonzero stress graphs and twenty timing cases", flush=True)


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "exports").exists():
        raise ValueError("retain existing exports; no silent restart")
    binary = output / "assets/soundex-context-replay"
    graphs, exports, parity = {}, {}, {}
    for seed in config["seeds"]:
        for mode in config["observations"]:
            label = f"{seed}-{mode}"
            base = model_for(config["parents"][str(seed)], seed).base
            model = BandPhaseCorrection(
                base, seed=seed + 68, phase_support="native", phase_observation=mode
            ).eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 68)
                model.phase_head[-1].weight.normal_(0, 0.05)
                model.phase_head[-1].bias.fill_(0.2)
            directory = output / "exports" / label
            exports[label] = export(model, directory)
            graphs[label] = directory / "diagnostic.onnx"
            torch.save(
                {"model": model.state_dict(), "seed": seed, "observation": mode},
                directory / "fixture.pth",
            )
            parity[label] = [
                native_parity(
                    model, graphs[label], binary, output / "parity" / label / str(rate), rate
                )
                for rate in (44100, 48000)
            ]
            print("export and native parity complete", label, flush=True)
    schedule = [
        {"label": label, "rate": rate, "channels": channels, "paced": False}
        for rate in (44100, 48000)
        for channels in (1, 2)
        for label in graphs
    ]
    schedule.extend(
        {"label": label, "rate": 48000, "channels": 2, "paced": True} for label in graphs
    )
    graph_hashes = {str(p): sha256_file(p) for p in graphs.values()}
    registered({"schedule": schedule, "graphs": graph_hashes}, output / "timing-registration.json")
    results = []
    for i, spec in enumerate(schedule):
        before = os.getloadavg()
        result = cost(
            binary,
            graphs[spec["label"]],
            output / "assets" / f"input-{spec['rate']}-{spec['channels']}.sxa",
            output / "timing" / f"{i:02d}",
            paced=spec["paced"],
        )
        results.append(
            {**spec, **result, "host_load_before": before, "host_load_after": os.getloadavg()}
        )
        print(f"timing {i + 1}/{len(schedule)} complete", flush=True)
    verify(config["hashes"])
    verify(graph_hashes)
    registered(
        {
            "exports": exports,
            "native_parity": parity,
            "timing": results,
            "training_updates": 0,
            "all_compute_under_5ms": all(r["stages"]["total_ms"]["over_5ms"] == 0 for r in results),
            "all_service_under_hop": all(
                r["stages"]["total_ms"]["service_over_hop"] == 0 for r in results
            ),
            "all_paced_completions_on_time": all(
                r["paced_late_hops"] == 0 for r in results if r["paced"]
            ),
        },
        output / "assessment.json",
    )
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.name != "run.log"
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.audit.resolve(),
                args.binary.resolve(),
                args.inputs.resolve(),
                args.output.resolve(),
            )
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
