"""P71 small versus approximately twofold same-depth band feasibility.

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
from models.band_capacity import TOKEN_WIDTHS, capacity_model
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def run(panel_root, prior, output):
    if output.exists():
        raise ValueError("retain prior attempts; unused output required")
    checked_receipt(panel_root / "complete.json")
    old = read(prior / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    assets = output / "assets"
    assets.mkdir()
    binary = assets / "soundex-context-replay"
    shutil.copy2(old["binary"], binary)
    for source in Path(old["assets"]).glob("input-*.sxa"):
        shutil.copy2(source, assets / source.name)
    hashes = {str(panel_root / "complete.json"): sha256_file(panel_root / "complete.json")}
    hashes.update(
        {
            str(p): sha256_file(p)
            for d in (assets, output / "source")
            for p in d.rglob("*")
            if p.is_file()
        }
    )
    registered(
        {
            "study": "p71-joint-band-capacity-feasibility",
            "panel": str(panel_root),
            "seed": 71,
            "token_widths": TOKEN_WIDTHS,
            "training_updates": 0,
            "scope": "Fresh nonzero stress fixtures, all parameters connected for training. Same 6 temporal dilations, 8 bands, 96 readout, 32 phase hidden, native support and adjacent inputs. Width24->67 gives204418->408023 parameters. No learned quality or convergence claim.",
            "changed_hypothesis": "P23/P26 scaled an older global representation; P69 froze magnitude. Screen the P63 band representation with joint trainability, holding inference depth/history/output geometry fixed. No old 640-wide route restart.",
            "timing_protocol": "Each model: 44.1/48 kHz mono/stereo, 100 warmup +3000 measured unpaced hops; 48k stereo 100+1000 paced. One pass, retain every miss. <5ms compute AND each hop deadline; fixed256-frame buffer excluded. No actual device qualification.",
            "machine": platform.platform(),
            "hashes": hashes,
        },
        output / "registration.json",
    )
    verify(hashes)
    exports, graphs, parity = {}, {}, {}
    for kind in TOKEN_WIDTHS:
        model = capacity_model(kind, nonzero=True)
        directory = output / "exports" / kind
        exports[kind] = export(model, directory)
        exports[kind]["state_values_per_channel"] = model.state_size
        graphs[kind] = directory / "diagnostic.onnx"
        torch.save(
            {"kind": kind, "seed": 71, "nonzero_stress": True, "model": model.state_dict()},
            directory / "fixture.pth",
        )
        parity[kind] = [
            native_parity(model, graphs[kind], binary, output / "parity" / kind / str(rate), rate)
            for rate in (44100, 48000)
        ]
        print("capacity export/native parity", kind, flush=True)
    schedule = [
        {"kind": k, "rate": rate, "channels": channels, "paced": False}
        for rate in (44100, 48000)
        for channels in (1, 2)
        for k in TOKEN_WIDTHS
    ]
    schedule += [{"kind": k, "rate": 48000, "channels": 2, "paced": True} for k in TOKEN_WIDTHS]
    graph_hashes = {str(p): sha256_file(p) for p in graphs.values()}
    registered({"schedule": schedule, "graphs": graph_hashes}, output / "timing-registration.json")
    results = []
    for i, spec in enumerate(schedule):
        before = os.getloadavg()
        result = cost(
            binary,
            graphs[spec["kind"]],
            assets / f"input-{spec['rate']}-{spec['channels']}.sxa",
            output / "timing" / f"{i:02d}",
            paced=spec["paced"],
        )
        results.append(
            {**spec, **result, "host_load_before": before, "host_load_after": os.getloadavg()}
        )
        print("capacity timing", i + 1, "/10", flush=True)
    verify(hashes)
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        run(args.panel.resolve(), args.prior.resolve(), args.output.resolve())
    except Exception:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
