"""P74 matched small phase-observation feasibility; one registered timing grid.

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
from models.phase_temporal import OBSERVATIONS, phase_temporal_model
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def run(prior, output):
    if output.exists():
        raise ValueError("fresh destination required; retain failed attempts")
    checked_receipt(prior / "p72-complete.json")
    old = read(prior / "registration.json")
    sources = [Path(old["binary"]), *sorted(Path(old["assets"]).glob("input-*.sxa"))]
    verify({str(p): old["hashes"][str(p)] for p in sources})
    output.mkdir(parents=True)
    snapshot(output)
    assets = output / "assets"
    assets.mkdir()
    for p in sources:
        shutil.copy2(p, assets / p.name)
    binary = assets / "soundex-context-replay"
    hashes = {
        str(p): sha256_file(p)
        for folder in (assets, output / "source")
        for p in folder.rglob("*")
        if p.is_file()
    }
    hashes[str(prior / "p72-complete.json")] = sha256_file(prior / "p72-complete.json")
    schedule = []
    for order in (OBSERVATIONS, tuple(reversed(OBSERVATIONS))):
        for rate, channels, paced in (
            (44100, 1, False),
            (44100, 2, False),
            (48000, 1, False),
            (48000, 2, False),
            (48000, 2, True),
        ):
            schedule += [dict(kind=k, rate=rate, channels=channels, paced=paced) for k in order]
    registered(
        {
            "study": "p74-phase-temporal-feasibility",
            "prior": str(prior),
            "seed": 74,
            "arms": OBSERVATIONS,
            "training_updates": 0,
            "hashes": hashes,
            "changed_hypothesis": "P68/P69 added observations only to a post-decoder rotation head. P73 implicates amplitude/phase coupling. Inject local temporal and adjacent-bin phase relations before the small joint band TCN; same-parameter magnitude-only projection control, same original-reference objective later.",
            "scope": "Fresh nonzero stress fixtures. No quality, training convergence, production worker, callback or physical-device qualification. P72 late-hop failure remains retained.",
            "geometry": "Token24, six temporal layers, 8 bands. Additional258 previous-carrier state per physical channel; no lookahead, unchanged80-hop enhancement warmup and256-frame fixed buffer.",
            "timing_protocol": "Two predeclared opposite arm orders; each44.1/48k mono/stereo100+3000 unpaced and48k stereo100+1000 paced. 52000 measured hops; one pass, every failure retained. All<5ms compute, all service<hop, zero paced misses required before full training. No retry-to-pass.",
            "schedule": schedule,
            "machine": platform.platform(),
        },
        output / "registration.json",
    )
    verify(hashes)
    exports, graphs, parity = {}, {}, {}
    for kind in OBSERVATIONS:
        model = phase_temporal_model(kind, nonzero=True)
        directory = output / "exports" / kind
        exports[kind] = export(model, directory)
        exports[kind]["state_values_per_channel"] = model.state_size
        graphs[kind] = directory / "diagnostic.onnx"
        torch.save(
            {"kind": kind, "seed": 74, "model": model.state_dict(), "nonzero_stress": True},
            directory / "fixture.pth",
        )
        parity[kind] = [
            native_parity(model, graphs[kind], binary, output / "parity" / kind / str(rate), rate)
            for rate in (44100, 48000)
        ]
        print("export/native parity", kind, flush=True)
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
        print("timing", i + 1, "/", len(schedule), flush=True)
    verify(hashes)
    verify(graph_hashes)
    registered(
        {
            "exports": exports,
            "native_parity": parity,
            "timing": results,
            "all_compute_under_5ms": all(r["stages"]["total_ms"]["over_5ms"] == 0 for r in results),
            "all_service_under_hop": all(
                r["stages"]["total_ms"]["service_over_hop"] == 0 for r in results
            ),
            "all_paced_completions_on_time": all(
                r["paced_late_hops"] == 0 for r in results if r["paced"]
            ),
            "training_updates": 0,
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
    print("complete", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        run(args.prior.resolve(), args.output.resolve())
    except Exception:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
