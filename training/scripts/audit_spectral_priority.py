#!/usr/bin/env python3
"""P43 post-hoc direction control on frozen P42 training roles, not new testing.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_priority_probe import directions_for
from models.restoration_head import load_restoration_head
from scripts.audit_pooled_updates import probe
from scripts.run_consistency_followup import read, registered


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    prior, output = args.prior.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("fresh output required")
    previous = read(prior / "registration.json")
    old_results = read(prior / "summary.json")
    if old_results["status"] != "complete":
        raise ValueError("P42 must finish before the changed hypothesis is registered")
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    # Bind all P42 evidence. Mutable diagnostic scripts are separately rebound
    # below; original executed versions remain in P42/source, never overwritten.
    for name, digest in previous["hashes"].items():
        if Path(name).is_relative_to(root / "scripts"):
            snapshot = prior / "source" / Path(name).name
            if sha256_file(Path(name)) != digest:
                if not snapshot.exists() or sha256_file(snapshot) != digest:
                    raise ValueError(f"missing historical source: {name}")
                hashes[str(snapshot)] = digest
                continue
        hashes[name] = digest
    files = [
        Path(__file__),
        root / "evaluation/spectral_priority_probe.py",
        root / "evaluation/batch_update_probe.py",
        root / "scripts/audit_pooled_updates.py",
    ]
    for p in [*files, prior / "registration.json", prior / "summary.json"]:
        hashes[str(p.resolve())] = sha256_file(p)
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift: {name}")
    study = next(Path(p).parent for p in hashes if p.endswith("study/review-registration.json"))
    config = read(study / "registration.json")
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for p in files:
        shutil.copy2(p, output / "source" / p.name)
    registered(
        {
            "scope": "P43 post-hoc spectrum-priority mechanism control, reusing P42 TRAIN roles; not independent transfer evidence or saved training",
            "hashes": hashes,
            "groups": previous["groups"],
            "endpoints": previous["endpoints"],
            "directions": ["pooled_spectrum", "pooled_spectrum_guarded"],
            "direction": "unit-normalized raw amplitude plus MR gradients averaged over four fit draws; guarded variant projects against amplitude, MR, dynamics, identity MR, identity risk and budget. No complex bisector or complex first-order guard.",
            "unchanged_assessment": "exact P42 four radii and strict/0.1% finite-step rules, including complex and original total checks; identical history and PCM metrics. No tuning after P43 scores.",
            "decision": "compare all endpoints with retained P42 controls, all-case/no-op and accepted-only metrics; no formal training or deployment promotion from these local probes",
        },
        output / "registration.json",
    )
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    reports = {}
    for seed, groups in previous["groups"].items():
        schedule = read(Path(config["schedule_files"][seed]))
        pool = LazyPool(schedule["specs"], clips)
        label = f"control-{config['steps']}-{seed}"
        checkpoint = torch.load(
            previous["endpoints"][label]["path"], map_location="cpu", weights_only=False
        )
        model = load_restoration_head(checkpoint).eval()
        cases = []
        for index, indices in enumerate(groups):
            old = old_results["cases"][label][index]
            case = probe(
                model, [pool.select(schedule["draws"][i]) for i in indices], directions_for
            )
            if case["base"] != old["base"] or case["base_transfer_pcm"] != old["base_transfer_pcm"]:
                raise ValueError("baseline changed between P42 and P43")
            cases.append({"indices": indices, "samples": old["samples"], **case})
            write_json_report(cases, output / f"{label}.json")
            print(label, indices[0], "complete", flush=True)
        reports[label] = cases
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift after probe: {name}")
    write_json_report(
        {
            "status": "complete",
            "registration_sha256": sha256_file(output / "registration.json"),
            "cases": reports,
            "coverage": old_results["coverage"],
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
