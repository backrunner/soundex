#!/usr/bin/env python3
"""Resume P49's frozen proposal review after a recorded execution-only correction.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.reporting import sha256_file, write_json_report
from scripts.audit_adapter_scale import recheck
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.summarize_adapter_scale import summarize


def run(args):
    torch.set_num_threads(1)
    calibration, output = args.calibration.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(calibration / "registration.json")
        entries = read(calibration / "proposal-registration.json")
        expected = {
            f"{s}-{a}-{g['index']:03d}"
            for s in config["starts"]
            for a in config["supports"]
            for g in config["groups"]
        }
        if set(entries) != expected or len(entries) != 64:
            raise ValueError("complete frozen proposal barrier required")
        root = Path(__file__).resolve().parents[1]
        registration = output / "registration.json"
        if registration.exists():
            repair = read(registration)
            if repair["calibration"] != str(calibration):
                raise ValueError("calibration changed")
            verify(repair["hashes"])
        else:
            hashes = {}
            for name, expected_hash in config["hashes"].items():
                path = Path(name)
                # The original code is preserved in its registered source snapshot.
                frozen = (
                    calibration / "source" / path.relative_to(root)
                    if path.is_relative_to(root)
                    else path
                )
                if not frozen.exists():
                    frozen = path
                if sha256_file(frozen) != expected_hash:
                    raise ValueError(f"original frozen input drift: {frozen}")
                hashes[str(frozen)] = expected_hash
            for label, entry in entries.items():
                path = Path(entry["path"])
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError("original proposal drift")
                proposal = torch.load(path, map_location="cpu", weights_only=False)
                if proposal["signature"] != digest({"config": config, "label": label}):
                    raise ValueError("proposal registration mismatch")
                hashes[str(path)] = entry["sha256"]
            sources = [
                p
                for p in root.rglob("*.py")
                if not any(
                    part in {".venv", "__pycache__", "outputs"}
                    for part in p.relative_to(root).parts
                )
            ]
            sources += list((root / "evaluation").glob("*.json"))
            for p in sources:
                hashes[str(p)] = sha256_file(p)
                target = output / "source" / p.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            for p in (
                calibration / "registration.json",
                calibration / "proposal-registration.json",
                calibration / "execution-control.json",
                calibration / "execution-failure.json",
            ):
                hashes[str(p)] = sha256_file(p)
            repair = {
                "calibration": str(calibration),
                "hashes": hashes,
                "scope": "Review-only execution correction: pathlib receipt suffix requires leading dot. All 64 original proposals and selected scales reused unchanged; no guard scores existed before the correction. Original source snapshot, failure and registration preserved. No optimizer or model/metric change.",
            }
            registered(repair, registration)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            print("P49 frozen review complete", flush=True)
            return
        paths = [Path(p) for p in config["hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            output / "feature-cache",
            capacity=320,
        )
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(schedule["specs"], clips)
        reports = recheck(config, entries, pool, schedule, output)
        write_json_report(summarize(config, entries, reports), output / "summary.json")
        verify(repair["hashes"])
        files = [output / "summary.json", *(output / "review").glob("*.json")]
        registered(
            {
                "registration_sha256": sha256_file(registration),
                "files": {str(p): sha256_file(p) for p in files},
            },
            output / "complete.json",
        )
        write_json_report(
            {"stage": "complete", "probes": 64, "calibration": str(calibration)},
            output / "progress.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("calibration", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
