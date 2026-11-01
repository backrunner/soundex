"""Receipt-checked P67 endpoint integrity and matched-population summary.

Derives reports from frozen scores; no rerender, optimizer update or selection.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
from pathlib import Path

import torch

from evaluation.reporting import sha256_file
from scripts.assess_band_phase import compare, records
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def summarize(root):
    destination = root / "endpoint-summary-v2.json"
    if destination.exists():
        raise ValueError("summary already exists; preserve immutable evidence")
    checked_receipt(root / "complete.json")
    config = read(root / "registration.json")
    endpoints = read(root / "endpoints.json")
    report = {"integrity": {}, "matched_raw": {}, "loss_means": {}}
    for label, entry in endpoints.items():
        verify({entry["path"]: entry["sha256"]})
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        parent = torch.load(
            config["parents"][str(saved["seed"])]["path"], map_location="cpu", weights_only=False
        )
        if saved["step"] != 512 or len(saved["history"]) != 512:
            raise ValueError("incomplete endpoint")
        if any(not torch.equal(v, saved["model"]["base." + k]) for k, v in parent["model"].items()):
            raise ValueError("frozen backbone changed")
        if not all(torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("nonfinite endpoint")
        reproduced = None
        if saved["arm"] == "broad-common":
            old = torch.load(
                Path(config["prior"]) / "training" / str(saved["seed"]) / "endpoint.pth",
                map_location="cpu",
                weights_only=False,
            )
            reproduced = all(torch.equal(v, old["model"][k]) for k, v in saved["model"].items())
            if not reproduced:
                raise ValueError("P65 common control did not reproduce exactly")
        report["integrity"][label] = {
            "checkpoint_sha256": entry["sha256"],
            "steps": saved["step"],
            "frozen_backbone_exact": True,
            "p65_control_tensors_exact": reproduced,
        }
    native_count = 0
    for path in (root / "native-review").glob("*/report.json"):
        value = read(path)
        verify(value["artifacts"])
        native_count += sum(len(case["identity"]) for case in value["cases"])
    if native_count != 240 or len(endpoints) != 8:
        raise ValueError("incomplete registered review")
    for seed in config["seeds"]:
        native = records(read(root / "native-review" / f"{seed}-zero/report.json"), "native")
        selected = {(r["case"], r["role"]) for r in native}

        def raw(arm, seed=seed, selected=selected):
            return [
                r
                for r in records(read(root / "raw-review" / f"{seed}-{arm}.json"), "scores")
                if (r["case"], r["role"]) in selected
            ]

        for arm in config["arms"]:
            report["matched_raw"][f"{seed}-{arm}"] = compare(raw("zero"), raw(arm))
    for path in (root / "loss-review").glob("*.json"):
        values = read(path)
        report["loss_means"][path.stem] = {
            k: sum(r["balanced"][k] for r in values) / len(values) for k in values[0]["balanced"]
        }
    report.update(
        native_roles_verified=native_count,
        scope="Completed 512-update TRAIN fit calibration, not convergence or independent quality. Same native subset selected from saved raw scores without rescoring. All original counterexamples retained.",
    )
    registered(report, destination)
    frozen_source = root / "endpoint-summary-source-v2.py"
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    registered(
        {
            "files": {
                str(p): sha256_file(p) for p in (destination, root / "complete.json", frozen_source)
            }
        },
        root / "endpoint-summary-receipt-v2.json",
    )
    print("verified eight frozen endpoints, exact P65 controls and 240 native roles", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(args.run.resolve())
