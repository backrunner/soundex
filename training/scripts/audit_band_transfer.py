#!/usr/bin/env python3
"""P15 frozen per-band interventions, residual transfer and native counterexamples."""

import argparse
import fcntl
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.band_transfer import intervene, transfer
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_objective import amplitude_features
from models.rendered_probe import make_rendered_model
from scripts.run_fit_diagnostic import replay
from scripts.run_subset_study import Pool, features


def score(prediction, objective):
    value = objective(prediction)
    return {
        "paired": float(value["restoration"][0]),
        "identity": float(value["preservation"][1]),
        "total": float(value["total"]),
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((args.study / "config.json").read_text())
        parent = json.loads(args.parent.read_text())
        registered = json.loads((args.audit / "registration.json").read_text())
        receipt = json.loads((args.audit / "summary.json").read_text())
        if receipt["status"] != "complete":
            raise ValueError("P14 incomplete")
        for name, sha in receipt["reports"].items():
            if sha256_file(args.audit / name) != sha:
                raise ValueError("P14 audit drift")
        if (
            sha256_file(args.study / "config.json") != registered["study_sha256"]
            or sha256_file(args.parent) != config["parent_sha256"]
        ):
            raise ValueError("registered study/parent drift")
        rows, manifests = load_sources([Path(p) for p in config["manifests"]])
        if manifests != config["manifests"]:
            raise ValueError("manifest drift")
        detector = args.study / "binaries/soundex-gain-data"
        binary = args.study / "binaries/soundex-prediction-replay"
        onnx = args.study / "replay-contract.onnx"
        for file, key in (
            (detector, "detector_sha256"),
            (binary, "replay_sha256"),
            (onnx, "onnx_sha256"),
        ):
            if sha256_file(file) != config[key]:
                raise ValueError("native contract drift")
        source = Path(__file__).resolve().parents[1]
        registration = {
            "scope": "post-hoc frozen intervention; no training or promotion",
            "study_sha256": registered["study_sha256"],
            "audit_receipt_sha256": sha256_file(args.audit / "summary.json"),
            "checkpoints": registered["checkpoints"],
            "native_row_ids": registered["native_row_ids"],
            "native_selection": "P14 worst-three music clips; destination by largest identity band loss; source by largest isolated residual RMS in that destination",
            "source_files": {
                str(p.relative_to(source)): sha256_file(p)
                for p in source.rglob("*.py")
                if not any(
                    x in {".venv", "tests", "outputs", "__pycache__"}
                    for x in p.relative_to(source).parts
                )
            },
        }
        path = root / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("intervention registration changed")
        write_json_report(registration, path)
        clips = EnvelopeClips(rows, detector, Path(parent["cache_path"]))
        for group, specs in (
            ("training_bank", config["crop_sampling"]["bank"]),
            ("development", config["development_specs"]),
        ):
            pool = Pool(specs, clips)
            for label, sha in registered["checkpoints"].items():
                seed = label.split("-")[-1]
                checkpoint = args.study / f"size-40-{seed}" / "step-2048.pth"
                if sha256_file(checkpoint) != sha:
                    raise ValueError("checkpoint drift")
                prior = json.loads((args.audit / f"{group}-{label}.json").read_text())["records"]
                model = make_rendered_model("rendered_balanced").eval()
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
                )
                records, native = [], []
                for i, row in enumerate(pool.rows):
                    if prior[i]["spec"] != specs[i]:
                        raise ValueError("case order drift")
                    objective, context = pool.select([i])
                    b = objective.batch
                    initial = amplitude_features(torch.pow(10.0, b["x"][:, 0] / 20), b["x"])
                    prediction = features(model, objective, context)
                    result = transfer(prediction, initial, b)
                    if (
                        result["wave_superposition_max_error"].max() > 2e-5
                        or result["spectrum_superposition_max_error"].max() > 2e-4
                    ):
                        raise ValueError("linear residual superposition failed")
                    record = {
                        "spec": specs[i],
                        "track_id": row["track_id"],
                        "corpus": row["corpus"],
                        "genre": row.get("source_metadata", {}).get("genre"),
                        "initial": score(initial, objective),
                        "full": score(prediction, objective),
                        "transfer": {
                            key: value.tolist() for key, value in result.items() if key != "masks"
                        },
                        "interventions": [],
                    }
                    destination = max(
                        range(5), key=lambda j: prior[i]["roles"][1]["rendered"][j]["total"]
                    )
                    selected = int(result["matrix"][1, :, destination].argmax())
                    for band in range(5):
                        only = intervene(prediction, initial, result["masks"], band)
                        without = intervene(prediction, initial, result["masks"], band, remove=True)
                        record["interventions"].append(
                            {
                                "source_band": band,
                                "only": score(only, objective),
                                "without": score(without, objective),
                            }
                        )
                        if (
                            group == "development"
                            and specs[i]["row_id"] in registered["native_row_ids"]
                            and band == selected
                        ):
                            native.append(
                                {
                                    "spec": specs[i],
                                    "destination_band": destination,
                                    "source_band": band,
                                    "only": replay(only, b, objective, binary, onnx),
                                    "without": replay(without, b, objective, binary, onnx),
                                }
                            )
                    records.append(record)
                    if (i + 1) % 40 == 0:
                        progress = {
                            "group": group,
                            "model": label,
                            "cases": i + 1,
                            "total": len(specs),
                        }
                        write_json_report(progress, root / "progress.json")
                        print(json.dumps(progress), flush=True)
                write_json_report(
                    {"group": group, "model": label, "records": records, "native": native},
                    root / f"{group}-{label}.json",
                )
        write_json_report(
            {
                "status": "complete",
                "reports": {
                    p.name: sha256_file(p)
                    for p in root.glob("*.json")
                    if p.name not in {"summary.json", "progress.json"}
                },
            },
            root / "summary.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent", "audit", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
