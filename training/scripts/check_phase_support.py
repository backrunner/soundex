"""Freeze and check P66 native-support counterfactuals; no optimizer or quality selection.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.adapter_runtime import export
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from models.band_phase import BandPhaseCorrection
from scripts.audit_phase_transport import execute
from scripts.run_anchored_pilot import verify
from scripts.run_band_phase_pilot import model_for
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def prepare(audit, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve every attempt")
    checked_receipt(audit / "complete.json")
    previous = read(audit / "registration.json")
    prior = Path(previous["prior"])
    config = read(prior / "registration.json")
    endpoints = read(prior / "endpoints.json")
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for path in source.rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".py", ".json", ".toml", ".html", ".yaml"}
            and not set(path.relative_to(source).parts)
            & {".venv", "outputs", "__pycache__", "tests"}
        ):
            target = output / "source/training" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    models = {}
    paths = [audit / "complete.json", audit / "registration.json", prior / "registration.json"]
    for seed in (20261012, 20261013):
        endpoint = endpoints[str(seed)]
        if sha256_file(endpoint["path"]) != endpoint["sha256"]:
            raise ValueError("frozen endpoint drift")
        broad = model_for(config["parents"][str(seed)], seed)
        saved = torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        if saved["step"] != 512:
            raise ValueError("incomplete P65 endpoint")
        native = BandPhaseCorrection(broad.base, seed=seed + 65, phase_support="native").eval()
        native.load_state_dict(saved["model"])
        destination = output / "exports" / str(seed)
        export(native, destination)
        models[str(seed)] = str(destination / "diagnostic.onnx")
        paths.extend((Path(endpoint["path"]), Path(config["parents"][str(seed)]["path"])))
    paths.extend(p for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(Path(p) for p in read(audit / "complete.json")["files"])
    registered(
        {
            "format": "soundex-phase-support-p66-v1",
            "audit": str(audit),
            "prior": str(prior),
            "binary": previous["binary"],
            "models": models,
            "cases": previous["cases"],
            "scope": "P65 frozen weights with input-only output availability; zero training updates. No quality improvement, held-out test or timing claim.",
            "changed_question": "P66 found most correction power in retained bins. Can the head exclude those output updates without altering actual native output, while retaining all spectral inputs?",
            "gates": "Exact support membership after the 80-hop native warmup for every P66 trace; all 40 candidate native PCMs bit-identical to P65. No adjustment of thresholds or exclusions after execution.",
            "hashes": {str(p): sha256_file(p) for p in paths},
        },
        output / "registration.json",
    )
    print("registered support counterfactuals: two frozen endpoints, 40 native roles", flush=True)


@torch.no_grad()
def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "renders").exists():
        raise ValueError("existing renders retained; no silent restart")
    audit, prior = Path(config["audit"]), Path(config["prior"])
    memberships = []
    for path in sorted((audit / "renders").glob("*/*/runtime.json")):
        traces = read(path)["traces"]
        conditions = torch.tensor([h["condition"] for h in traces], dtype=torch.float32)
        predicted = BandPhaseCorrection.native_support(conditions).bool().numpy()
        actual = np.asarray([h["dsp"][0]["phase_weights"] for h in traces]) > 0
        # The diagnostic runtime overrides all predictions during its first 80
        # hops. Its condition alone is not an enhancement warmup clock.
        np.testing.assert_array_equal(predicted[80:], actual[80:])
        memberships.append({"trace": str(path), "checked_bins": int(actual[80:].size)})
    parity = []
    for seed, graph in config["models"].items():
        for case in config["cases"]:
            index = case["index"]
            for role in range(len(case["identity"])):
                old = prior / "native-after" / f"{seed}-phase" / f"{index:02d}-{role:02d}"
                destination = output / "renders" / seed / f"{index:02d}-{role:02d}"
                _wave, runtime = execute(
                    config["binary"], graph, old / "input.sxa", destination, None
                )
                expected, rate = read_sxa(old / "output.sxa")
                actual, actual_rate = read_sxa(destination / "output.sxa")
                if rate != actual_rate or not runtime["finite"]:
                    raise ValueError("timeline/rate/finite failure")
                np.testing.assert_array_equal(actual, expected)
                parity.append({"seed": seed, "case": index, "role": role, "pcm_exact": True})
        print("native support parity complete", seed, flush=True)
    verify(config["hashes"])
    registered(
        {"memberships": memberships, "parity": parity, "scope": config["scope"]},
        output / "assessment.json",
    )
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
    print("support feasibility complete; no retraining or quality improvement claimed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.audit.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
