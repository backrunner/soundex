#!/usr/bin/env python3
"""Native validation of post-hoc P19 band interventions, never release evidence."""

import argparse
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.band_transfer import band_masks, intervene
from evaluation.reporting import sha256_file, write_json_report
from models.rendered_probe import make_rendered_model
from scripts.audit_coverage_failure import read
from scripts.run_fit_diagnostic import replay
from scripts.run_subset_study import Pool, features


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    config = read(args.study / "registration.json")
    audit = read(args.audit / "summary.json")
    for path, expected in config["input_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("input drift")
    for name, expected in audit["reports"].items():
        if sha256_file(args.audit / name) != expected:
            raise ValueError("stage report drift")
    inputs = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
    detector = next(p for p in inputs if p.name == "soundex-gain-data")
    binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
    onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
    parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    checks = read(args.study / "review-registration.json")
    results = []
    for seed in config["seeds"]:
        label = f"wide-16192-{seed}"
        if sha256_file(Path(checks[label]["path"])) != checks[label]["sha256"]:
            raise ValueError("checkpoint drift")
        model = make_rendered_model("rendered_balanced").eval()
        model.load_state_dict(
            torch.load(checks[label]["path"], map_location="cpu", weights_only=False)["model"]
        )
        records = read(args.audit / f"development-{label}.json")["records"]
        selected = [r for r in records if r["role"] == "identity" and r["interventions"]]
        for record in selected:
            objective, context = Pool([record["spec"]], clips).select([0])
            b = objective.batch
            predicted = features(model, objective, context)
            destination = max(range(5), key=lambda j: record["stages"]["limited"][j]["total"])
            source = max(range(5), key=lambda j: record["transfer"]["matrix"][j][destination])
            masks = band_masks(b["rate"], dtype=predicted.dtype, device=predicted.device)
            for remove in (False, True):
                modified = intervene(predicted, b["x"], masks, source, remove=remove)
                paired, identity = replay(modified, b, objective, binary, onnx)
                results.append(
                    {
                        "label": label,
                        "spec": record["spec"],
                        "source_band": source,
                        "destination_band": destination,
                        "remove": remove,
                        "paired": paired,
                        "identity": identity,
                    }
                )
        print(f"{label}: {len(selected) * 4} native role replays", flush=True)
    write_json_report(
        {
            "scope": "post-hoc mechanism validation only; no band suppression deployment",
            "audit_sha256": sha256_file(args.audit / "summary.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "results": results,
            "native_count": len(results) * 2,
            "maximum_proxy_error": max(
                r[role]["proxy_max_error"] for r in results for role in ("paired", "identity")
            ),
        },
        args.audit / "native-interventions.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    run(parser.parse_args())
