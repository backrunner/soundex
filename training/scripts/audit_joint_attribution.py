"""Attribute P57 endpoint changes by source, band and raw/native stage (P58)."""

import argparse
import json
from pathlib import Path

from evaluation.joint_attribution import attribute
from evaluation.reporting import sha256_file, write_json_report


def read(path):
    return json.loads(path.read_text())


def bound(path, receipt):
    if receipt["files"].get(str(path)) != sha256_file(path):
        raise ValueError("receipt mismatch: " + str(path))


def run(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists():
        raise ValueError("fresh output required; preserve previous evidence")
    receipt = read(study / "complete.json")
    for name in ("endpoints.json", "cost-summary.json", "longform-recheck/complete.json"):
        bound(study / name, receipt)
    review = study / "longform-recheck"
    bound(review / "summary.json", read(review / "complete.json"))
    endpoints = read(study / "endpoints.json")
    for entry in endpoints.values():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError("changed endpoint")
    pairs = {}
    for name in sorted(endpoints):
        if name.endswith("-full"):
            base = name.removesuffix("-full")
            pairs[base + "/full-vs-adapter"] = (name, base + "-adapter")
            pairs[base + "/full-vs-parent"] = (name, base + "-parent")
            pairs[base + "/full-vs-legacy"] = (name, "legacy")
    if len(pairs) != 6:
        raise ValueError("both P57 seeds and all comparisons required")
    files = [
        study / "complete.json",
        study / "endpoints.json",
        study / "cost-summary.json",
        review / "complete.json",
        review / "summary.json",
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "evaluation/joint_attribution.py",
    ]
    hashes = {str(p): sha256_file(p) for p in files}
    output.mkdir(parents=True)
    write_json_report(
        {
            "format": "soundex-joint-attribution-p58-v1",
            "hashes": hashes,
            "pairs": pairs,
            "scope": "Post-hoc exposed P55 recheck, all sources retained. Endpoint output-stage attribution only; no intermediate training checkpoints or causal component intervention.",
        },
        output / "registration.json",
    )
    result = attribute(read(review / "summary.json")["records"], pairs)
    timing = read(study / "cost-summary.json")
    late = {k: v for k, v in timing.items() if v["paced_late_hops"]}
    write_json_report(
        {
            "comparisons": result,
            "retained_paced_failures": late,
            "qualified": False,
            "new_training": False,
        },
        output / "summary.json",
    )
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError("input changed during attribution")
    write_json_report(
        {
            "files": {
                str(p): sha256_file(p)
                for p in (output / "registration.json", output / "summary.json")
            }
        },
        output / "complete.json",
    )
    print(
        json.dumps(
            {"complete": str(output), "comparisons": len(pairs), "paced_failures": len(late)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
