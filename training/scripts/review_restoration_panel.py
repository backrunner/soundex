"""P71 frozen panel calibration on the complete P70 diagnostic selection.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

from evaluation.longform_review import aligned_output
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import VERSION, paired_recordings, panel
from evaluation.rust_bridge import read_sxa
from scripts.audit_baseline_metrics import controls
from scripts.audit_fused_spectrum import snapshot
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def run(prior, output):
    if output.exists():
        raise ValueError("retain every attempt; unused output required")
    checked_receipt(prior / "summary-receipt.json")
    old = read(prior / "registration.json")
    native = Path(old["prior"]) / "longform-recheck/renders"
    output.mkdir(parents=True)
    snapshot(output)
    hashes = {str(prior / "registration.json"): sha256_file(prior / "registration.json")}
    for case in old["cases"]:
        paths = [Path(case[k]) for k in ("reference", "input")]
        paths += [native / label / case["id"] / "native.sxa" for label in old["models"]]
        paths += [prior / "metrics" / (case["id"] + ".json")]
        hashes.update({str(p): sha256_file(p) for p in paths})
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    comparisons = [
        ("native/" + label, control)
        for label in old["models"]
        if "adjacent" in label
        for control in (
            "native/legacy",
            "untreated",
            "native/" + label.replace("adjacent", "temporal_repeat"),
        )
    ]
    config = {
        "version": VERSION,
        "prior": str(prior),
        "cases": old["cases"],
        "models": old["models"],
        "hashes": hashes,
        "comparisons": comparisons,
        "protocol": "All 20 P70 first-excerpt MP3 sources; all eight native paths, untreated and five controls. Preserve historical scores. Independent 512/2048/8192 analysis, normalized-power shape, SC, envelope/dynamics and absolute RMS. Complex/stereo rulers are diagnostics. Equal recording weights; 2000 paired source-bootstrap resamples seed71, no independent-test or multiplicity-adjusted significance claim. No automatic promotion or retroactive change to P69 outcome.",
        "perception": "No installed ViSQOL executable or listener ratings. Missing evidence remains missing; no proxy MOS. Primary-source guidance reviewed separately.",
        "future_capacity_rule": "Both seeds must reduce native source-equal MR and HF SC versus matched small full-training control. Bootstrap CI, shape/energy/dynamics, absolute weak-band errors, identity, preserved bands, stereo and all adverse sources required. Conflicting rulers mean a tradeoff, never an overall quality win. Listening/device evidence required for promotion. Freeze endpoints before scoring; no intermediate winner.",
    }
    registered(config, output / "registration.json")
    verify(hashes)
    records = []
    for i, case in enumerate(old["cases"]):
        reference, rate = read_sxa(Path(case["reference"]))
        dry, dry_rate = read_sxa(Path(case["input"]))
        if rate != dry_rate:
            raise ValueError("rate mismatch")
        start, n = case["score_start"], case["score_length"]
        reference, dry = reference[start : start + n], dry[start : start + n]
        waves = {"untreated": dry, **controls(dry, reference, rate, 70 + i)}
        for label in old["models"]:
            wave, actual = read_sxa(native / label / case["id"] / "native.sxa")
            if actual != rate:
                raise ValueError("native rate mismatch")
            waves["native/" + label] = aligned_output(wave, start, n)
        historical = read(prior / "metrics" / (case["id"] + ".json"))["scores"]
        cases = []
        for label, wave in waves.items():
            scored = panel(wave, reference, rate)
            row = {
                "source": case["source"],
                "case": case["id"],
                "path": label,
                "scores": {**historical[label], **scored["metrics"]},
                "resolutions": scored["resolutions"],
            }
            cases.append(row)
        registered(cases, output / "records" / (case["id"] + ".json"))
        records.extend(cases)
        print("panel", i + 1, "/20", flush=True)
    compared = {a + " vs " + b: paired_recordings(records, a, b) for a, b in comparisons}
    # Retain the complete negative-control behavior rather than selecting one metric.
    calibration = {
        label: paired_recordings(records, label, "untreated")
        for label in ("correct_hf", "half_correct_hf", "same_energy_scrambled_hf", "mute_hf")
    }
    verify(hashes)
    registered(
        {
            "comparisons": compared,
            "calibration": calibration,
            "records": len(records),
            "version": VERSION,
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
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.prior.resolve(), args.output.resolve())
    except Exception:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
