"""P64 frozen P63 codec, exact PCM error and retained-deadline diagnostics.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

from evaluation.band_attribution import deadline_accounting, spectral_error_components
from evaluation.joint_attribution import compare, index_records, mean
from evaluation.longform_review import aligned_output
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.audit_joint_attribution import bound


def read(path):
    return json.loads(path.read_text())


def grouped_pcm(records):
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        for codec in ("all", r["codec"]):
            groups[(r["path"], r["role"], codec)][r["source"]].append(r["scores"])
    return {
        "/".join(key): {
            "sources": len(sources),
            "per_source": {k: mean(rows) for k, rows in sources.items()},
            "source_mean": mean([mean(rows) for rows in sources.values()]),
        }
        for key, sources in sorted(groups.items())
    }


def run(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists():
        raise ValueError("fresh output required; preserve all failed evidence")
    review = study / "longform-recheck"
    receipt, long_receipt = read(study / "complete.json"), read(review / "complete.json")
    for name in ("endpoints.json", "cost-summary.json", "longform-recheck/complete.json"):
        bound(study / name, receipt)
    for name in ("summary.json", "registration.json", "prepared.json", "endpoints.json"):
        bound(review / name, long_receipt)
    endpoints = read(study / "endpoints.json")
    expected = {
        f"{s}-{k}" for s in (20261012, 20261013) for k in ("band", "global", "global_matched")
    }
    if set(endpoints) != expected:
        raise ValueError("all six frozen P63 endpoints required")
    for e in endpoints.values():
        if sha256_file(e["path"]) != e["sha256"]:
            raise ValueError("endpoint drift")
    labels = expected | {"legacy", "identity_dsp"}
    records = [
        r
        for r in read(review / "summary.json")["records"]
        if r["path"] == "untreated" or r["path"].split("/", 1)[-1] in labels
    ]
    if len(records) != 2720 or len({r["id"] for r in records}) != 160:
        raise ValueError("all 160 cases and 17 paths required")
    hashes = {}
    files = [
        study / "complete.json",
        study / "endpoints.json",
        study / "cost-summary.json",
        review / "complete.json",
        review / "summary.json",
        review / "prepared.json",
        Path(__file__).resolve(),
    ]
    files += [
        Path(__file__).resolve().parents[1] / "evaluation" / n
        for n in (
            "band_attribution.py",
            "joint_attribution.py",
            "longform_review.py",
            "rust_bridge.py",
        )
    ]
    hashes.update({str(p): sha256_file(p) for p in files})
    prepared = read(review / "prepared.json")["files"]
    for r in records:
        hashes[r["reference"]] = prepared[r["reference"]]
        if r["path"] == "untreated":
            hashes[r["input"]] = prepared[r["input"]]
        else:
            mode, label = r["path"].split("/")
            directory = review / "renders" / label / r["id"]
            rp = directory / "complete.json"
            bound(rp, long_receipt)
            hashes[str(rp)] = sha256_file(rp)
            wave = directory / (mode + ".sxa")
            hashes[str(wave)] = read(rp)["files"][str(wave)]
    costs = read(study / "cost-summary.json")
    for name, value in costs.items():
        if value["paced"]:
            hashes[str(study / "native-cost" / name / "raw.json")] = value["raw_sha256"]
    output.mkdir(parents=True)
    write_json_report(
        {
            "format": "soundex-band-attribution-p64-v1",
            "hashes": hashes,
            "scope": "Post-hoc exposed P55 diagnosis; same endpoints, every case retained. No new training, candidate selection, threshold, rendering or timing run.",
            "changed_question": "P58 located band-level tradeoffs for another architecture; it did not separate P63 codec-specific changes or exact PCM magnitude/phase-interaction power. P64 adds that decomposition and accounts previously recorded paced deadlines without rerunning benchmarks.",
            "pcm_ruler": "10-second rectangular FFT, one-sided Parseval normalization, physical-channel mean power. Exact magnitude plus amplitude-weighted phase-interaction identity; not phase-only error or deployable reference-phase oracle. Includes Nyquist; not a new MR objective or transient metric.",
            "timing_exclusion": "First 100 benchmark hops, separate from 80-hop model-history warmup. Retain every late completion including those with short service time.",
        },
        output / "registration.json",
    )
    try:
        # Bind all consumed files before any arithmetic; keep receipt paths logical.
        for i, (p, expected_hash) in enumerate(hashes.items()):
            if sha256_file(p) != expected_hash:
                raise ValueError("input drift: " + p)
            if i % 500 == 0:
                print("verified", i, "/", len(hashes), flush=True)
        groups = index_records(records)
        attribution = {}
        for seed in (20261012, 20261013):
            for control in (f"{seed}-global", f"{seed}-global_matched", "legacy"):
                for mode in ("raw", "native"):
                    for role in ("paired", "identity"):
                        a, b = (
                            groups[(f"{mode}/{seed}-band", role)],
                            groups[(f"{mode}/{control}", role)],
                        )
                        for codec in ("all", *(sorted({r["codec"] for r in a.values()}))):
                            aa = {
                                k: v for k, v in a.items() if codec == "all" or v["codec"] == codec
                            }
                            bb = {
                                k: v for k, v in b.items() if codec == "all" or v["codec"] == codec
                            }
                            attribution[f"{seed}/{control}/{mode}/{role}/{codec}"] = compare(aa, bb)
        write_json_report(attribution, output / "score-attribution.json")
        timing = {}
        for name, value in costs.items():
            if value["paced"]:
                timing[name] = deadline_accounting(read(study / "native-cost" / name / "raw.json"))
                if (
                    timing[name]["measured_hops"] != value["measured_hops"]
                    or timing[name]["late_hops"] != value["paced_late_hops"]
                ):
                    raise ValueError("recorded deadline counts mismatch")
        write_json_report(timing, output / "deadline-accounting.json")
        pcm = []
        reference, current = None, None
        for index, r in enumerate(records):
            if r["id"] != current:
                reference, rate = read_sxa(Path(r["reference"]))
                start, length = r["score_start"], r["score_length"]
                reference = reference[start : start + length]
                current = r["id"]
            if r["path"] == "untreated":
                wave, actual_rate = read_sxa(Path(r["input"]))
                wave = wave[start : start + length]
            else:
                mode, label = r["path"].split("/")
                wave, actual_rate = read_sxa(review / "renders" / label / r["id"] / (mode + ".sxa"))
                wave = aligned_output(wave, start, length)
            if actual_rate != rate or rate != r["rate"]:
                raise ValueError("rate mismatch")
            pcm.append(
                {
                    **{
                        k: r[k]
                        for k in ("id", "path", "source", "codec", "role", "channels", "rate")
                    },
                    "scores": spectral_error_components(wave, reference, rate),
                }
            )
            if (index + 1) % 170 == 0:
                write_json_report(
                    {"stage": "pcm-attribution", "records": index + 1, "total": len(records)},
                    output / "progress.json",
                )
                print("PCM", index + 1, "/", len(records), flush=True)
        write_json_report(pcm, output / "pcm-records.json")
        write_json_report(grouped_pcm(pcm), output / "pcm-summary.json")
        for p, expected_hash in hashes.items():
            if sha256_file(p) != expected_hash:
                raise ValueError("input changed during diagnosis: " + p)
        paths = [
            output / n
            for n in (
                "registration.json",
                "score-attribution.json",
                "deadline-accounting.json",
                "pcm-records.json",
                "pcm-summary.json",
            )
        ]
        write_json_report(
            {"files": {str(p): sha256_file(p) for p in paths}}, output / "complete.json"
        )
        write_json_report(
            {"stage": "complete", "records": len(pcm), "qualified": False}, output / "progress.json"
        )
        print("complete", output, flush=True)
    except BaseException:
        write_json_report({"traceback": traceback.format_exc()}, output / "failure.json")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study.absolute(), args.output.absolute())
