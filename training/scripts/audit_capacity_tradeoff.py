"""P73: account frozen P72 errors, without training, rendering or timing retries.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
import traceback
from collections import defaultdict
from pathlib import Path

from evaluation.band_attribution import (
    deadline_accounting,
    phase_change_factors,
    spectral_error_components,
)
from evaluation.joint_attribution import compare, index_records, mean
from evaluation.longform_review import aligned_output
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.audit_band_attribution import grouped_pcm, read
from scripts.audit_joint_attribution import bound


def source_stages(review):
    """Match actual source/case/role before attributing native-minus-raw MR deltas."""
    result = {}
    for seed, panels in review["source_review"].items():
        raw = {(r["case"], r["role"]): r for r in panels["raw_native_population"]["cases"]}
        native = {(r["case"], r["role"]): r for r in panels["native"]["cases"]}
        if raw.keys() != native.keys():
            raise ValueError("unmatched source review cases")
        rows, groups = [], defaultdict(lambda: defaultdict(list))
        for key, n in native.items():
            r = raw[key]
            fields = ("case", "role", "source", "source_id", "codec", "group")
            if any(n[k] != r[k] for k in fields):
                raise ValueError("unmatched source review metadata")
            scores = {
                "raw_small": r["before"]["mr"],
                "raw_double": r["after"]["mr"],
                "native_small": n["before"]["mr"],
                "native_double": n["after"]["mr"],
                "raw_delta": r["delta"]["mr"],
                "native_delta": n["delta"]["mr"],
                "native_minus_raw_delta": n["delta"]["mr"] - r["delta"]["mr"],
            }
            rows.append({**{k: n[k] for k in fields}, "scores": scores})
            for codec in ("all", n["codec"]):
                groups[n["group"] + "/" + codec][n["source"]].append(scores)
        result[seed] = {
            "records": rows,
            "groups": {
                k: {
                    "sources": len(sources),
                    "per_source": {s: mean(v) for s, v in sources.items()},
                    "source_mean": mean([mean(v) for v in sources.values()]),
                }
                for k, sources in sorted(groups.items())
            },
        }
    return result


def run(study, output):
    if output.exists():
        raise ValueError("fresh destination required; preserve previous evidence")
    review = study / "longform-recheck"
    receipt, long_receipt = read(study / "complete.json"), read(review / "complete.json")
    files = [study / "complete.json", review / "complete.json"]
    for name in ("endpoints.json", "cost-summary.json", "longform-recheck/complete.json"):
        bound(study / name, receipt)
        files.append(study / name)
    for name in ("summary.json", "prepared.json"):
        bound(review / name, long_receipt)
        files.append(review / name)
    completion = study / "completion-review-20261107"
    bound(completion / "summary.json", read(completion / "receipt.json"))
    files += [completion / "summary.json", completion / "receipt.json"]
    endpoints = read(study / "endpoints.json")
    expected = {f"{s}-{k}" for s in (20261012, 20261013) for k in ("small", "double")}
    if set(endpoints) != expected:
        raise ValueError("all four P72 endpoints required")
    records = read(review / "summary.json")["records"]
    cases = [r for r in records if r["path"] == "untreated"]
    paths = {"untreated"} | {
        f"{mode}/{label}"
        for mode in ("raw", "native")
        for label in expected | {"legacy", "identity_dsp"}
    }
    if len(cases) != 160 or len(records) != 2080 or {r["path"] for r in records} != paths:
        raise ValueError("complete P72 long-form population required")
    hashes = {str(p): sha256_file(p) for p in files}
    hashes.update({e["path"]: e["sha256"] for e in endpoints.values()})
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
    source_root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/audit_capacity_tradeoff.py",
        "scripts/audit_band_attribution.py",
        "scripts/audit_joint_attribution.py",
        "evaluation/band_attribution.py",
        "evaluation/joint_attribution.py",
        "evaluation/longform_review.py",
        "evaluation/rust_bridge.py",
        "evaluation/reporting.py",
    ):
        src, dst = source_root / relative, output / "source" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[str(src)] = hashes[str(dst)] = sha256_file(src)
    write_json_report(
        {
            "format": "soundex-capacity-tradeoff-p73-v1",
            "hashes": hashes,
            "scope": "Post-hoc exposed-recording diagnosis of all four P72 frozen endpoints. No training, model selection, threshold change, rerender or timing rerun. Defaults unchanged.",
            "changed_question": "P64 decomposed P63 band versus global architectures, not fully joint P72 capacity. Account P72 matched-size magnitude/angular tradeoffs and same-case development native/raw transfer before choosing another mechanism.",
            "ruler": "Whole-excerpt rectangular FFT, exact one-sided Parseval physical-channel mean power; magnitude and amplitude-weighted phase interaction, then symmetric magnitude/angular factors. Descriptive algebra, not causal intervention, MR replacement, audibility or achievable phase oracle.",
            "coverage": "All 160 cases, 2080 PCM paths, 640 matched-capacity factors; paired/identity, every codec and source retained. Development source comparison is separately matched by case/role.",
            "timing": "Account original paced traces excluding first 100 hops. Direct replay sleeps until arrival; it is not production worker/callback/device qualification. Preserve every failure.",
        },
        output / "registration.json",
    )
    try:
        for p, h in hashes.items():
            if sha256_file(p) != h:
                raise ValueError("input drift: " + p)
        print("all inputs verified", len(hashes), flush=True)
        groups, attribution = index_records(records), {}
        for seed in (20261012, 20261013):
            for mode in ("raw", "native"):
                for role in ("paired", "identity"):
                    a, b = (groups[(f"{mode}/{seed}-{k}", role)] for k in ("double", "small"))
                    for codec in ("all", *sorted({r["codec"] for r in a.values()})):
                        aa, bb = (
                            {k: v for k, v in g.items() if codec == "all" or v["codec"] == codec}
                            for g in (a, b)
                        )
                        attribution[f"{seed}/{mode}/{role}/{codec}"] = compare(aa, bb)
        write_json_report(attribution, output / "mr-attribution.json")
        write_json_report(
            source_stages(read(completion / "summary.json")), output / "source-stages.json"
        )
        timing = {
            name: deadline_accounting(read(study / "native-cost" / name / "raw.json"))
            for name, v in costs.items()
            if v["paced"]
        }
        if any(
            t["late_hops"] != costs[n]["paced_late_hops"]
            or t["measured_hops"] != costs[n]["measured_hops"]
            for n, t in timing.items()
        ):
            raise ValueError("deadline accounting differs from frozen report")
        write_json_report(timing, output / "deadline-accounting.json")
        pcm, factors = [], []
        for index, case in enumerate(cases):
            reference, rate = read_sxa(Path(case["reference"]))
            start, length = case["score_start"], case["score_length"]
            reference = reference[start : start + length]
            meta = {k: case[k] for k in ("id", "source", "codec", "role", "channels", "rate")}
            waves = {}
            for path in sorted(paths):
                if path == "untreated":
                    wave, actual_rate = read_sxa(Path(case["input"]))
                    wave = wave[start : start + length]
                else:
                    mode, label = path.split("/")
                    wave, actual_rate = read_sxa(
                        review / "renders" / label / case["id"] / (mode + ".sxa")
                    )
                    wave = aligned_output(wave, start, length)
                if actual_rate != rate or rate != case["rate"]:
                    raise ValueError("rate mismatch")
                waves[path] = wave
                pcm.append(
                    {
                        **meta,
                        "path": path,
                        "scores": spectral_error_components(wave, reference, rate),
                    }
                )
            for seed in (20261012, 20261013):
                for mode in ("raw", "native"):
                    factors.append(
                        {
                            **meta,
                            "path": f"{mode}/{seed}-double-minus-small",
                            "scores": phase_change_factors(
                                waves[f"{mode}/{seed}-double"],
                                waves[f"{mode}/{seed}-small"],
                                reference,
                                rate,
                            ),
                        }
                    )
            if (index + 1) % 10 == 0:
                write_json_report(
                    {"stage": "accounting", "cases": index + 1, "total": len(cases)},
                    output / "progress.json",
                )
                print("cases", index + 1, "/", len(cases), flush=True)
        for name, rows in (("pcm", pcm), ("factors", factors)):
            write_json_report(rows, output / (name + "-records.json"))
            write_json_report(grouped_pcm(rows), output / (name + "-summary.json"))
        for p, h in hashes.items():
            if sha256_file(p) != h:
                raise ValueError("input changed during audit: " + p)
        write_json_report(
            {
                "files": {
                    str(p): sha256_file(p)
                    for p in output.glob("*.json")
                    if p.name != "progress.json"
                }
            },
            output / "complete.json",
        )
        write_json_report(
            {
                "stage": "complete",
                "pcm_records": len(pcm),
                "factor_records": len(factors),
                "qualified": False,
            },
            output / "progress.json",
        )
        print("complete", output, flush=True)
    except BaseException:
        write_json_report({"traceback": traceback.format_exc()}, output / "failure.json")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
