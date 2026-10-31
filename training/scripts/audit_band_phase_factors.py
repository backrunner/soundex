"""P64 follow-up: separate magnitude and angular factors without changing audio."""

import argparse
import traceback
from pathlib import Path

from evaluation.band_attribution import phase_change_factors
from evaluation.longform_review import aligned_output
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.audit_band_attribution import grouped_pcm, read
from scripts.audit_joint_attribution import bound


def run(study, prior, output):
    if output.exists():
        raise ValueError("retain previous results")
    bound(prior / "registration.json", read(prior / "complete.json"))
    review = study / "longform-recheck"
    bound(review / "summary.json", read(review / "complete.json"))
    cases = [r for r in read(review / "summary.json")["records"] if r["path"] == "untreated"]
    if len(cases) != 160:
        raise ValueError("all paired and identity cases required")
    evidence = read(prior / "registration.json")["hashes"]
    files = [
        prior / "complete.json",
        prior / "registration.json",
        review / "summary.json",
        Path(__file__).resolve(),
    ]
    files.extend(
        Path(__file__).resolve().parents[1] / "evaluation" / n
        for n in ("band_attribution.py", "rust_bridge.py", "longform_review.py")
    )
    hashes = {str(p): sha256_file(p) for p in files}
    output.mkdir(parents=True)
    write_json_report(
        {
            "hashes": hashes,
            "scope": "P64 follow-up diagnostic after phase-interaction increase was observed. Symmetric factorization separates magnitude scaling from unit-angle distance changes; it is descriptive, not a pure phase loss, causal claim, selected candidate or training intervention. All 160 cases, two seeds, raw/native, matched control retained.",
            "zero_phase": "unit(0)=0 convention, zero-reference terms have zero weight; do not infer physical phase quality at zero amplitude",
        },
        output / "registration.json",
    )
    consumed = {}

    def load(path):
        expected = evidence[str(path)]
        if sha256_file(path) != expected:
            raise ValueError("PCM evidence changed")
        consumed[str(path)] = expected
        return read_sxa(path)

    try:
        records = []
        for index, case in enumerate(cases):
            reference, rate = load(Path(case["reference"]))
            start, length = case["score_start"], case["score_length"]
            reference = reference[start : start + length]
            for seed in (20261012, 20261013):
                for mode in ("raw", "native"):
                    waves = []
                    for kind in ("band", "global_matched"):
                        wave, actual_rate = load(
                            review / "renders" / f"{seed}-{kind}" / case["id"] / (mode + ".sxa")
                        )
                        if actual_rate != rate:
                            raise ValueError("rate mismatch")
                        waves.append(aligned_output(wave, start, length))
                    records.append(
                        {
                            **{k: case[k] for k in ("id", "source", "codec", "role")},
                            "path": f"{mode}/{seed}-band-minus-matched",
                            "scores": phase_change_factors(*waves, reference, rate),
                        }
                    )
            if (index + 1) % 20 == 0:
                print("cases", index + 1, "/", len(cases), flush=True)
        write_json_report(records, output / "records.json")
        write_json_report(grouped_pcm(records), output / "summary.json")
        for p, h in {**hashes, **consumed}.items():
            if sha256_file(p) != h:
                raise ValueError("evidence drift during diagnostic")
        write_json_report({"files": consumed}, output / "pcm-inputs.json")
        write_json_report(
            {"files": {str(p): sha256_file(p) for p in output.glob("*.json")}},
            output / "complete.json",
        )
        print("complete", output, flush=True)
    except BaseException:
        write_json_report({"traceback": traceback.format_exc()}, output / "failure.json")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "prior", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.prior.resolve(), args.output.resolve())
