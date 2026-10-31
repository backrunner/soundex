"""P66 same-input native phase support and synthesis-stage attribution.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.adapter_runtime import environment
from evaluation.fidelity_calibration import assess_wave
from evaluation.phase_transport import correction_support
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.assess_band_phase import compare
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt

STAGES = ("raw_ola", "activity_ola", "magnitude_ola", "phase_ola", "wet_limited", "native")


def prepare(prior, binary, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve every attempt")
    checked_receipt(prior / "complete.json")
    checked_receipt(prior / "assessment-receipt.json")
    old = read(prior / "registration.json")
    output.mkdir(parents=True)
    asset = output / "soundex-context-replay"
    shutil.copy2(binary, asset)
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
    repository = source.parent
    rust_paths = [repository / "Cargo.toml", repository / "Cargo.lock"] + [
        p
        for p in (repository / "crates").rglob("*")
        if p.is_file() and p.suffix in {".rs", ".toml"}
    ]
    for path in rust_paths:
        target = output / "source-rust" / path.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    models = {
        f"{seed}-{arm}": str(prior / "exports" / f"{seed}-{arm}" / "diagnostic.onnx")
        for seed in (20261012, 20261013)
        for arm in ("zero", "phase")
    }
    models["identity_dsp"] = str(Path(old["prior"]) / "assets/identity_dsp.onnx")
    paths = [
        asset,
        prior / "complete.json",
        prior / "assessment-receipt.json",
        prior / "registration.json",
        *map(Path, models.values()),
    ]
    paths.extend(
        p
        for directory in ("source", "source-rust")
        for p in (output / directory).rglob("*")
        if p.is_file()
    )
    cases = read(prior / "native-after/20261012-zero/report.json")["cases"]
    for case in cases:
        for sample in case["samples"]["calibration"]:
            row = sample["row"]
            clean = Path(row["_root"]) / row["clean_path"]
            if sha256_file(clean) != old["hashes"][str(clean)]:
                raise ValueError("registered clean reference drift")
            paths.append(clean)
    for label in models:
        if label == "identity_dsp":
            continue
        report = prior / "native-after" / label / "report.json"
        paths.append(report)
        paths.extend(Path(p) for p in read(report)["artifacts"])
        verify(read(report)["artifacts"])
    if sha256_file(models["identity_dsp"]) != old["hashes"][models["identity_dsp"]]:
        raise ValueError("identity control drift")
    config = {
        "format": "soundex-phase-transport-p66-v1",
        "prior": str(prior),
        "binary": str(asset),
        "models": models,
        "order": list(models),
        "cases": cases,
        "stages": STAGES,
        "native_roles": 100,
        "trace_off_controls": 5,
        "scope": "Frozen P65 TRAIN recheck, same two batches/eight music and two speech sources with identities. No optimization, held-out scoring, reference phase injection, policy change or timing qualification.",
        "changed_question": "P30 oracle fusion limits do not locate P65 learned corrections. Measure actual native phase support and ordered stage deltas with frozen magnitudes; distinguish suppressed corrections from wrong corrections in passed bins.",
        "stage_semantics": "shadow Rust synthesis: raw prediction, activity bypass, magnitude crossover, actual phase crossover, actual wet limiter, actual final gate/dry limiter/ceiling; ordered counterfactuals are not order-independent causal shares. PhaseSmoother is stateless within-frame unit-phasor mixing.",
        "gates": "all 80 P65 final PCMs must match saved outputs bit-for-bit; first case of all five graphs must match trace-off PCM; clean identity inputs must match reconstructed clean references; no retrospective tolerance or case exclusion",
        "score_start_samples": 89 * 128,
        "score_end_trim_samples": 128,
        "hashes": {str(p): sha256_file(p) for p in paths},
    }
    registered(config, output / "registration.json")
    print("registered P66: 100 traced roles + five trace-off controls", flush=True)


def target_for(sample, frames, rate):
    row, spec = sample["row"], sample["spec"]
    pcm, source_rate = sf.read(
        Path(row["_root"]) / row["clean_path"], dtype="float32", always_2d=True
    )
    if source_rate != rate or pcm.shape != (row["length_samples"], 1):
        raise ValueError("reference geometry mismatch")
    pcm = pcm[spec["start"] : spec["start"] + frames, 0] * 10 ** (spec["gain_db"] / 20)
    pcm = np.pad(pcm, (0, frames - len(pcm)))
    return pcm, np.pad(pcm, (128, 0))[:frames]


def execute(binary, graph, source, directory, mode):
    directory.mkdir(parents=True)
    command = [
        str(binary),
        graph,
        str(source),
        str(directory / "output.sxa"),
        str(directory / "runtime.json"),
        "1",
    ]
    if mode:
        command.append(mode)
    result = subprocess.run(command, env=environment(), capture_output=True, text=True)
    (directory / "stderr.txt").write_text(result.stderr)
    result.check_returncode()
    return read_sxa(directory / "output.sxa")[0][:, 0], read(directory / "runtime.json")


@torch.no_grad()
def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "renders").exists():
        raise ValueError("existing renders retained; no silent restart")
    prior = Path(config["prior"])
    records, all_support, parity = [], [], []
    for label in config["order"]:
        graph = config["models"][label]
        parent_label = "20261012-zero" if label == "identity_dsp" else label
        for case in config["cases"]:
            index = case["index"]
            samples = case["samples"]["calibration"]
            for role, identity in enumerate(case["identity"]):
                sample = samples[role % len(samples)]
                source = (
                    prior / "native-after" / parent_label / f"{index:02d}-{role:02d}" / "input.sxa"
                )
                pcm, rate = read_sxa(source)
                clean, target = target_for(sample, len(pcm), rate)
                if identity:
                    np.testing.assert_array_equal(pcm[:, 0], clean)
                directory = output / "renders" / label / f"{index:02d}-{role:02d}"
                final, runtime = execute(Path(config["binary"]), graph, source, directory, "stages")
                if label != "identity_dsp":
                    saved, _ = read_sxa(source.with_name("output.sxa"))
                    np.testing.assert_array_equal(final, saved[:, 0])
                if index == config["cases"][0]["index"] and role == 0:
                    normal, _ = execute(
                        Path(config["binary"]), graph, source, output / "trace-off" / label, None
                    )
                    np.testing.assert_array_equal(final, normal)
                traces = runtime["traces"]
                stages = {
                    name: np.asarray([h["dsp"][0][name] for h in traces], dtype=np.float32).reshape(
                        -1
                    )
                    for name in STAGES[:-1]
                }
                stages["native"] = final
                if any(len(w) != len(target) or not np.isfinite(w).all() for w in stages.values()):
                    raise ValueError("invalid stage PCM")
                support = slice(config["score_start_samples"], len(target) - 128)
                scores = {
                    name: {
                        k: v[0]
                        for k, v in assess_wave(
                            torch.from_numpy(wave[support])[None],
                            torch.from_numpy(target[support])[None],
                            torch.tensor([[rate]]),
                        ).items()
                    }
                    for name, wave in stages.items()
                }
                row = sample["row"]
                records.append(
                    {
                        "label": label,
                        "case": index,
                        "role": role,
                        "source": row["source_checksum"],
                        "source_id": row["source_id"],
                        "codec": row["codec_id"],
                        "group": row["corpus"] + ("/identity" if identity else "/paired"),
                        "stages": scores,
                    }
                )
                if label.endswith("-phase"):
                    zero = read(
                        output
                        / "renders"
                        / label.replace("-phase", "-zero")
                        / f"{index:02d}-{role:02d}"
                        / "runtime.json"
                    )["traces"]

                    def get_pred(ts):
                        return np.asarray([h["prediction"] for h in ts]).reshape(-1, 2, 129)[:-1]

                    weights = np.asarray([h["dsp"][0]["phase_weights"] for h in traces])
                    active = [h["dsp"][0]["active"] for h in traces]
                    if not np.array_equal(weights, [h["dsp"][0]["phase_weights"] for h in zero]):
                        raise ValueError("phase correction changed input-only weights")
                    result = correction_support(
                        get_pred(zero), get_pred(traces), weights[:-1], active[:-1], rate
                    )

                    def fused(ts):
                        return np.asarray(
                            [[h["dsp"][0]["fused_db"], h["dsp"][0]["fused_phase"]] for h in ts]
                        )[:-1]

                    actual = correction_support(
                        fused(zero), fused(traces), weights[:-1], active[:-1], rate
                    )
                    all_support.append(
                        {
                            "label": label,
                            "case": index,
                            "role": role,
                            "group": records[-1]["group"],
                            "source_id": row["source_id"],
                            "predicted": result,
                            "fused": actual,
                        }
                    )
                parity.append(
                    {
                        "label": label,
                        "case": index,
                        "role": role,
                        "old_pcm_exact": label != "identity_dsp",
                    }
                )
        print("stage review complete", label, flush=True)
    comparisons = {}
    for seed in (20261012, 20261013):
        comparisons[str(seed)] = {}
        for stage in STAGES:
            sets = [
                [
                    {
                        **{
                            k: r[k]
                            for k in ("case", "role", "source", "source_id", "codec", "group")
                        },
                        "scores": r["stages"][stage],
                    }
                    for r in records
                    if r["label"] == f"{seed}-{arm}"
                ]
                for arm in ("zero", "phase")
            ]
            comparisons[str(seed)][stage] = compare(*sets)
    verify(config["hashes"])
    write_json_report(
        {
            "records": records,
            "comparisons": comparisons,
            "correction_support": all_support,
            "parity": parity,
            "scope": config["scope"],
        },
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
    print("P66 complete; no new training or timing qualification", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.binary.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
