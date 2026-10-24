#!/usr/bin/env python3
"""Prepare a frozen P55 source-disjoint long-form review; wait for all P54 endpoints.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from data.audio_prep import align_codec_pair, encode_decode_ffmpeg, estimate_delay_samples
from data.protocol import load_recipe_from_profile
from data.source_quality import inspect_master
from evaluation.adapter_runtime import environment
from evaluation.longform_review import (
    aggregate,
    aligned_output,
    history_files,
    publisher_evidence,
    raw_from_trace,
    score_audio,
    select_sources,
    source_candidates,
)
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered

CODECS = ("mp3-cbr-64", "aac-lc-cbr-96", "vorbis-vbr-q3")


def receipt(paths, destination):
    registered({"files": {str(p): sha256_file(p) for p in paths if p.is_file()}}, destination)


def checked_receipt(path):
    verify(read(path)["files"])


def prepare(args):
    output = args.output.resolve()
    if (output / "registration.json").exists():
        config = read(output / "registration.json")
        if config["p54"] != str(args.p54.resolve()) or config["catalog"] != str(
            args.catalog.resolve()
        ):
            raise ValueError("review registration inputs changed")
        verify(config["hashes"])
        return config
    project = Path(__file__).resolve().parents[2]
    histories = history_files(project / "data")
    catalog = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    eligible, excluded = source_candidates(catalog, histories)
    reviewed = []
    for row in eligible:
        try:
            proof = publisher_evidence(row)
        except (ValueError, KeyError) as error:
            excluded.append({"id": row["id"], "reason": str(error)})
        else:
            reviewed.append({**row, "evaluation_rights_evidence": proof})
    sources = select_sources(reviewed)
    if len(sources) < 12:
        raise ValueError(
            "insufficient source-disjoint reviewed recordings; do not silently reuse tests"
        )
    old = read(args.common / "registration.json")
    p54 = read(args.p54 / "registration.json")
    models = {name: old["native"][name] for name in ("legacy", "identity_dsp")}
    for seed in p54["parents"]:
        for objective in ("parent", *p54["objectives"]):
            name = seed + "-" + objective
            models[name] = str(args.p54.resolve() / "exports" / name / "diagnostic.onnx")
    profile = project / "training/configs/music_library.yaml"
    ffmpeg = Path(args.ffmpeg).resolve()
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    paths = [
        *histories,
        args.catalog.resolve(),
        args.common.resolve() / "registration.json",
        args.p54.resolve() / "registration.json",
        profile,
        ffmpeg,
        Path(old["binary"]),
        *[Path(old["native"][name]) for name in ("legacy", "identity_dsp")],
    ]
    paths.extend(Path(r[k]) for r in sources for k in ("path", "source_metadata_path"))
    for directory in ("data", "evaluation", "models", "scripts"):
        paths.extend((project / "training" / directory).glob("*.py"))
    paths.append(project / "training/evaluation/listening.html")
    hashes = {str(p): sha256_file(p) for p in sorted(set(paths))}
    for source in sources:
        if hashes[str(Path(source["path"]))] != source["audio_sha256"]:
            raise ValueError("changed source master")
    config = {
        "format": "soundex-longform-p55-v1",
        "p54": str(args.p54.resolve()),
        "catalog": str(args.catalog.resolve()),
        "binary": old["binary"],
        "models": models,
        "ffmpeg": str(ffmpeg),
        "ffmpeg_version": version,
        "profile": str(profile),
        "sources": sources,
        "exclusions": excluded,
        "history_files": [str(p) for p in histories],
        "hashes": hashes,
        "selection": "At most four sources per catalog genre, at most three per exact author label. Rare genre first, SHA256([20261024, source hash]) order. Exclude every retained processed manifest and non-snapshot run-specific catalog via file/PCM/work aliases. No candidate outputs read. Artist-disjointness and unknown historical uses are not certified.",
        "isolation": "Source/work-disjoint against retained history, first registered use in this model review. Historical catalog exposure is conservatively excluded. Region labels missing from this catalog remain unknown; source pages alone do not establish national style.",
        "clips": "Two source positions at 25%/65% of available 13-second source support. Two seconds real prelude, ten scored seconds, one second codec/tail guard. Excerpt 0 at 44100 Hz, excerpt 1 at 48000 Hz. Original simultaneous physical channels. No per-condition gain normalization. No result-dependent crop replacement.",
        "codecs": list(CODECS),
        "score_seconds": 10,
        "prelude_seconds": 2,
        "coverage": {
            "sources": len(sources),
            "genres": dict(Counter(r.get("genre", "unlabeled") for r in sources)),
            "authors": len({r.get("author", "unknown") for r in sources}),
            "channels": dict(Counter(str(r["source_quality"]["channels"]) for r in sources)),
            "regions": dict(Counter(r.get("music_region", "unknown") for r in sources)),
            "unique_scored_source_seconds": 20 * len(sources),
            "paired_codec_seconds_per_path": 60 * len(sources),
            "identity_seconds_per_path": 20 * len(sources),
        },
        "metrics": "Primary diagnostic family: original-reference HF8k amplitude, log-magnitude and reference dynamics; equal 8-12/12-16/16+ kHz weights and existing three resolutions. Report all physical bands, MR, auraloss, calibrated absolute band error/reference RMS, waveform and simultaneous mid/side errors. Average channels, then excerpts/codecs within source, then sources equally. Clip means and genre/codec/rate/channel strata are sensitivities. Paired 5000 source-bootstrap intervals are descriptive, without multiplicity-adjusted superiority claims. No one-number release gate or automatic promotion.",
        "barrier": "No source scoring or listening pack until P54 complete.json and runtime-complete.json verify, all four endpoints plus parents and historical/identity graphs are frozen in endpoints.json. Training/loss/rate/budget unchanged. Quality traces are not timing qualification. Missing/failed cases stop completion; retain evidence.",
        "listening": "One preselected MP3 excerpt per source, excerpt 1 at 48 kHz; all four final candidates, historical, identity DSP, untreated, hidden reference and 3.5 kHz anchor. Common gain for all conditions per trial; original objective scores remain unscaled. Randomization seed 20261024, private answer key outside public pack. No perceptual result without human ratings. ViSQOL is not installed or asserted by this protocol.",
    }
    registered(config, output / "registration.json")
    return config


def prepare_pairs(config, output):
    if (output / "prepared.json").exists():
        checked_receipt(output / "prepared.json")
        return read(output / "cases.json")
    os.environ["PATH"] = str(Path(config["ffmpeg"]).parent) + os.pathsep + os.environ["PATH"]
    recipe = load_recipe_from_profile(Path(config["profile"]))
    codecs = [next(c for c in recipe.codecs if c.id == name) for name in CODECS]
    cases, artifacts = [], []
    for source_index, source in enumerate(config["sources"]):
        quality_path = output / "quality" / f"{source_index:02d}.json"
        if quality_path.exists():
            quality = read(quality_path)
        else:
            quality = inspect_master(Path(source["path"]), source["audio_sha256"])
            write_json_report(quality, quality_path)
        if quality["review_flags"] or quality["decoded_pcm_sha256"] != source["decoded_pcm_sha256"]:
            raise ValueError(
                "source quality changed; retain and review before using any alternate source"
            )
        artifacts.append(quality_path)
        for excerpt, rate in enumerate((44100, 48000)):
            name = f"{source_index:02d}-{excerpt}"
            directory = output / "pairs" / name
            done = directory / "complete.json"
            if done.exists():
                checked_receipt(done)
                cases.extend(read(directory / "cases.json"))
                artifacts.extend(Path(p) for p in read(done)["files"])
                artifacts.append(done)
                continue
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"incomplete paired preparation retained: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            original_rate = quality["sample_rate"]
            start = int(
                (quality["duration_seconds"] - 13)
                * (0.25 if excerpt == 0 else 0.65)
                * original_rate
            )
            original, sr = sf.read(
                source["path"],
                start=start,
                frames=13 * original_rate,
                dtype="float32",
                always_2d=True,
            )
            if sr != original_rate or len(original) != 13 * sr:
                raise ValueError("short original source support")
            common = math.gcd(sr, rate)
            clean = np.ascontiguousarray(
                resample_poly(original, rate // common, sr // common, axis=0), dtype=np.float32
            )
            clip_cases = []
            for codec in (*codecs, None):
                label = codec.id if codec else "identity"
                if codec:
                    pair = align_codec_pair(
                        clean, encode_decode_ffmpeg(clean, rate, codec), rate, recipe
                    )
                    target, degraded = pair.clean, pair.degraded
                    score_start = 2 * rate - max(0, -pair.offset_samples)
                    offset = pair.offset_samples
                else:
                    target = degraded = clean
                    score_start, offset = 2 * rate, 0
                residuals = [
                    estimate_delay_samples(
                        target[:, c],
                        degraded[:, c],
                        rate,
                        max_offset_samples=4,
                        correlation_samples=65536,
                    )
                    for c in range(clean.shape[1])
                ]
                if any(abs(r) > recipe.alignment.residual_tolerance_samples for r in residuals):
                    raise ValueError(
                        "per-channel alignment failed; no candidate-dependent realignment"
                    )
                length = (len(target) // 128) * 128
                reference_path = directory / f"{label}-reference.sxa"
                input_path = directory / f"{label}-input.sxa"
                write_sxa(reference_path, target[:length], rate)
                write_sxa(input_path, degraded[:length], rate)
                aligned_output(degraded[:length], score_start, 10 * rate)
                clip_cases.append(
                    {
                        "id": name + "-" + label,
                        "source": source["audio_sha256"],
                        "source_index": source_index,
                        "excerpt": excerpt,
                        "source_start_seconds": start / sr + 2,
                        "genre": source.get("genre", "unlabeled"),
                        "region": source.get("music_region", "unknown"),
                        "author": source.get("author", "unknown"),
                        "rate": rate,
                        "channels": target.shape[1],
                        "codec": label,
                        "role": "identity" if codec is None else "paired",
                        "reference": str(reference_path),
                        "input": str(input_path),
                        "score_start": score_start,
                        "score_length": 10 * rate,
                        "codec_offset_samples": offset,
                        "residual_offsets_per_channel": residuals,
                    }
                )
            write_json_report(clip_cases, directory / "cases.json")
            receipt([p for p in directory.iterdir() if p.is_file()], done)
            artifacts.extend([p for p in directory.iterdir() if p.is_file()])
            cases.extend(clip_cases)
            print("prepared", name, flush=True)
    write_json_report(cases, output / "cases.json")
    receipt([*artifacts, output / "cases.json"], output / "prepared.json")
    return cases


def endpoint_barrier(config, output):
    p54 = Path(config["p54"])
    if not all((p54 / n).exists() for n in ("complete.json", "runtime-complete.json")):
        return None
    for name in ("complete.json", "runtime-complete.json"):
        checked_receipt(p54 / name)
    expected = {
        seed + "-" + objective
        for seed in read(p54 / "registration.json")["parents"]
        for objective in ("common", "auraloss")
    }
    endpoints = read(p54 / "review-registration.json")
    if endpoints.keys() != expected:
        raise ValueError("P54 endpoint set changed")
    paths = [Path(p) for p in config["models"].values()]
    paths += [
        p54 / n for n in ("complete.json", "runtime-complete.json", "review-registration.json")
    ]
    frozen = {"models": config["models"], "files": {str(p): sha256_file(p) for p in paths}}
    registered(frozen, output / "endpoints.json")
    return frozen


def render(config, case, label, output):
    directory = output / "renders" / label / case["id"]
    if (directory / "complete.json").exists():
        checked_receipt(directory / "complete.json")
        return read(directory / "scores.json")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"incomplete render retained: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / "trace.json"
    native_path = directory / "native.sxa"
    result = subprocess.run(
        [
            config["binary"],
            config["models"][label],
            case["input"],
            str(native_path),
            str(trace_path),
            "1",
            "trace",
        ],
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    (directory / "stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"native render failed: {directory}; {result.stderr}")
    native, rate = read_sxa(native_path)
    target, reference_rate = read_sxa(Path(case["reference"]))
    runtime = read(trace_path)
    if (
        rate != reference_rate
        or rate != case["rate"]
        or native.shape != target.shape
        or not runtime["finite"]
        or runtime["dsp_delay_samples"] != 128
    ):
        raise ValueError("native geometry, finite or timeline mismatch")
    raw = raw_from_trace(runtime["traces"], case["channels"])
    if raw.shape != native.shape:
        raise ValueError("raw OLA timeline mismatch")
    write_sxa(directory / "raw.sxa", raw, rate)
    # Preserve predictions compactly; successful verbose frontend traces are regenerable.
    predictions = np.asarray([t["prediction"] for t in runtime["traces"]], dtype=np.float32)
    np.savez_compressed(directory / "predictions.npz", prediction=predictions)
    runtime.pop("traces")
    runtime["trace_retention"] = (
        "Raw predictions retained as float32 NPZ; redundant successful observed/context JSON removed after receipt. This trace run is not a timing benchmark. Failed traces are retained."
    )
    runtime["model_sha256"] = sha256_file(Path(config["models"][label]))
    runtime["input_sha256"] = sha256_file(Path(case["input"]))
    write_json_report(runtime, directory / "runtime.json")
    start, count = case["score_start"], case["score_length"]
    reference = target[start : start + count]
    scores = [
        {
            **case,
            "path": path + "/" + label,
            "scores": score_audio(aligned_output(wave, start, count), reference, rate),
        }
        for path, wave in (("native", native), ("raw", raw))
    ]
    write_json_report(scores, directory / "scores.json")
    receipt(
        [p for p in directory.iterdir() if p.is_file() and p != trace_path],
        directory / "complete.json",
    )
    trace_path.unlink()
    return scores


def review(config, cases, output):
    checked_receipt(output / "endpoints.json")
    records = []
    for case in cases:
        reference, rate = read_sxa(Path(case["reference"]))
        dry, _ = read_sxa(Path(case["input"]))
        start, count = case["score_start"], case["score_length"]
        records.append(
            {
                **case,
                "path": "untreated",
                "scores": score_audio(
                    dry[start : start + count], reference[start : start + count], rate
                ),
            }
        )
        for label in config["models"]:
            records.extend(render(config, case, label, output))
        write_json_report(
            {"stage": "review", "last_case": case["id"], "cases": len(cases)},
            output / "progress.json",
        )
        print("scored", case["id"], flush=True)
    write_json_report({"records": records, **aggregate(records)}, output / "summary.json")
    strata = {}
    for field in ("genre", "codec", "rate", "channels"):
        strata[field] = {
            str(value): aggregate([r for r in records if r[field] == value])
            for value in sorted({r[field] for r in records})
        }
    write_json_report(strata, output / "strata.json")
    from evaluation.longform_listening import build_pack

    build_pack(config, cases, output)
    receipt(
        [
            output / "registration.json",
            output / "prepared.json",
            output / "endpoints.json",
            output / "summary.json",
            output / "strata.json",
            *(output / "renders").rglob("complete.json"),
            *(output / "listening").rglob("*"),
            output / "listening-key.json",
        ],
        output / "complete.json",
    )
    write_json_report(
        {
            "stage": "complete",
            "qualified": False,
            "listening_ratings": "pending",
            "scope": "Frozen source-disjoint objective review and blinded materials, no human listening/device result or automatic weight promotion.",
        },
        output / "progress.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args)
        cases = prepare_pairs(config, output)
        if args.prepare_only:
            write_json_report(
                {"stage": "prepared", "cases": len(cases), "models_scored": 0},
                output / "progress.json",
            )
            return
        while endpoint_barrier(config, output) is None:
            write_json_report(
                {
                    "stage": "waiting_for_all_p54_endpoints_and_timing",
                    "cases": len(cases),
                    "models_scored": 0,
                },
                output / "progress.json",
            )
            if not args.wait:
                return
            time.sleep(30)
        verify(config["hashes"])
        # Waiting may last hours. Recheck generated pairs after the barrier too,
        # not only before waiting, so changed PCM cannot enter frozen scoring.
        checked_receipt(output / "prepared.json")
        if (output / "complete.json").exists():
            checked_receipt(output / "complete.json")
            return
        review(config, cases, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "p54", "common", "catalog"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        write_json_report(
            {"stage": "failed", "error": repr(error), "evidence_retained": True},
            args.output / "failure.json",
        )
        raise
