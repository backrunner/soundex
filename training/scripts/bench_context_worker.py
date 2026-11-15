"""P77 once-only frozen host callback/worker grid; no device qualification.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import platform
import re
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import write_sxa
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt

HOPS = 1100
WARMUP = 100


def summarize(report):
    events = report["events"]
    if len(events) != HOPS or [e["hop"] for e in events] != list(range(HOPS)):
        raise ValueError("callback clock changed")
    bits = [int(e["worker_presentation_missed"]) for e in events]
    cumulative, burst, longest = 0, 0, 0
    for event, missed in zip(events, bits, strict=True):
        cumulative += missed
        burst = burst + 1 if missed else 0
        longest = max(longest, burst)
        if event["deadline_misses"] != cumulative:
            raise ValueError("raw presentation bits disagree with cumulative misses")
    if (
        cumulative != report["deadline_misses"]
        or longest != report["max_consecutive_deadline_misses"]
    ):
        raise ValueError("whole-run availability counters disagree")
    scored = events[WARMUP:]
    burst, longest = 0, 0
    for event in scored:
        burst = burst + 1 if event["worker_presentation_missed"] else 0
        longest = max(longest, burst)
    callback = np.asarray([e["callback_ns"] for e in scored]) / 1e6
    misses = sum(e["worker_presentation_missed"] for e in scored)
    result = {
        "scored_hops": len(scored),
        "misses": misses,
        "miss_ratio": misses / len(scored),
        "max_miss_run": longest,
        "max_miss_run_ms": longest * 128000 / report["sample_rate"],
        "callback_p99_ms": float(np.quantile(callback, 0.99)),
        "callback_max_ms": float(callback.max()),
        "worker_max_ms_all_input_hops": report["max_processing_ns"] / 1e6,
        "queue_wait_max_ms_all_input_hops": report["max_queue_wait_ns"] / 1e6,
        "producer_start_lateness_max_ms": max(e["callback_start_late_ns"] for e in scored) / 1e6,
        "state_history_warmup_hops": report["state_history_warmup_hops"],
        "processing_period_overruns": report["processing_period_overruns"],
        "whole_run_misses": cumulative,
        "normal_miss_ratio": sum(bits[2:]) / (len(bits) - 2),
        "whole_run_max_miss_run_ms": max(e["max_consecutive_deadline_misses"] for e in events)
        * 128000
        / report["sample_rate"],
        "whole_run_max_miss_run": max(e["max_consecutive_deadline_misses"] for e in events),
    }
    result["software_screen_pass"] = (
        report["finite"]
        and report["worker_drain_complete"]
        and not report["worker_failed"]
        and report["output_frames"] == HOPS * 128
        and report["fixed_delay_frames"] == 256
        and result["callback_p99_ms"] < 0.1
        and result["callback_max_ms"] < 0.5
        and result["miss_ratio"] <= 0.001
        and result["max_miss_run_ms"] <= 20
        and result["normal_miss_ratio"] <= 0.001
        and result["whole_run_max_miss_run_ms"] <= 20
        and result["worker_max_ms_all_input_hops"] < 5
        and result["processing_period_overruns"] == 0
    )
    return result


def prepare(prior, binary, output):
    if output.exists():
        raise ValueError("fresh grid required; retain all timings")
    output.mkdir(parents=True)
    checked_receipt(prior / "complete.json")
    config = read(prior / "registration.json")
    graphs = {
        label: prior / "exports" / label / "parent/diagnostic.onnx" for label in config["endpoints"]
    }
    repo = Path(__file__).resolve().parents[2]
    frozen_binary = output / binary.name
    shutil.copy2(binary, frozen_binary)
    source = output / "source"
    for path in [
        repo / "Cargo.toml",
        repo / "Cargo.lock",
        Path(__file__),
        *(p for p in (repo / "crates").rglob("*") if p.suffix in {".rs", ".toml"}),
    ]:
        destination = source / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    inputs, provenance = {}, {}
    for rate, channels in itertools.product((44100, 48000), (1, 2)):
        t = np.arange(HOPS * 128) / rate
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
        waves = []
        for channel in range(channels):
            wave = (
                sum(
                    0.04 * np.sin(2 * np.pi * f * t + 0.6 * channel) for f in (173, 431, 1291, 3917)
                )
                * envelope
            )
            wave[(np.arange(len(t)) // 128) % 300 < 30] = 0
            if channel == 0 and channels == 2:
                wave[len(t) // 2 : 3 * len(t) // 4] = 0
            wave[80 * 128 :: 200 * 128] += 0.1
            waves.append(wave)
        path = output / f"synthetic-{rate}-{channels}.sxa"
        write_sxa(path, np.stack(waves, 1).astype(np.float32), rate)
        inputs[path.stem] = str(path)
    # Use actual mono music channel views; never join unrelated excerpts as stereo.
    records = read(prior / "records.json")
    for rate in (44100, 48000):
        for row in records:
            sample = row["sample"]["row"]
            if sample["corpus"] != "music_library" or sample["sample_rate"] != rate:
                continue
            path = Path(sample["_root"]) / sample["degraded_path"]
            info = sf.info(path)
            if info.channels != 1 or info.frames < HOPS * 128:
                continue
            audio, actual_rate = sf.read(path, frames=HOPS * 128, dtype="float32", always_2d=True)
            if actual_rate != rate:
                raise ValueError("source rate mismatch")
            destination = output / f"music-{rate}-1.sxa"
            write_sxa(destination, audio, rate)
            inputs[destination.stem] = str(destination)
            provenance[destination.stem] = {
                "source": str(path),
                "sha256": sha256_file(path),
                "start": 0,
                "sample": sample,
                "scope": "Reused P76 TRAIN source, one stored mono view (see channel_role; may be derived mid), not physical stereo evidence",
            }
            break
        else:
            raise ValueError(
                "both physical rates require an existing sufficiently long mono music view"
            )
    schedule = []
    for index, (input_name, label) in enumerate(itertools.product(inputs, graphs)):
        for policy in ("QOS", "RT") if index % 2 == 0 else ("RT", "QOS"):
            schedule.append({"input": input_name, "model": label, "policy": policy})
    hashes = {
        str(p): sha256_file(p)
        for p in [
            prior / "complete.json",
            *graphs.values(),
            frozen_binary,
            *(Path(p) for p in inputs.values()),
            *(p for p in source.rglob("*") if p.is_file()),
        ]
    }
    hashes.update({p["source"]: p["sha256"] for p in provenance.values()})
    registered(
        {
            "format": "soundex-context-worker-p77-v1",
            "binary": str(frozen_binary),
            "graphs": {k: str(p) for k, p in graphs.items()},
            "inputs": inputs,
            "provenance": provenance,
            "schedule": schedule,
            "hashes": hashes,
            "machine": platform.platform(),
            "hops": HOPS,
            "warmup_excluded": WARMUP,
            "protocol": "48 fixed cases, four frozen P74 parent graphs, four synthetic rate/channel inputs and two real mono music views; QOS/RT order alternates by input/model. One run each, 1100 submitted/1000 scored callbacks; retain whole-run and post100-hop miss bits. Availability must pass both normal (only first2 startup hops excluded) and post100 windows. No retries, no concurrent training/build. Same fixed256-frame delay, <5ms compute, zero hop compute overruns, callback p99<100us/max<500us, miss ratio<=0.1%, burst<=20ms. OS acceptance does not establish live realtime priority.",
            "scope": "Actual shared callback/worker, host simulation only. Resource time includes load/teardown/IO, not isolated per-thread CPU. No physical device, listening or P77 candidate-weight timing qualification.",
        },
        output / "registration.json",
    )
    print("registered", len(schedule), "worker cases", flush=True)


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "cases").exists():
        raise ValueError("existing grid retained; no retry-to-pass")
    results = []
    for index, case in enumerate(config["schedule"]):
        destination = output / "cases" / f"{index:02d}"
        destination.mkdir(parents=True)
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-l",
                config["binary"],
                config["graphs"][case["model"]],
                config["inputs"][case["input"]],
                str(destination / "output.sxa"),
                str(destination / "raw.json"),
                case["policy"],
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (destination / "time.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"worker case{index} failed; retain outputs and stderr")
        report = read(destination / "raw.json")
        assessment = {
            **case,
            **summarize(report),
            "rt_accepted": report["time_constraint_accepted"],
        }
        memory = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
        cpu = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", result.stderr)
        if memory is None or cpu is None or float(cpu[1]) <= 0:
            raise ValueError("missing process CPU/RSS evidence; retain case")
        assessment.update(
            peak_rss_bytes=int(memory[1]),
            process_wall_seconds=float(cpu[1]),
            cpu_user_seconds=float(cpu[2]),
            cpu_system_seconds=float(cpu[3]),
            process_one_core_percent=100 * (float(cpu[2]) + float(cpu[3])) / float(cpu[1]),
        )
        registered(assessment, destination / "assessment.json")
        results.append(assessment)
        write_json_report({"cases": results, "complete": False}, output / "progress.json")
        print(
            "worker case",
            index,
            assessment["misses"],
            "misses;",
            assessment["software_screen_pass"],
            flush=True,
        )
    verify(config["hashes"])
    registered(
        {
            "cases": results,
            "all_software_screens_pass": all(r["software_screen_pass"] for r in results),
            "scope": config["scope"],
            "qualified": False,
        },
        output / "assessment.json",
    )
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.log"
            }
        },
        output / "complete.json",
    )


def review(output, destination):
    """Reconcile retained cases without rerunning or overwriting failed evidence."""
    if destination.exists():
        raise ValueError("fresh read-only review required")
    destination.mkdir(parents=True)
    source = destination / Path(__file__).name
    shutil.copy2(Path(__file__), source)
    config = read(output / "registration.json")
    verify(config["hashes"])
    results = []
    for index, case in enumerate(config["schedule"]):
        directory = output / "cases" / f"{index:02d}"
        report = read(directory / "raw.json")
        resources = (directory / "time.txt").read_text()
        memory = re.search(r"(\d+)\s+maximum resident set size", resources)
        cpu = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", resources)
        if memory is None or cpu is None or float(cpu[1]) <= 0:
            raise ValueError("missing retained process resources")
        rejected = report["events"][-1]["queue_overflows"]
        result = {
            "index": index,
            **case,
            **summarize(report),
            "reported_drain_complete": report["worker_drain_complete"],
            "rejected_input_hops": rejected,
            "expected_enqueued_hops": HOPS - rejected,
            "accounted_input_hops": report["processed_hops"] + report["discarded_input_hops"],
            "input_accounting_complete": report["processed_hops"] + report["discarded_input_hops"]
            == HOPS - rejected,
            "finite": report["finite"],
            "output_frames": report["output_frames"],
            "fixed_delay_frames": report["fixed_delay_frames"],
            "worker_failed": report["worker_failed"],
            "rt_accepted": report["time_constraint_accepted"],
            "peak_rss_bytes": int(memory[1]),
            "process_wall_seconds": float(cpu[1]),
            "cpu_user_seconds": float(cpu[2]),
            "cpu_system_seconds": float(cpu[3]),
            "process_one_core_percent": 100 * (float(cpu[2]) + float(cpu[3])) / float(cpu[1]),
        }
        results.append(result)
    registered(
        {
            "cases": results,
            "all_software_screens_pass": all(r["software_screen_pass"] for r in results),
            "original_completion_failure_retained": (output / "failure.json").exists(),
            "scope": "Read-only accounting of all48 original cases, no retries or changed pass rules. Frozen binary checked drain against attempted rather than enqueued hops; rejected input explains case47's false drain failure, but its availability still fails. Historical maxima were read before worker join and may race the last metric update; they are observed values, not final-call compute certification. Stored mono music views can be derived mid, not physical stereo. No device or listening qualification.",
            "qualified": False,
        },
        destination / "assessment.json",
    )
    evidence = [p for p in output.rglob("*") if p.is_file() and destination not in p.parents]
    evidence.extend((source, destination / "assessment.json"))
    registered({"files": {str(p): sha256_file(p) for p in evidence}}, destination / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "review"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    args = parser.parse_args()
    if args.mode == "review" and args.review_output is None:
        parser.error("review requires --review-output")
    target = args.review_output if args.mode == "review" else args.output
    fresh = not target.exists()
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.binary.resolve(), args.output.resolve())
        elif args.mode == "run":
            run(args.output.resolve())
        else:
            review(args.output.resolve(), args.review_output.resolve())
    except BaseException:
        failure = target / "failure.json"
        if target.exists() and not failure.exists() and (args.mode == "run" or fresh):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
