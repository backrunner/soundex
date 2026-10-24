"""P48 registration, matched adapter updates and frozen original-reference review.

SPDX-License-Identifier: Apache-2.0
"""

import shutil
from pathlib import Path

import torch

from data.broad_coverage import LazyPool
from evaluation.anchored_review import compare, review_case
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import load_restoration_head, predict_budget
from models.spectral_adapter import SUPPORTS, load_adapter
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import metrics
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered

STEPS = 128


def prepare(prior: Path, evidence: Path, binary: Path, assets: Path, output: Path) -> dict:
    destination = output / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["prior"] != str(prior) or config["evidence"] != str(evidence):
            raise ValueError("resume inputs changed")
        verify(config["hashes"])
        return config
    old = read(prior / "registration.json")
    verify(old["hashes"])
    receipt = read(prior / "complete.json")
    verify({str(prior / k): h for k, h in receipt["files"].items()})
    audit = read(evidence / "complete.json")
    verify({str(evidence / "summary.json"): audit["summary_sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    hashes = dict(old["hashes"])
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(x in {".venv", "__pycache__", "outputs"} for x in p.relative_to(root).parts)
    ]
    sources += list((root / "evaluation").glob("*.json"))
    for path in sources:
        hashes[str(path)] = sha256_file(path)
        target = output / "source" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in [
        prior / "registration.json",
        prior / "complete.json",
        evidence / "registration.json",
        evidence / "complete.json",
        evidence / "summary.json",
    ]:
        hashes[str(path)] = sha256_file(path)
    frozen = output / "binaries" / binary.name
    frozen.parent.mkdir()
    shutil.copy2(binary, frozen)
    hashes[str(frozen)] = sha256_file(frozen)
    for rate in (44100, 48000):
        for channels in (1, 2):
            path = assets / f"input-{rate}-{channels}.sxa"
            hashes[str(path)] = sha256_file(path)
    config = {
        "format": "soundex-adapter-p48-v1",
        "prior": str(prior),
        "evidence": str(evidence),
        "steps": STEPS,
        "supports": SUPPORTS,
        "starts": old["endpoints"],
        "groups": old["groups"],
        "schedule": old["schedule"],
        "coverage": old["coverage"],
        "calibration": old["calibration"],
        "hashes": hashes,
        "binary": str(frozen),
        "assets": str(assets),
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "gradient_clip": 5,
            "state": "fresh, adapter parameters only",
        },
        "objective": "unchanged P38/P45 metrics.total: equal paired/identity mean(log1p(direct)+log1p(MR)) plus 0.1 budget auxiliary. Base/budget fully frozen, auxiliary identical and has no adapter gradient; not an acceptance gate. Original unclipped clean targets, spectra primary.",
        "method": "one frozen causal feature pass and original decoder, plus zero-initialized 192-to-258 adapter in fixed input-only residual units. Full versus high support only; high smoothstep 4-8k, zero <=4k, one >=8k. Same nominal 49,794 adapter parameters; masked low rows have zero effective gradient in high arm.",
        "updates": "128 actual Adam steps, four equal fit batches each; fifth source-disjoint guard batch diagnostic only. No proposal veto, no learning-rate search, early selection, changed budget or weight promotion. Not full convergence/capacity training. P46 attempted exposure is now effective fit exposure; old Adam comparison is contextual because optimizer history differs.",
        "scope": "post-hoc mechanism pilot using same P45 TRAIN schedule and P44 reused TRAIN recheck. 4 music+1 speech per paired batch and clean identity counterparts. No independent test or simultaneous stereo music claim.",
        "barrier": "finish all four trajectories, freeze hashes, then score all endpoints together against original clean, starting models and untreated input. No intermediate review-driven selection.",
        "runtime": "preflight each seed/support after one real update: nonzero export/state parity and 3000 native 48k stereo hops. Throughput feasibility requires p99 below hop; max >=5ms retained and prevents timing qualification, not hidden. Final four endpoints: 3000 hops at both rates/mono/stereo plus 1000 paced 48k stereo. No concurrent training/tests/build during timing. Fixed 256-frame buffering excluded from compute and disclosed.",
        "native_review": "eight equally spaced P44 batches indices 0,4,...28, all paired/identity mono views, both starts/four endpoints/identity-through-real-DSP control. Preserve PCM and raw reports. Default Rust DSP remains unchanged. Not callback/device qualification.",
    }
    verify(hashes)
    registered(config, destination)
    return config


def step(model, items: list, optimizer) -> dict:
    optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in model.adapter.parameters()]
    fit = {}
    for objective, context in items[:4]:
        values, _, _ = metrics(model, objective, context)
        for key, value in values.items():
            fit[key] = fit.get(key, 0) + float(value.detach()) / 4
        (values["total"] / 4).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.adapter.parameters()):
        raise ValueError("nonfinite adapter update")
    delta = sum(
        (p.detach() - old).square().sum()
        for p, old in zip(model.adapter.parameters(), before, strict=True)
    ).sqrt()
    return {
        "fit_before": fit,
        "guard_after": measured(model, items[4]),
        "gradient_norm": float(norm),
        "parameter_delta_l2": float(delta),
    }


def assert_frozen(model, initial: dict) -> None:
    if any(not torch.equal(model.state_dict()[k], v) for k, v in initial["model"].items()):
        raise ValueError("frozen base changed")
    if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith("adapter.")):
        raise ValueError("gradient reached frozen base")


def load_endpoint(entry: dict):
    checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
    return (
        load_adapter(checkpoint)
        if checkpoint.get("format") == "soundex-spectral-adapter-v1"
        else load_restoration_head(checkpoint)
    ).eval()


@torch.no_grad()
def review(config: dict, endpoints: dict, clips, output: Path) -> tuple:
    original = next(iter(read(Path(config["calibration"]))["cases"].values()))
    specs = [s["spec"] for case in original for s in case["samples"]["calibration"]]
    pool = LazyPool(specs, clips)
    items = [pool.select(list(range(i * 5, i * 5 + 5))) for i in range(len(original))]
    reports, anchors = {}, {}
    for seed, entry in config["starts"].items():
        model = load_endpoint(entry)
        anchors[seed] = [
            supervised_waves(predict_budget(model, item[0].batch, item[1])[0], item[0].batch)[0]
            for item in items
        ]
    for label, entry in endpoints.items():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("endpoint drift")
        model = load_endpoint(entry)
        seed = next(s for s in config["starts"] if label == s or label.startswith(s + "-"))
        cases = []
        for index, item in enumerate(items):
            case = {
                "index": original[index]["index"],
                "samples": original[index]["samples"],
                **review_case(model, item),
            }
            prediction, _ = predict_budget(model, item[0].batch, item[1])
            wave, _ = supervised_waves(prediction, item[0].batch)
            rulers = {
                str(b): pcm_guard([wave], [anchors[seed][index]], [item], b) for b in (16, 20, 24)
            }
            case["reference_rulers"] = rulers
            cases.append(case)
        reports[label] = cases
        write_json_report(
            {"checkpoint_sha256": entry["sha256"], "cases": cases},
            output / "review" / f"{label}.json",
        )
        print("raw review", label, "complete", flush=True)
    comparisons = {}
    for seed in config["starts"]:
        for support in SUPPORTS:
            label = seed + "-" + support
            comparisons[label] = {
                "against_start": compare(reports[label], reports[seed]),
                "against_full_adapter": compare(reports[label], reports[seed + "-full"]),
            }
    write_json_report(comparisons, output / "assessment.json")
    write_json_report(
        {label: sha256_file(output / "review" / f"{label}.json") for label in reports},
        output / "review-hashes.json",
    )
    return items, original
