#!/usr/bin/env python3
"""P49 frozen adapter scale calibration and recording-disjoint TRAIN recheck.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import assert_frozen
from evaluation.adapter_scale import ARMIJO, FACTORS, FIXED_SCALE, calibrate
from evaluation.anchored_review import review_case
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from models.spectral_adapter import SUPPORTS, from_base
from scripts.audit_guarded_transfer import measured
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_adapter_scale import summarize
from scripts.summarize_guarded_transfer import coverage

INDICES = tuple(range(1, 128, 8))


def prepare(prior, output):
    path = output / "registration.json"
    if path.exists():
        config = read(path)
        if config["prior"] != str(prior):
            raise ValueError("prior changed")
        verify(config["hashes"])
        return config
    old = read(prior / "registration.json")
    verify(old["hashes"])
    verify(read(prior / "complete.json")["files"])
    paths = [Path(p) for p in old["hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    schedule = read(Path(old["schedule"]))
    groups = [{"index": i, "draws": old["groups"][i]} for i in INDICES]
    cases = []
    for group in groups:
        roles = {}
        for role, indices in (("fit", group["draws"][:4]), ("guard", group["draws"][4:])):
            roles[role] = [
                {"spec": schedule["specs"][j], "row": rows[schedule["specs"][j]["row_id"]]}
                for i in indices
                for j in schedule["draws"][i]
            ]
        for field in ("track_id", "source_checksum"):
            if {s["row"][field] for s in roles["fit"]} & {s["row"][field] for s in roles["guard"]}:
                raise ValueError("fit/guard recording overlap")
        if any(
            s["row"]["split"] != "train" or s["row"]["held_out"]
            for samples in roles.values()
            for s in samples
        ):
            raise ValueError("TRAIN roles required")
        cases.append({"index": group["index"], "samples": roles})
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(
            part in {".venv", "__pycache__", "outputs"} for part in p.relative_to(root).parts
        )
    ]
    sources += list((root / "evaluation").glob("*.json"))
    hashes = dict(old["hashes"])
    for p in sources:
        hashes[str(p)] = sha256_file(p)
        target = output / "source" / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    for p in (
        prior / "registration.json",
        prior / "complete.json",
        prior / "update-scale-diagnosis.json",
        prior / "analysis-integrity.json",
        prior / "preflight.json",
    ):
        hashes[str(p)] = sha256_file(p)
    for row in read(prior / "preflight.json")["rows"]:
        p = Path(row["onnx"]).parent / "smoke.pth"
        hashes[str(p)] = sha256_file(p)
    config = {
        "format": "soundex-adapter-scale-p49-v1",
        "prior": str(prior),
        "hashes": hashes,
        "starts": old["starts"],
        "schedule": old["schedule"],
        "groups": groups,
        "execution_control_draws": old["groups"][0][:4],
        "cases": cases,
        "coverage": {role: coverage(cases, role) for role in ("fit", "guard")},
        "supports": SUPPORTS,
        "factors": FACTORS,
        "armijo": ARMIJO,
        "fixed_scale": FIXED_SCALE,
        "objective": old["objective"],
        "optimizer": old["optimizer"],
        "scope": "16 groups indices 1,9,...121, excluding P48 post-hoc group 0. Each of two starts/full-high supports resets to zero adapter and fresh Adam. Independent first-step probes, not a training trajectory or convergence evidence. Reused TRAIN, not unseen recordings or test data.",
        "selection": "Fit-only Armijo backtracking on exact original total; largest factor with decrease >=max(c*alpha*(-g.dot(delta)),8*f32eps*abs(base)). Ten factors 1,1/4,...; explicit no-op on failure. Guard inaccessible during selection. Fixed 0.001 is a P48 diagnostic control, not a chosen production rate. Budget frozen/constant.",
        "barrier": "Freeze all 64 displacement hashes and fit-selected factors before reading any guard scores. Review initial, full Adam, fixed-scale and fit-backtracked changes with common raw spectra and original-reference 16/20/24-bit rulers.",
        "isolation": "Fit and guard disjoint by recording and source checksum within each group; recordings can recur across groups and were used in historical training. Retain original time-support exclusions. Four music/one speech per batch plus identities. No joint stereo or genre/region generalization claim.",
        "decision": "Calibration only; no default weight, runtime/DSP change or autonomous long-run launch from fit-only descent. Report failures, no-ops, numerical-scale changes, original-reference absolute PCM and all guard metrics. P48 timing evidence remains historical, not a new benchmark.",
    }
    verify(hashes)
    registered(config, path)
    return config


def execution_control(config, pool, schedule, output):
    path = output / "execution-control.json"
    if path.exists():
        if read(path)["signature"] != digest(config):
            raise ValueError("execution control drift")
        return
    items = [pool.select(schedule["draws"][i]) for i in config["execution_control_draws"]]
    rows = []
    for seed, entry in config["starts"].items():
        initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for support in SUPPORTS:
            model = from_base(initial, support).eval()
            report, delta = calibrate(model, items)
            old = torch.load(
                Path(config["prior"]) / "preflight" / f"{seed}-{support}" / "smoke.pth",
                map_location="cpu",
                weights_only=False,
            )
            if any(
                not torch.equal(change, old["model"]["adapter." + name])
                for name, change in delta.items()
            ):
                raise ValueError("P48 first Adam displacement does not reproduce exactly")
            if report["base"] != old["history"][0]["fit_before"]:
                raise ValueError("P48 fit objective does not reproduce exactly")
            assert_frozen(model, initial)
            rows.append(
                {
                    "seed": seed,
                    "support": support,
                    "p48_displacement_exact": True,
                    "fit_objective_exact": True,
                    "result": report,
                }
            )
    registered({"signature": digest(config), "rows": rows}, path)
    print("execution control: all four P48 displacements and fit objectives exact", flush=True)


def proposals(config, pool, schedule, output):
    entries = {}
    for seed, entry in config["starts"].items():
        initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for support in SUPPORTS:
            for group in config["groups"]:
                label = f"{seed}-{support}-{group['index']:03d}"
                directory = output / "proposals" / label
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "proposal.pth"
                signature = digest({"config": config, "label": label})
                receipt = directory / "complete.json"
                if receipt.exists():
                    saved = read(receipt)
                    if saved["signature"] != signature or saved["sha256"] != sha256_file(path):
                        raise ValueError("proposal drift")
                else:
                    if path.exists():
                        raise ValueError("incomplete proposal retained")
                    items = [pool.select(schedule["draws"][i]) for i in group["draws"][:4]]
                    model = from_base(initial, support).eval()
                    report, delta = calibrate(model, items)
                    assert_frozen(model, initial)
                    atomic_save({"report": report, "delta": delta, "signature": signature}, path)
                    registered({"signature": signature, "sha256": sha256_file(path)}, receipt)
                entries[label] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "seed": seed,
                    "support": support,
                    "index": group["index"],
                }
                write_json_report(
                    {
                        "stage": "fit_calibration",
                        "completed": len(entries),
                        "total": 64,
                        "last": label,
                    },
                    output / "progress.json",
                )
                print("fit", len(entries), "/ 64", label, flush=True)
    registered(entries, output / "proposal-registration.json")
    return entries


@torch.no_grad()
def recheck(config, entries, pool, schedule, output):
    reports = {}
    for label, entry in entries.items():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("frozen proposal drift")
        path = output / "review" / f"{label}.json"
        receipt = path.with_suffix("receipt.json")
        if receipt.exists():
            if read(receipt)["sha256"] != sha256_file(path):
                raise ValueError("guard review drift")
            reports[label] = read(path)
            continue
        if path.exists():
            raise ValueError("incomplete guard review retained")
        initial = torch.load(
            config["starts"][entry["seed"]]["path"], map_location="cpu", weights_only=False
        )
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        group = next(g for g in config["groups"] if g["index"] == entry["index"])
        item = pool.select(schedule["draws"][group["draws"][4]])
        samples = next(c for c in config["cases"] if c["index"] == entry["index"])["samples"][
            "guard"
        ]
        model = from_base(initial, entry["support"]).eval()
        anchor = supervised_waves(predict_budget(model, item[0].batch, item[1])[0], item[0].batch)[
            0
        ]
        cases = {}
        for arm, scale in {
            "start": 0.0,
            "adam": 1.0,
            "fixed": FIXED_SCALE,
            "backtrack": saved["report"]["search"]["selected_scale"],
        }.items():
            for name, p in model.adapter.named_parameters():
                p.copy_(saved["delta"][name] * scale)
            case = {
                "index": entry["index"],
                "samples": {"calibration": samples},
                "metrics": measured(model, item),
                **review_case(model, item),
            }
            wave = supervised_waves(
                predict_budget(model, item[0].batch, item[1])[0], item[0].batch
            )[0]
            case["reference_rulers"] = {
                str(b): pcm_guard([wave], [anchor], [item], b) for b in (16, 20, 24)
            }
            cases[arm] = case
        assert_frozen(model, initial)
        result = {"proposal_sha256": entry["sha256"], "fit": saved["report"], "guard": cases}
        write_json_report(result, path)
        registered({"sha256": sha256_file(path)}, receipt)
        reports[label] = result
        write_json_report(
            {"stage": "guard_review", "completed": len(reports), "total": 64},
            output / "progress.json",
        )
        print("guard", len(reports), "/ 64", label, flush=True)
    return reports


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), output)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            print("P49 complete", flush=True)
            return
        paths = [Path(p) for p in config["hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            output / "feature-cache",
            capacity=320,
        )
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(schedule["specs"], clips)
        execution_control(config, pool, schedule, output)
        entries = proposals(config, pool, schedule, output)
        reports = recheck(config, entries, pool, schedule, output)
        write_json_report(summarize(config, entries, reports), output / "summary.json")
        verify(config["hashes"])
        files = [
            output / "summary.json",
            output / "proposal-registration.json",
            output / "execution-control.json",
            *(output / "review").glob("*.json"),
        ]
        registered(
            {
                "registration_sha256": sha256_file(output / "registration.json"),
                "files": {str(p): sha256_file(p) for p in files},
            },
            output / "complete.json",
        )
        write_json_report({"stage": "complete", "probes": 64}, output / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
