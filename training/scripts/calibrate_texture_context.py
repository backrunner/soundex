"""P60 matched 2x2 critic study: observed-band summaries and structured negatives."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from evaluation.texture_calibration import aggregate, assess, controls, measure
from evaluation.trainability_audit import assert_reserved_isolation
from models.adapter_trainability import load_trainable
from models.context_texture import ContextTextureCritic, shifted_high, structured_loss
from scripts.calibrate_spectral_texture import item_for, metadata, payload, progress, sources
from scripts.preflight_spectral_texture import read, verify
from scripts.run_causal_study import atomic_save

STEPS = 1024
SEED = 20261026
ARMS = {
    "high_prediction": {"context": False, "structured": False},
    "context_prediction": {"context": True, "structured": False},
    "high_structured": {"context": False, "structured": True},
    "context_structured": {"context": True, "structured": True},
}


def prepare(parent, output):
    config = read(parent / "registration.json")
    verify(read(parent / "complete.json")["files"])
    verify(config["hashes"])
    hashes = dict(config["hashes"])
    catalogs = sorted((parent.parent).glob("*/evaluation-catalog.jsonl"))
    reserved = [json.loads(s) for p in catalogs for s in p.read_text().splitlines() if s]
    catalog = parent.parents[1] / "catalogs/current-approved.jsonl"
    universe = list(sources(config).values()) + [
        json.loads(s) for s in catalog.read_text().splitlines() if s
    ]
    assert_reserved_isolation(list(sources(config).values()), universe, reserved)
    assert_reserved_isolation(
        [s["row"] for s in config["roles"]["optimization"]["samples"]],
        universe,
        [s["row"] for s in config["roles"]["calibration"]["samples"]],
    )
    for p in (parent / "registration.json", parent / "complete.json", catalog, *catalogs):
        hashes[str(p)] = sha256_file(p)
    caches = {}
    for label, entry in config["endpoints"].items():
        caches[label] = []
        for index in range(256):
            path = parent / "training" / label / f"fit-{index:03}.pth"
            receipt = read(path.with_suffix(".json"))
            if receipt["generator_sha256"] != entry["sha256"] or receipt[
                "registration_sha256"
            ] != sha256_file(parent / "registration.json"):
                raise ValueError("unbound parent PCM cache")
            verify({str(path): receipt["sha256"]})
            hashes[str(path)] = receipt["sha256"]
            hashes[str(path.with_suffix(".json"))] = sha256_file(path.with_suffix(".json"))
            caches[label].append(str(path))
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for p in root.rglob("*"):
        relative = p.relative_to(root)
        if p.suffix not in (".py", ".json") or any(
            x in {".venv", "__pycache__", "outputs", ".pytest_cache", ".ruff_cache"}
            for x in relative.parts
        ):
            continue
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
        hashes[str(p)] = sha256_file(p)
        hashes[str(destination)] = sha256_file(destination)
    registration = {
        "format": "soundex-texture-context-p60-v1",
        "parent": str(parent),
        "hashes": hashes,
        "parent_config": config,
        "caches": caches,
        "arms": ARMS,
        "steps": STEPS,
        "seed": SEED,
        "protocol": "Eight critics: four arms per frozen P57 generator. Identical 3970-parameter initialization (zero added context weights), fresh Adam lr1e-4 betas .9/.999 eps1e-8 clip5, 1024 steps cycling the exact P59 256-draw bank four times. Three fake forwards in every arm; total fake weight one. No generator training or intermediate scoring. Freeze all eight endpoints first.",
        "context": "Eight 3kHz observed log-magnitude band means broadcast to every candidate frequency at the same analysis time. Full-band summaries, not full spectral detail. Off arms have zero context inputs and the same allocated parameters.",
        "negative": "Structured arms mix prediction/phase/time at .5/.25/.25, skip roundtrip-equivalent synthetic negatives and renormalize. Phase seed changes per update; time fractions cycle 1/8,1/6,1/3. Controls repeat prediction in all three slots. All conditioning is observed PCM.",
        "review": "P59 16 fit/16 calibration draws, same source-equal controls plus 3/8 time shift never trained. P59 quarter-shift also not a training fraction. Calibration is an exposed recheck, not independent testing. Initial random features and per-corpus fit-normalized feature distances retained. No calibration-tuned coefficient.",
        "gate": "Unchanged P59 prediction logistic improvement and positive margin/strict source majority for prediction, energy-matched, phase, quarter-shift, additionally unseen 3/8 shift, separately per corpus and seed. Both generator seeds must pass for any arm. Mechanism prerequisite only, no automatic generator launch or winner from submetrics.",
    }
    write_json_report(registration, output / "registration.json")
    return registration


def train(config, output):
    entries = {}
    for generator, paths in config["caches"].items():
        data = [torch.load(p, weights_only=False) for p in paths]
        for arm, settings in config["arms"].items():
            label = generator + "__" + arm
            root = output / "training" / label
            root.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(SEED)
            critic = ContextTextureCritic(settings["context"])
            optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
            initial = root / "initial.pth"
            if not initial.exists():
                atomic_save({"model": critic.state_dict()}, initial)
            state = torch.load(initial, weights_only=False)["model"]
            if any(not torch.equal(v, state[k]) for k, v in critic.state_dict().items()):
                raise ValueError("changed initial critic")
            history, start = [], 0
            latest = root / "latest.pth"
            if latest.exists():
                verify({str(latest): read(root / "latest.json")["sha256"]})
                state = torch.load(latest, weights_only=False)
                if state["registration_sha256"] != sha256_file(output / "registration.json"):
                    raise ValueError("changed resume registration")
                critic.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                history, start = state["history"], state["step"]
            for step in range(start, STEPS):
                optimizer.zero_grad(set_to_none=True)
                loss = structured_loss(
                    critic,
                    *data[step % 256],
                    structured=settings["structured"],
                    seed=SEED + step,
                    step=step,
                ).mean()
                loss.backward()
                if not torch.isfinite(loss) or any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()
                ):
                    raise ValueError("nonfinite critic gradient")
                norm = torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in critic.parameters()):
                    raise ValueError("nonfinite critic update")
                history.append(
                    {"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm)}
                )
                if (step + 1) % 64 == 0:
                    atomic_save(
                        {
                            "model": critic.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "history": history,
                            "step": step + 1,
                            "registration_sha256": sha256_file(output / "registration.json"),
                        },
                        latest,
                    )
                    write_json_report({"sha256": sha256_file(latest)}, root / "latest.json")
                    progress(
                        output,
                        stage="critic_training",
                        generator=generator,
                        arm=arm,
                        step=step + 1,
                        budget=STEPS,
                        completed_critics=len(entries),
                    )
            endpoint = root / "endpoint.pth"
            if not endpoint.exists():
                shutil.copy2(latest, endpoint)
            if sha256_file(endpoint) != sha256_file(latest):
                raise ValueError("changed frozen endpoint")
            entries[label] = {
                "generator": generator,
                "arm": arm,
                **settings,
                "initial": {"path": str(initial), "sha256": sha256_file(initial)},
                "trained": {"path": str(endpoint), "sha256": sha256_file(endpoint)},
            }
    write_json_report(entries, output / "critics.json")
    return entries


def load_critic(entry, version):
    item = entry[version]
    verify({item["path"]: item["sha256"]})
    critic = ContextTextureCritic(entry["context"])
    critic.load_state_dict(torch.load(item["path"], weights_only=False)["model"])
    return critic


def review(config, entries, output):
    parent = config["parent_config"]
    clips = BoundedEnvelopeClips(
        sources(parent), Path(parent["detector"]), output / "cache", capacity=40
    )
    records, condition_checks = [], []
    for generator, original in parent["endpoints"].items():
        checkpoint = torch.load(original["path"], map_location="cpu", weights_only=False)
        model = load_trainable(checkpoint, "full")
        data = {"optimization": {}, "calibration": {}}
        for index in parent["fit_review_indices"]:
            data["optimization"][index] = torch.load(
                config["caches"][generator][index], weights_only=False
            )
        for index in range(16):
            data["calibration"][index] = payload(
                model, item_for(parent, clips, "calibration", index)
            )
        if any(not torch.equal(v, checkpoint["model"][k]) for k, v in model.state_dict().items()):
            raise ValueError("frozen generator changed")
        for label, entry in entries.items():
            if entry["generator"] != generator:
                continue
            critics = {v: load_critic(entry, v) for v in ("initial", "trained")}
            for role, draws in data.items():
                for index, values in draws.items():
                    predicted, reference, observed, rates = values
                    variants = controls(*values, SEED + index)
                    variants["time_shift_unseen"] = shifted_high(reference, rates, 3 / 8)
                    meta = metadata(parent, role, index)
                    for version, critic in critics.items():
                        for name, wave in variants.items():
                            scores = measure(critic, wave, reference, observed, rates)
                            for i, sample in enumerate(meta):
                                record = {
                                    "generator": label,
                                    "critic": version,
                                    "role": role,
                                    "corpus": sample["row"]["corpus"],
                                    "source": sample["row"]["source_checksum"],
                                    "draw": index,
                                    "view": sample["row"]["channel_role"],
                                    "control": name,
                                    "scores": {k: float(v[i]) for k, v in scores.items()},
                                }
                                if not all(math.isfinite(v) for v in record["scores"].values()):
                                    raise ValueError("nonfinite review")
                                records.append(record)
                        # Same observed samples/RMS, wrong temporal alignment. This
                        # probes condition use; it is not an audio-quality gate.
                        original_score = measure(critic, predicted, reference, observed, rates)
                        shifted_score = measure(
                            critic,
                            predicted,
                            reference,
                            observed.roll(observed.shape[-1] // 3, -1),
                            rates,
                        )
                        for i, sample in enumerate(meta):
                            condition_checks.append(
                                {
                                    "generator": label,
                                    "critic": version,
                                    "role": role,
                                    "source": sample["row"]["source_checksum"],
                                    "corpus": sample["row"]["corpus"],
                                    "draw": index,
                                    "margin_change": float(
                                        shifted_score["margin"][i] - original_score["margin"][i]
                                    ),
                                    "feature_change": float(
                                        shifted_score["feature_loss"][i]
                                        - original_score["feature_loss"][i]
                                    ),
                                }
                            )
                progress(output, stage="review", label=label, role=role, records=len(records))
    groups = aggregate(records)
    assessment = assess(groups, entries)
    for label in entries:
        for corpus in ("music_library", "speech_library"):
            group = groups[f"{label}/trained/calibration/{corpus}/time_shift_unseen"]
            rounding = groups[f"{label}/trained/calibration/{corpus}/roundtrip"]["per_source"]
            margins = [
                v["margin"] - abs(rounding[s]["margin"]) for s, v in group["per_source"].items()
            ]
            result = assessment["groups"][label + "/" + corpus]
            result["unseen_shift_passes"] = (
                sum(margins) > 0 and sum(v > 0 for v in margins) > len(margins) / 2
            )
            result["passes"] &= result["unseen_shift_passes"]
    arm_passes = {
        arm: all(v["passes"] for k, v in assessment["groups"].items() if "__" + arm + "/" in k)
        for arm in ARMS
    }
    # Fit-only scale removes simple feature-amplitude differences from diagnostics.
    normalized = {}
    for key, group in groups.items():
        label, version, _role, corpus, _control = key.split("/")
        denominator = groups[f"{label}/{version}/optimization/{corpus}/prediction"]["source_mean"][
            "feature_loss"
        ]
        normalized[key] = (
            group["source_mean"]["feature_loss"] / denominator if denominator > 0 else None
        )
    assessment["prerequisite_passes"] = any(arm_passes.values())
    assessment["arm_passes_both_generators_and_corpora"] = arm_passes
    write_json_report(
        {
            "records": records,
            "groups": groups,
            "condition_checks": condition_checks,
            "fit_normalized_feature_distance": normalized,
            "assessment": assessment,
            "generator_updates": 0,
            "qualified_audio": False,
        },
        output / "summary.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.resume:
        config = read(output / "registration.json")
        verify(config["hashes"])
    else:
        if output.exists():
            raise ValueError("fresh output required")
        config = prepare(args.parent.resolve(), output)
    if args.prepare_only:
        progress(output, stage="registered", critic_updates=0, generator_updates=0)
        return
    entries = train(config, output)
    review(config, entries, output)
    verify(config["hashes"])
    for entry in entries.values():
        verify({entry[v]["path"]: entry[v]["sha256"] for v in ("initial", "trained")})
    write_json_report(
        {
            "files": {
                str(output / name): sha256_file(output / name)
                for name in ("registration.json", "critics.json", "summary.json")
            }
        },
        output / "complete.json",
    )
    progress(
        output,
        stage="complete",
        critic_updates=len(entries) * STEPS,
        generator_updates=0,
        qualified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json_report(
                {"error": repr(error), "qualified": False}, args.output / "failure.json"
            )
        raise
