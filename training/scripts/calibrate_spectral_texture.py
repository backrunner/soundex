"""P59 broad TRAIN critic learning, false-detail controls and gradient calibration."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.texture_calibration import aggregate, assess, controls, isolated_draws, measure
from evaluation.trainability_audit import assert_reserved_isolation
from models.adapter_trainability import load_trainable
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss, supervised_waves
from models.restoration_head import predict_budget
from models.spectral_texture import SpectralTextureCritic
from scripts.preflight_spectral_texture import read, verify
from scripts.run_causal_study import atomic_save
from scripts.summarize_guarded_transfer import coverage

STEPS = 1024
FIT_DRAWS = 256
REVIEW_DRAWS = 16
SEED = 20261026


def spaced(length, count):
    if length < count or count < 2:
        raise ValueError("insufficient registered draws")
    return [i * (length - 1) // (count - 1) for i in range(count)]


def prepare(study, output):
    parent = read(study / "registration.json")
    # P57 binds the P54 schedule directly; do not select new crops from scores.
    config_path = Path(parent["study"]) / "registration.json"
    if parent["hashes"].get(str(config_path)) != sha256_file(config_path):
        raise ValueError("changed inherited P54 registration")
    config = read(config_path)
    if config["schedule"] != parent["schedule"]:
        raise ValueError("inherited source schedule mismatch")
    endpoint_path = study / "endpoints.json"
    if read(study / "complete.json")["files"][str(endpoint_path)] != sha256_file(endpoint_path):
        raise ValueError("changed generator endpoints")
    endpoints = {k: v for k, v in read(endpoint_path).items() if k.endswith("-full")}
    if len(endpoints) != 2:
        raise ValueError("both frozen P57 full generators required")
    rows, hashes = load_sources(list(map(Path, config["manifests"])))
    schedule = read(Path(config["schedule"]))
    root = Path(__file__).resolve().parents[1]
    catalogs = sorted((root.parent / "data/runs").glob("*/evaluation-catalog.jsonl"))
    if not catalogs:
        raise ValueError("reserved sources required")
    reserved = [json.loads(line) for p in catalogs for line in p.read_text().splitlines() if line]
    catalog = root.parent / "data/catalogs/current-approved.jsonl"
    universe = list(rows.values()) + [json.loads(s) for s in catalog.read_text().splitlines() if s]
    calibration_rows = [rows[s["row_id"]] for s in schedule["calibration"]["specs"]]
    eligible = isolated_draws(schedule["optimization"], rows, universe, reserved + calibration_rows)
    roles = {}
    for role, count in (
        ("optimization", FIT_DRAWS),
        ("calibration", len(schedule["calibration"]["draws"])),
    ):
        source = schedule[role]
        indices = (
            [eligible[i] for i in spaced(len(eligible), count)]
            if role == "optimization"
            else spaced(len(source["draws"]), count)
        )
        specs, draws = [], []
        for i in indices:
            draw = []
            for j in source["draws"][i]:
                draw.append(len(specs))
                specs.append(source["specs"][j])
            draws.append(draw)
        samples = [{"row": rows[s["row_id"]], "spec": s} for s in specs]
        roles[role] = {
            "indices": indices,
            "specs": specs,
            "draws": draws,
            "samples": samples,
            "coverage": coverage([{"samples": {role: samples}}], role),
        }
    selected = [s["row"] for r in roles.values() for s in r["samples"]]
    assert_reserved_isolation(selected, universe, reserved)
    assert_reserved_isolation(
        [s["row"] for s in roles["optimization"]["samples"]],
        universe,
        [s["row"] for s in roles["calibration"]["samples"]],
    )
    hashes.update({e["path"]: e["sha256"] for e in endpoints.values()})
    for row in selected:
        for kind in ("clean", "degraded"):
            hashes[str(Path(row["_root"]) / row[kind + "_path"])] = row[kind + "_checksum"]
    for p in (
        config_path,
        Path(config["schedule"]),
        endpoint_path,
        study / "complete.json",
        Path(config["detector"]),
        catalog,
        *catalogs,
        root / "requirements.txt",
    ):
        hashes[str(p)] = sha256_file(p)
    verify(hashes)
    output.mkdir(parents=True)
    for p in root.rglob("*"):
        relative = p.relative_to(root)
        if p.suffix not in (".py", ".json") or any(
            x in {".venv", "__pycache__", "outputs", ".pytest_cache", ".ruff_cache"}
            for x in relative.parts
        ):
            continue
        dst = output / "source" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        hashes[str(p)] = sha256_file(p)
        hashes[str(dst)] = sha256_file(dst)
    registration = {
        "format": "soundex-texture-calibration-p59-v1",
        "hashes": hashes,
        "endpoints": endpoints,
        "roles": roles,
        "detector": config["detector"],
        "steps": STEPS,
        "critic_seed": SEED,
        "eligible_optimization_draws": len(eligible),
        "excluded_optimization_draws": len(schedule["optimization"]["draws"]) - len(eligible),
        "lr": 1e-4,
        "fit_review_indices": spaced(FIT_DRAWS, REVIEW_DRAWS),
        "optimizer": "Fresh Adam .9/.999 eps1e-8 clip5. One 4:1 music:speech paired batch per update, cycling 256 fixed draws four times. Both critics start with identical random tensors. No generator updates.",
        "barrier": "Train and hash both 1024-step critic endpoints before any calibration scoring; no early choice, budget reduction or holdout use. Keep initial critics as controls.",
        "gate": "Each seed/corpus on source-equal TRAIN calibration: prediction logistic loss below initialization and balanced constant 2*ln(2). Prediction, reference-band-energy-matched prediction, reference-phase-scrambled HF and quarter-clip-shifted reference HF all need positive mean margin and strict source majority above the original-reference roundtrip numerical control. Report every failure; no audio qualification.",
        "gradient_scale": "Fit-only shared auxiliary coefficient gives 0.1 pooled gradient RMS relative to unchanged common objective; report per-corpus/identity gradients and actual fresh-Adam displacement before considering generator training. This is diagnostic, not a quality gate or proof of equivalent Adam updates.",
        "scope": "Reused TRAIN music/speech. Critic fit/calibration globally recording/work-alias disjoint; both exclude reserved P55 aliases. Fixed generators historically saw other crops of calibration recordings. Finite crop pool, mono/channel views, not full-library training, independent test, converged model, device or listening evidence.",
    }
    write_json_report(registration, output / "registration.json")
    return registration


def sources(config):
    return {s["spec"]["row_id"]: s["row"] for r in config["roles"].values() for s in r["samples"]}


def item_for(config, clips, role, index):
    r = config["roles"][role]
    return LazyPool(r["specs"], clips).select(r["draws"][index])


def metadata(config, role, index):
    r = config["roles"][role]
    return [r["samples"][j] for j in r["draws"][index]]


def payload(model, item):
    batch = item[0].batch
    with torch.no_grad():
        pred, _ = predict_budget(model, batch, item[1])
        wave, reference = supervised_waves(pred, batch)
        observed, _ = supervised_waves(batch["x"], batch)
    paired = ~batch["identity"].bool()
    return tuple(x[paired] for x in (wave, reference, observed, batch["rate"]))


def progress(output, **fields):
    write_json_report(fields, output / "progress.json")
    print(json.dumps(fields), flush=True)


def train(config, output, clips):
    endpoints = {}
    for label, entry in config["endpoints"].items():
        root = output / "training" / label
        root.mkdir(parents=True, exist_ok=True)
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False), "full"
        )
        torch.manual_seed(SEED)
        critic = SpectralTextureCritic()
        initial_path = root / "initial.pth"
        if not initial_path.exists():
            atomic_save({"model": critic.state_dict()}, initial_path)
        initial = torch.load(initial_path, weights_only=False)["model"]
        if any(not torch.equal(v, initial[k]) for k, v in critic.state_dict().items()):
            raise ValueError("changed initial critic control")
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        history, step = [], 0
        latest = root / "latest.pth"
        if latest.exists():
            receipt = read(root / "latest.json")
            if sha256_file(latest) != receipt["sha256"]:
                raise ValueError("changed resumable critic")
            checkpoint = torch.load(latest, weights_only=False)
            if checkpoint["registration_sha256"] != sha256_file(output / "registration.json"):
                raise ValueError("resume registration mismatch")
            critic.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            history, step = checkpoint["history"], checkpoint["step"]
        # Fixed generator outputs, reused from checked per-draw cache receipts.
        data = []
        for index in range(FIT_DRAWS):
            path = root / f"fit-{index:03}.pth"
            receipt = path.with_suffix(".json")
            if receipt.exists():
                recorded = read(receipt)
                if (
                    sha256_file(path) != recorded["sha256"]
                    or recorded["generator_sha256"] != entry["sha256"]
                    or recorded["registration_sha256"] != sha256_file(output / "registration.json")
                ):
                    raise ValueError("changed fixed-generator PCM cache")
                values = torch.load(path, weights_only=False)
            else:
                values = payload(model, item_for(config, clips, "optimization", index))
                atomic_save(values, path)
                write_json_report(
                    {
                        "sha256": sha256_file(path),
                        "generator_sha256": entry["sha256"],
                        "registration_sha256": sha256_file(output / "registration.json"),
                    },
                    receipt,
                )
            data.append(values)
            if (index + 1) % 64 == 0:
                progress(output, stage="fixed_generator_cache", generator=label, draws=index + 1)
        for i in range(step, STEPS):
            optimizer.zero_grad(set_to_none=True)
            value = critic.discriminator_loss(*data[i % FIT_DRAWS]).mean()
            value.backward()
            if not torch.isfinite(value) or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()
            ):
                raise ValueError("nonfinite critic gradient")
            norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in critic.parameters()):
                raise ValueError("nonfinite critic parameters")
            history.append(
                {
                    "step": i + 1,
                    "draw": i % FIT_DRAWS,
                    "loss": float(value.detach()),
                    "gradient_norm": float(norm),
                }
            )
            if (i + 1) % 64 == 0:
                atomic_save(
                    {
                        "model": critic.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": i + 1,
                        "history": history,
                        "registration_sha256": sha256_file(output / "registration.json"),
                    },
                    latest,
                )
                write_json_report({"sha256": sha256_file(latest)}, root / "latest.json")
                progress(output, stage="critic_training", generator=label, step=i + 1, budget=STEPS)
        path = root / "endpoint.pth"
        if not path.exists():
            shutil.copy2(latest, path)
        if sha256_file(path) != sha256_file(latest):
            raise ValueError("frozen critic endpoint differs from final state")
        endpoints[label] = {
            "trained": {"path": str(path), "sha256": sha256_file(path)},
            "initial": {"path": str(initial_path), "sha256": sha256_file(initial_path)},
        }
    write_json_report(endpoints, output / "critics.json")
    return endpoints


def load_critic(entry):
    if sha256_file(entry["path"]) != entry["sha256"]:
        raise ValueError("changed frozen critic")
    critic = SpectralTextureCritic()
    critic.load_state_dict(torch.load(entry["path"], weights_only=False)["model"])
    return critic


def gradients(model, critic, item, meta):
    batch = item[0].batch
    pred, _ = predict_budget(model, batch, item[1])
    wave, reference = supervised_waves(pred, batch)
    observed, _ = supervised_waves(batch["x"], batch)
    common = (
        direct_spectrum_loss(pred, batch)["per_example"].log1p()
        + multires_spectrum_loss(pred, batch)["per_example"].log1p()
    )
    feature = critic.reference_loss(wave, reference, observed, batch["rate"])
    identity = batch["identity"].bool()
    speech = torch.tensor([s["row"]["corpus"] == "speech_library" for s in meta] * 2)
    parameters = list(model.parameters())

    def flat(value):
        gs = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        g = torch.cat(
            [
                (torch.zeros_like(p) if v is None else v).detach().flatten()
                for p, v in zip(parameters, gs, strict=True)
            ]
        )
        if not torch.isfinite(g).all():
            raise ValueError("nonfinite generator diagnostic gradient")
        return g

    def geometry(g, a):
        return {
            "common_norm": float(g.double().norm()),
            "texture_norm": float(a.double().norm()),
            "cosine": float(
                g.double().dot(a.double())
                / (g.double().norm() * a.double().norm()).clamp_min(1e-30)
            ),
        }

    report = {}
    for role, rm in (("paired", ~identity), ("identity", identity)):
        for corpus, cm in (("music", ~speech), ("speech", speech)):
            m = rm & cm
            report[role + "/" + corpus] = geometry(flat(common[m].mean()), flat(feature[m].mean()))
    g, a = flat(common.mean()), flat(feature.mean())
    report["total"] = geometry(g, a)
    return report, g, a


def adam_displacement(parameters, gradient):
    p = torch.nn.Parameter(parameters.clone())
    optimizer = torch.optim.Adam([p], lr=1e-7, betas=(0.9, 0.999), eps=1e-8)
    p.grad = gradient.clone()
    torch.nn.utils.clip_grad_norm_([p], 5, error_if_nonfinite=True)
    optimizer.step()
    return (p.detach() - parameters).double()


def calibrate_gradients(config, critics, output, clips):
    fit, norms = {}, []
    for label, entry in config["endpoints"].items():
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False), "full"
        )
        critic = load_critic(critics[label]["trained"])
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        fit[label] = []
        for index in config["fit_review_indices"]:
            report, g, a = gradients(
                model,
                critic,
                item_for(config, clips, "optimization", index),
                metadata(config, "optimization", index),
            )
            fit[label].append((report, g, a, parameters))
            norms.append((float(g.double().square().sum()), float(a.double().square().sum())))
    denominator = sum(x[1] for x in norms)
    if not denominator > 0:
        raise ValueError("no texture gradient on fit data")
    weight = 0.1 * math.sqrt(sum(x[0] for x in norms) / denominator)
    report = {"weight": weight, "target_gradient_rms_ratio": 0.1, "fit": {}}
    for label, rows in fit.items():
        report["fit"][label] = []
        for values, g, a, p in rows:
            base, mixed = adam_displacement(p, g), adam_displacement(p, g + weight * a)
            values["fresh_adam"] = {
                "control_displacement_norm": float(base.norm()),
                "mixed_displacement_norm": float(mixed.norm()),
                "difference_norm": float((mixed - base).norm()),
                "common_directional_derivative": float(g.double().dot(mixed)),
                "texture_directional_derivative": float(a.double().dot(mixed)),
            }
            report["fit"][label].append(values)
    write_json_report(report, output / "fit-gradient-calibration.json")
    return weight


def review(config, critics, output, clips, weight):
    records, gradient_reports = [], {}
    for label, entry in config["endpoints"].items():
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False), "full"
        )
        critic_models = {k: load_critic(v) for k, v in critics[label].items()}
        gradient_reports[label] = []
        for role in ("optimization", "calibration"):
            indices = (
                config["fit_review_indices"]
                if role == "optimization"
                else range(len(config["roles"][role]["draws"]))
            )
            for index in indices:
                item = item_for(config, clips, role, index)
                values = payload(model, item)
                _, reference, observed, rates = values
                meta = metadata(config, role, index)
                variations = controls(*values, SEED + index)
                for version, critic in critic_models.items():
                    for name, variant in variations.items():
                        scores = measure(critic, variant, reference, observed, rates)
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
                                raise ValueError("nonfinite review score")
                            records.append(record)
                if role == "calibration":
                    r, _, _ = gradients(model, critic_models["trained"], item, meta)
                    gradient_reports[label].append(r)
            progress(
                output, stage="critic_review", generator=label, role=role, records=len(records)
            )
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        if any(not torch.equal(v, checkpoint["model"][k]) for k, v in model.state_dict().items()):
            raise ValueError("frozen generator changed")
    groups = aggregate(records)
    write_json_report(
        {
            "records": records,
            "groups": groups,
            "assessment": assess(groups, config["endpoints"]),
            "calibration_gradients": gradient_reports,
            "fit_only_weight": weight,
            "generator_updates": 0,
            "qualified": False,
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
        config = prepare(args.study.resolve(), output)
    clips = BoundedEnvelopeClips(
        sources(config), Path(config["detector"]), output / "cache", capacity=40
    )
    critics = train(config, output, clips)
    weight = calibrate_gradients(config, critics, output, clips)
    review(config, critics, output, clips, weight)
    verify(config["hashes"])
    for entries in critics.values():
        verify({e["path"]: e["sha256"] for e in entries.values()})
    write_json_report(
        {
            "files": {
                str(p): sha256_file(p)
                for p in (
                    output / "registration.json",
                    output / "critics.json",
                    output / "fit-gradient-calibration.json",
                    output / "summary.json",
                )
            }
        },
        output / "complete.json",
    )
    progress(
        output, stage="complete", critic_updates=2 * STEPS, generator_updates=0, qualified=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json_report(
                {"error": repr(error), "qualified": False}, args.output / "failure.json"
            )
        raise
