"""Real-TRAIN integration preflight; trains critics, never restoration weights."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from evaluation.reporting import sha256_file, write_json_report
from evaluation.trainability_audit import assert_reserved_isolation
from models.adapter_trainability import load_trainable
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from models.spectral_texture import SpectralTextureCritic


def read(path):
    return json.loads(path.read_text())


def verify(hashes):
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError("changed input: " + path)


def run(study, audit, output):
    torch.set_num_threads(1)
    study, audit, output = study.resolve(), audit.resolve(), output.resolve()
    if output.exists():
        raise ValueError("fresh output required")
    config = read(audit / "registration.json")
    if read(audit / "complete.json")["files"][str(audit / "registration.json")] != sha256_file(
        audit / "registration.json"
    ):
        raise ValueError("changed P56 registration")
    if read(study / "complete.json")["files"][str(study / "endpoints.json")] != sha256_file(
        study / "endpoints.json"
    ):
        raise ValueError("changed P57 endpoints")
    endpoints = {k: v for k, v in read(study / "endpoints.json").items() if k.endswith("-full")}
    if len(endpoints) != 2:
        raise ValueError("both frozen generators required")
    roles = config["roles"]
    rows = {s["spec"]["row_id"]: s["row"] for r in roles.values() for s in r["samples"]}
    root = Path(__file__).resolve().parents[1]
    catalogs = sorted((root.parent / "data/runs").glob("*/evaluation-catalog.jsonl"))
    if not catalogs:
        raise ValueError("reserved source catalogs required")
    reserved = [json.loads(s) for p in catalogs for s in p.read_text().splitlines() if s]
    catalog = root.parent / "data/catalogs/current-approved.jsonl"
    universe = list(rows.values()) + [json.loads(s) for s in catalog.read_text().splitlines() if s]
    assert_reserved_isolation(list(rows.values()), universe, reserved)
    assert_reserved_isolation(
        [s["row"] for s in roles["optimization"]["samples"]],
        universe,
        [s["row"] for s in roles["calibration"]["samples"]],
    )
    hashes = {e["path"]: e["sha256"] for e in endpoints.values()}
    for row in rows.values():
        for kind in ("clean", "degraded"):
            hashes[str(Path(row["_root"]) / row[kind + "_path"])] = row[kind + "_checksum"]
    for p in [
        study / "endpoints.json",
        audit / "registration.json",
        *catalogs,
        catalog,
        Path(config["detector"]),
        root / "requirements.txt",
    ]:
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
    write_json_report(
        {
            "format": "soundex-texture-integration-v1",
            "hashes": hashes,
            "endpoints": endpoints,
            "roles": roles,
            "critic_steps": 8,
            "critic_lr": 1e-4,
            "seed": 20261026,
            "scope": "Integration only on reused P56 TRAIN crops. Fixed generators. Eight critic Adam steps averaged over all four paired fit batches. Freeze both critics before any recording-disjoint TRAIN calibration. No generator update, independent test, convergence, perceptual or runtime claim.",
        },
        output / "registration.json",
    )
    clips = BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=80)
    items = {
        r: [LazyPool(v["specs"], clips).select(d) for d in v["draws"]] for r, v in roles.items()
    }
    frozen = {}
    for label, entry in endpoints.items():
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False), "full"
        )
        data = []
        with torch.no_grad():
            for item in items["optimization"]:
                batch = item[0].batch
                pred, _ = predict_budget(model, batch, item[1])
                wave, reference = supervised_waves(pred, batch)
                observed, _ = supervised_waves(batch["x"], batch)
                paired = ~batch["identity"].bool()
                data.append(
                    (wave[paired], reference[paired], observed[paired], batch["rate"][paired])
                )
        torch.manual_seed(20261026)
        critic = SpectralTextureCritic()
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        history = []
        for step in range(8):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for values in data:
                loss = critic.discriminator_loss(*values).mean() / len(data)
                loss.backward()
                total += float(loss.detach())
            if not torch.isfinite(torch.tensor(total)) or any(
                not torch.isfinite(p.grad).all() for p in critic.parameters()
            ):
                raise ValueError("nonfinite critic update")
            norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            history.append({"step": step + 1, "loss": total, "gradient_norm": float(norm)})
        path = output / (label + ".pth")
        torch.save(
            {"model": critic.state_dict(), "optimizer": optimizer.state_dict(), "history": history},
            path,
        )
        frozen[label] = {"path": str(path), "sha256": sha256_file(path)}
    write_json_report(frozen, output / "critics.json")
    results = {}
    for label, entry in endpoints.items():
        critic = SpectralTextureCritic()
        critic.load_state_dict(torch.load(frozen[label]["path"], weights_only=False)["model"])
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False), "full"
        )
        results[label] = {}
        for role, batches in items.items():
            records = []
            for item in batches:
                batch = item[0].batch
                prediction, _ = predict_budget(model, batch, item[1])
                wave, reference = supervised_waves(prediction, batch)
                observed, _ = supervised_waves(batch["x"], batch)
                value = critic.reference_loss(wave, reference, observed, batch["rate"])
                grads = torch.autograd.grad(
                    value.mean(), list(model.parameters()), allow_unused=True
                )
                if not torch.isfinite(value).all() or any(
                    g is not None and not torch.isfinite(g).all() for g in grads
                ):
                    raise ValueError("nonfinite generator gradient")
                norm = sum(g.double().square().sum() for g in grads if g is not None).sqrt()
                if not norm > 0:
                    raise ValueError("missing generator gradient")
                identity = batch["identity"].bool()
                records.append(
                    {
                        "paired_feature_loss": float(value[~identity].mean().detach()),
                        "identity_feature_loss": float(value[identity].mean().detach()),
                        "generator_gradient_norm": float(norm),
                    }
                )
            results[label][role] = records
        original = torch.load(entry["path"], map_location="cpu", weights_only=False)["model"]
        if any(not torch.equal(v, original[k]) for k, v in model.state_dict().items()):
            raise ValueError("restoration generator changed")
    verify(hashes)
    verify({e["path"]: e["sha256"] for e in frozen.values()})
    write_json_report(
        {
            "results": results,
            "generator_updates": 0,
            "qualified": False,
            "critic_parameters": sum(p.numel() for p in critic.parameters()),
        },
        output / "summary.json",
    )
    write_json_report(
        {
            "files": {
                str(p): sha256_file(p)
                for p in (
                    output / "registration.json",
                    output / "critics.json",
                    output / "summary.json",
                )
            }
        },
        output / "complete.json",
    )
    print(json.dumps({"complete": str(output), "critic_updates": 16, "generator_updates": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.audit, args.output)
