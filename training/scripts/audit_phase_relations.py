"""P80 fixed phase-relation diagnostic with matched persistent-Adam controls.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import math
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export
from evaluation.phase_routed_update import auxiliary as complex_auxiliary
from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import canonical_model_state, continued, stepper
from evaluation.realized_complex import realized_hf_terms
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings
from models.fused_spectrum import BURN, balanced_risk, fused_stages, phase_objective, prepare_fusion
from models.multires_spectrum import spectral_loss, supervised_waves
from models.phase_relations import VERSION, auxiliary, relation_inputs
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_gradients import cosine, vector
from scripts.audit_phase_routing import add_records, exact, native, samples, scores
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import fresh, load

ARMS = ("old", "realized_phase", "circular_relation", "wrapped_relation")
NEW = ARMS[2:]


def prepare(prior, output, controls):
    if output.exists():
        raise ValueError("fresh registration required")
    checked_receipt(prior / "complete.json")
    p77 = read(prior / "registration.json")
    if not (controls / "failure.json").exists():
        raise ValueError("retain initial weak-fixture failure")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(p77["hashes"])
    hashes.update(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    for directory in (output / "source", controls):
        hashes.update(
            {
                str(p): sha256_file(p)
                for p in directory.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
        )
    config = {
        k: p77[k]
        for k in (
            "endpoints",
            "fit",
            "transfer",
            "native_draws",
            "coverage",
            "evaluation_catalogs",
            "p74",
        )
    }
    config.update(
        format="soundex-phase-relations-p80-v1",
        version=VERSION,
        prior=str(prior),
        hashes=hashes,
        arms=list(ARMS),
        steps=2,
        gain=64,
        question="Do explicit original-reference phase relations provide a useful direction beyond marginal complex distance? P79 reaches output through the readout but lacks joint spectral gains. Legacy helper exists, but is absent from active P74 objective; this is not a new invention or capacity experiment.",
        definition="Direct FFT256 target instantaneous/frequency/time phase errors, >=8kHz, interior bins, input-only binary native availability, original-reference relative amplitude and existing -60dB relative/-120dB feature-magnitude floors. Pair edge weight is minimum endpoint weight. Each of three weighted per-example means has equal1/3 weight. Circular=(1-cos)/2, wrapped=abs(d-2pi*round(d/2pi))/pi. Fixed0.1*log1p auxiliary, balanced paired/identity, gradients added only to phase_head. Floors are feature units, not PCM dBFS. Empty support has zero phase risk; unchanged magnitude terms still penalize generation. Wrapped cusps explicitly retained, no claim of unique antipodal derivative or equal gradient scale.",
        budget="32 FP64 frozen endpoint/fit-draw derivative probes, then4 origins x4 arms x2 actual updates. Sixteen new relation updates and16 exact-control replay updates. Same P77 gain64 physical readout LR, no new scale selection; persistent moments retained. Freeze16 endpoints before scoring. Only8 new relation graphs render12 raw/proxy draws and5 native draws=400 new native roles. Reuse old/realized/parent/identityDSP/untreated only from hash-bound P77 records after exact model/coordinate/Adam controls.",
        assessment="Unchanged FP64 P76 score panel and recording-paired uncertainty; fit/transfer, music/speech, paired/identity separate. Transfer sources are historical rechecks, not unseen tests. Keep old, realized-complex, circular and wrapped comparisons. Advancing any relation recipe requires every origin native paired-transfer MR and HFSC decrease beyond8*FP32epsilon*abs(old_mean), plus realized-complex decrease, no undefined sources; also retain shape/energy/dynamics/preservation conflicts. This is the unchanged P77 numerical gate, not audibility or sufficient training approval.",
        controls="Correct reference/stationarity, weak/silent support, preserved/warmup/channel exclusion, legacy-kernel correspondence, wrap periodicity, explicit antipodal nondifferentiability, finite differences, exact auxiliary routing, bitwise repeated old/realized model and coordinate Adam endpoints, native max2^-14/RMS2e-5 parity. Initial failed weak-fixture test is retained separately. No timing retry or changes to inference.",
        scope="Two-update objective diagnostic, not adequate training, convergence, architecture ceiling, independent generalization, listening, device qualification or default promotion. Same40 fit recordings/18.724499s supervised and20 exposed transfer recordings; no new library exposure. Frozen P77/P79 decisions unchanged.",
    )
    registered(config, output / "registration.json")


def probe(model, item):
    original, context = item
    batch = {
        k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in original.batch.items()
    }
    prediction, _ = predict_budget(model, batch, context.double())
    final = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
    region = slice(BURN * 128, final.shape[-1] - 128)
    final, reference = final[:, region], batch["clean_wave"][:, region]
    hf = realized_hf_terms(final, reference, batch["rate"])
    scalars = {
        "old": phase_objective(prediction, batch, include_fused=True),
        "fused_mr": balanced_risk(
            spectral_loss(final, reference, batch["rate"])["per_example"].log1p(), batch["identity"]
        ),
        "fused_hf_magnitude": balanced_risk(hf["magnitude"].log1p(), batch["identity"]),
        "realized_phase": complex_auxiliary(prediction, batch, "realized_phase"),
        **{arm: auxiliary(prediction, batch, arm) for arm in NEW},
    }
    head = [p for k, p in model.named_parameters() if k.startswith("phase_head.")]
    gradients = {
        k: vector(torch.autograd.grad(v, head, retain_graph=True)) for k, v in scalars.items()
    }
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("nonfinite registered phase derivative")
    proposals = {k: gradients["old"] + gradients[k] for k in ARMS[1:]}
    geometry = {}
    for name, (delta, weight) in relation_inputs(prediction, batch).items():
        wrapped = (delta - 2 * math.pi * torch.round(delta / (2 * math.pi))).abs()
        mass = weight.sum((1, 2))
        geometry[name] = {
            "support_mass": mass.tolist(),
            "near_antipodal_mass_fraction": (
                (weight * (wrapped >= 0.99 * math.pi)).sum((1, 2)) / mass.clamp_min(1e-12)
            ).tolist(),
            "circular_slope_mean": (
                (weight * (0.5 * delta.sin()).abs()).sum((1, 2)) / mass.clamp_min(1e-12)
            ).tolist(),
        }
    return {
        "scalars": {k: float(v.detach()) for k, v in scalars.items()},
        "norms": {k: float(v.norm()) for k, v in gradients.items()},
        "cosines": {
            a + "/" + b: cosine(gradients[a], gradients[b])
            for a, b in itertools.combinations(gradients, 2)
        },
        "proposed_gradient_cosines": {
            arm: {
                target: cosine(g, gradients[target])
                for target in ("old", "fused_mr", "fused_hf_magnitude", "realized_phase")
            }
            for arm, g in proposals.items()
        },
        "geometry": geometry,
        "roles": batch["identity"].tolist(),
        "scope": "Canonical FP64 phase-head Euclidean directions only, not physical Adam or output gains; zero-support cosines may be undefined.",
    }


def joint_gate(values):
    for metric in ("mr", "hf_sc", "realized_complex"):
        value = values.get(metric, {})
        delta = value.get("mean_delta")
        ruler = value.get("p77_numerical_scale") if metric != "realized_complex" else 0.0
        if (
            delta is None
            or ruler is None
            or not math.isfinite(delta)
            or not math.isfinite(ruler)
            or value.get("undefined_sources", ["missing"])
            or value.get("sources", 0) < 1
            or delta >= -ruler
        ):
            return False
    return True


def assess(records, config):
    comparisons = {}
    for population, role, corpus, route, origin in itertools.product(
        ("fit", "transfer"),
        ("paired", "identity"),
        ("all", "music_library", "speech_library"),
        ("raw", "proxy", "native"),
        config["endpoints"],
    ):
        selected = [
            r
            for r in records
            if r["population"] == population
            and r["role"] == role
            and (corpus == "all" or r["corpus"] == corpus)
        ]
        pairs = (
            [(arm, "old") for arm in ARMS[1:]]
            + [(arm, "realized_phase") for arm in NEW]
            + [("wrapped_relation", "circular_relation")]
        )
        for candidate, control in pairs:
            key = f"{population}/{role}/{corpus}/{route}/{origin}/{candidate}-vs-{control}"
            values = paired_recordings(
                selected, f"{route}/{origin}/{candidate}", f"{route}/{origin}/{control}", seed=80
            )
            for value in values.values():
                value["p77_numerical_scale"] = (
                    8 * torch.finfo(torch.float32).eps * abs(value["control_mean"])
                    if "control_mean" in value
                    else None
                )
            comparisons[key] = values
        if route == "native":
            for candidate, control in itertools.product(ARMS, ("untreated", "native/identity_dsp")):
                comparisons[
                    f"{population}/{role}/{corpus}/{route}/{origin}/{candidate}-vs-{control}"
                ] = paired_recordings(selected, f"{route}/{origin}/{candidate}", control, seed=80)
    gates = {}
    for origin, arm in itertools.product(config["endpoints"], NEW):
        values = comparisons[f"transfer/paired/all/native/{origin}/{arm}-vs-old"]
        gates[origin + "/" + arm] = joint_gate(values)
    return {
        "comparisons": comparisons,
        "joint_transfer_gate": gates,
        "qualified": False,
        "scope": config["scope"],
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "probes").exists() or (output / "proposals").exists():
        raise ValueError("existing experiment retained")
    prior, p74 = Path(config["prior"]), Path(config["p74"])
    old = read(p74 / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        config["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), p74 / "cache", capacity=100),
    )
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"] + config["transfer"]}
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = load(saved).double().eval()
        for draw in config["fit"]:
            registered(
                {
                    "draw": draw,
                    "samples": samples(schedule, rows, draw),
                    **probe(model, items[draw]),
                },
                output / "probes" / origin / f"{draw}.json",
            )
            print("probe", origin, draw, flush=True)
        exact(model.state_dict(), {k: v.double() for k, v in saved["model"].items()})
    previous = read(prior / "frozen.json")["endpoints"]
    endpoints, equivalence = {}, {}
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for arm in ARMS:
            model, optimizer = continued(saved, config["gain"])
            history = []
            for step in range(config["steps"]):
                selected = [items[i] for i in config["fit"][4 * step : 4 * step + 4]]
                history.append(
                    update(
                        model,
                        selected,
                        optimizer,
                        arm,
                        adam_step=stepper(model),
                        auxiliary_fn=auxiliary if arm in NEW else None,
                    )
                )
            name = origin + "/" + arm
            if arm not in NEW:
                original = torch.load(
                    previous[origin + "/gain64_" + arm]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                exact(canonical_model_state(model), original["model"])
                exact(model.state_dict(), original["coordinate_model"])
                exact(optimizer.state_dict(), original["coordinate_optimizer"])
                equivalence[name] = True
            path = output / "proposals" / name / "endpoint.pth"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    "format": "soundex-phase-relations-p80-v1",
                    "kind": saved["kind"],
                    "seed": saved["seed"],
                    "model": canonical_model_state(model),
                    "coordinate_model": model.state_dict(),
                    "coordinate_optimizer": optimizer.state_dict(),
                    "updates": history,
                    "parent": entry,
                    "steps": 2,
                    "gain": config["gain"],
                    "arm": arm,
                },
                path,
            )
            endpoints[name] = {"path": str(path), "sha256": sha256_file(path)}
            print("frozen", name, flush=True)
    registered({"endpoints": endpoints, "equivalence": equivalence}, output / "frozen.json")
    records = []
    for r in read(prior / "records.json"):
        path = r["path"]
        for arm in ARMS[:2]:
            if path.endswith("/gain64_" + arm):
                records.append({**r, "path": path.rsplit("/", 1)[0] + "/" + arm})
        if path in ("untreated", "native/identity_dsp") or path.endswith("/parent"):
            records.append(r)
    graphs, parity = {}, {}
    with torch.no_grad():
        # Freeze all graph hashes before rendering/scoring a changed endpoint.
        for name, entry in endpoints.items():
            if name.rsplit("/", 1)[1] not in NEW:
                continue
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = fresh(saved["kind"], saved["seed"]).eval()
            model.load_state_dict(saved["model"], strict=True)
            export(model, output / "exports" / name)
            graphs[name] = output / "exports" / name / "diagnostic.onnx"
        registered({str(p): sha256_file(p) for p in graphs.values()}, output / "native-graphs.json")
        for name, graph in graphs.items():
            saved = torch.load(endpoints[name]["path"], map_location="cpu", weights_only=False)
            model = fresh(saved["kind"], saved["seed"]).eval()
            model.load_state_dict(saved["model"], strict=True)
            for draw, (objective, context) in items.items():
                batch = objective.batch
                prediction, _ = predict_budget(model, batch, context)
                raw, reference = supervised_waves(prediction, batch)
                final = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
                region = slice(BURN * 128, final.shape[-1] - 128)
                for route, wave in (("raw", raw), ("proxy", final[:, region])):
                    add_records(
                        records,
                        config,
                        schedule,
                        rows,
                        draw,
                        route + "/" + name,
                        scores(wave, reference, batch["rate"]),
                    )
            print("raw/proxy", name, flush=True)
            for draw in config["native_draws"]:
                batch = items[draw][0].batch
                wave, check = native(
                    Path(old["binary"]), graph, batch, output / "native" / name / str(draw), True
                )
                region = slice(BURN * 128, wave.shape[-1] - 128)
                add_records(
                    records,
                    config,
                    schedule,
                    rows,
                    draw,
                    "native/" + name,
                    scores(wave[:, region], batch["clean_wave"][:, region], batch["rate"]),
                )
                parity[name + "/" + str(draw)] = check
            write_json_report(records, output / "records.json")
            write_json_report(parity, output / "correspondence.json")
            print("native", name, flush=True)
    result = assess(records, config)
    result["native_correspondence_pass"] = all(p["passed"] for p in parity.values())
    registered(result, output / "assessment.json")
    if not result["native_correspondence_pass"]:
        raise ValueError("native correspondence failure retained")
    verify(config["hashes"])
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
    verify(read(output / "native-graphs.json"))
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
    print("complete", result["joint_transfer_gate"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh_directory = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.output.resolve(), args.controls.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        failure = args.output / "failure.json"
        if (
            args.output.exists()
            and not failure.exists()
            and (args.mode == "run" or fresh_directory)
        ):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
