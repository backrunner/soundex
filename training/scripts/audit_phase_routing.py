"""P76 matched two-update Adam and native-output transfer diagnostic.

No full training, coefficient search, held-out selection or timing reruns.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import subprocess
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import environment, export
from evaluation.fidelity_calibration import assess_wave
from evaluation.phase_routed_update import ARMS, update
from evaluation.realized_complex import realized_hf_terms
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings, panel
from evaluation.rust_bridge import read_sxa, write_sxa
from evaluation.trainability_audit import assert_reserved_isolation
from models.fused_spectrum import BURN, fused_stages, prepare_fusion
from models.losses import build_missing_band_mask
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_context_deployment import unpack_trace
from scripts.audit_fused_spectrum import snapshot
from scripts.run_anchored_pilot import verify
from scripts.run_band_capacity_training import update as canonical_update
from scripts.run_band_training import assert_catalogs_unchanged, isolated_rows, jsonl
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import load
from scripts.summarize_guarded_transfer import coverage


def exact(a, b):
    """Optimizer states must be independent copies and bitwise reproducible."""
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise ValueError("canonical control/state mismatch")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("state keys mismatch")
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("state length mismatch")
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    elif a != b:
        raise ValueError("state value mismatch")


def continued(saved):
    model = load(saved).eval()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    exact(optimizer.state_dict(), saved["optimizer"])
    return model, optimizer


def samples(schedule, rows, draw):
    return [
        {"spec": schedule["specs"][j], "row": rows[schedule["specs"][j]["row_id"]]}
        for j in schedule["draws"][draw]
    ]


def prepare(prior, audit, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve every attempt")
    checked_receipt(prior / "p74-complete.json")
    checked_receipt(audit / "complete.json")
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    fit = list(range(8))
    fit_rows = [s["row"] for i in fit for s in samples(schedule, rows, i)]
    repo = Path(old["repository"])
    catalogs = sorted((repo / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repo / "data/catalogs/current-approved.jsonl"
    universe = list(rows.values()) + jsonl(catalog)
    review = read(Path(old["review_schedule"]))
    reserved = [r for p in catalogs for r in jsonl(p)]
    reserved += [rows[review["specs"][j]["row_id"]] for d in review["review"] for j in d]
    allowed = isolated_rows(
        rows, {r["track_id"] for r in rows.values()}, universe, fit_rows + reserved
    )
    transfer = []
    for i in range(8, len(schedule["draws"])):
        if all(schedule["specs"][j]["row_id"] in allowed for j in schedule["draws"][i]):
            transfer.append(i)
            if len(transfer) == 4:
                break
    if len(transfer) != 4:
        raise ValueError("four source/work-disjoint TRAIN draws required")
    selected = [s["row"] for i in fit + transfer for s in samples(schedule, rows, i)]
    assert_reserved_isolation(selected, universe, reserved)
    output.mkdir(parents=True)
    snapshot(output)
    endpoints = read(prior / "endpoints.json")
    hashes = dict(old["hashes"])
    hashes.update({e["path"]: e["sha256"] for e in endpoints.values()})
    for p in (
        prior / "registration.json",
        prior / "p74-complete.json",
        audit / "complete.json",
        audit / "assessment.json",
        catalog,
        *catalogs,
        *(p for p in (output / "source").rglob("*") if p.is_file()),
    ):
        hashes[str(p)] = sha256_file(p)
    registered(
        {
            "format": "soundex-phase-routing-p76-v1",
            "prior": str(prior),
            "audit": str(audit),
            "endpoints": endpoints,
            "fit": fit,
            "transfer": transfer,
            "native_draws": [fit[0], *transfer],
            "arms": list(ARMS),
            "steps": 2,
            "evaluation_catalogs": list(map(str, catalogs)),
            "hashes": hashes,
            "coverage": {
                role: coverage(
                    [{"samples": {role: [s for i in indices for s in samples(schedule, rows, i)]}}],
                    role,
                )
                for role, indices in (("fit", fit), ("transfer", transfer))
            },
            "question": "Does routing realized complex auxiliary only to phase_head preserve spectral improvement in actual copied-moment Adam and Rust outputs? P75 Euclidean gradients cannot answer this; P69 froze the base.",
            "controls": "All four P74 endpoints, unchanged old objective, realized_phase: +0.1 log1p(original-reference realized HF complex), direct_phase: exact log-risk difference from direct complex .1 to .2. Both auxiliaries add head derivatives only. No coefficient or seed selection; different scales/support prevent a pure loss-location claim.",
            "optimizer": "Exact copied persistent P74 FP32 Adam state, final stored learning rate and global clip5. Two cumulative updates, four draws/update, all endpoints/arms; canonical old update must reproduce model AND moments bitwise. Global clipping can change base displacement even when its preclip derivative is unchanged. No rescaling, line search, step veto or resets.",
            "budget": "24 diagnostic proposal updates plus8 duplicate canonical-control verification updates. Freeze all12 proposals before any transfer/output scoring. This tests finite update transfer, not convergence, a full training budget or a capacity ceiling.",
            "selection": "No automatic promotion. A positive diagnostic requires each endpoint's realized_phase native paired TRAIN-transfer recording mean to improve MR, independent HF SC and realized HF complex against its matched old update; retain shape, envelope, dynamics, calibrated powers, low-band and identity counterexamples. Missing/undefined scores are not passing. Never accept coherent improvement alone from attenuation. Failure blocks this unchanged route, not every phase architecture.",
            "native": "Frozen P74 Rust direct-stream binary,16 model paths plus identity DSP, first fit draw and all4 transfer draws, paired/identity mono views; untreated aligned dry reference control. Trace the first role of first fit draw for all paths to check the new term on actual native DSP. Existing P67 PCM max2^-14/RMS2e-5 correspondence bounds. No timing or callback/device qualification.",
            "scope": "Reused TRAIN sources with fit/transfer source and connected-work isolation only within this diagnostic; all were eligible for prior P74 training. No unseen generalization or physical stereo/listening evidence. Original lossless targets and default weights unchanged.",
        },
        output / "registration.json",
    )
    print("registered", fit, transfer, flush=True)


@torch.no_grad()
def scores(wave, reference, rates):
    # Float64 scoring lifts the actual FP32 wave; it does not change the output.
    wave, reference = wave.double(), reference.double()
    result = assess_wave(wave, reference, rates)
    hf = realized_hf_terms(wave, reference, rates)
    result.update(
        realized_complex=hf["complex"].tolist(), realized_magnitude=hf["magnitude"].tolist()
    )
    for key in (
        "reference_power",
        "prediction_power",
        "magnitude_error_power",
        "complex_error_power",
        "phase_interaction_power",
    ):
        # Sum three disjoint bands and average three resolutions: PCM HF power.
        result["hf_" + key] = (sum(b[key] for b in hf["bands"]) / 3).tolist()
    panels = [
        panel(a[:, None].numpy(), b[:, None].numpy(), int(rate))
        for a, b, rate in zip(wave, reference, rates.reshape(-1), strict=True)
    ]
    return [
        {**{k: v[i] for k, v in result.items()}, **panels[i]["metrics"]} for i in range(len(wave))
    ]


def add_records(records, config, schedule, rows, draw, path, output_scores):
    entries = samples(schedule, rows, draw)
    if len(output_scores) != 2 * len(entries):
        raise ValueError("paired/identity geometry changed")
    for i, values in enumerate(output_scores):
        sample = entries[i % len(entries)]
        records.append(
            {
                "path": path,
                "source": sample["row"]["source_checksum"],
                "case": f"{draw}/{i}",
                "draw": draw,
                "sample": sample,
                "population": "fit" if draw in config["fit"] else "transfer",
                "role": "paired" if i < len(entries) else "identity",
                "corpus": sample["row"]["corpus"],
                "scores": values,
            }
        )


def native(binary, graph, batch, directory, trace_first):
    rendered, correspondence = [], None
    for role, pcm in enumerate(batch["pcm"]):
        root = directory / str(role)
        root.mkdir(parents=True)
        rate = int(batch["rate"][role, 0])
        write_sxa(root / "input.sxa", pcm.numpy(), rate)
        trace = trace_first and role == 0
        result = subprocess.run(
            [
                str(binary),
                str(graph),
                str(root / "input.sxa"),
                str(root / "output.sxa"),
                str(root / "runtime.json"),
                "1",
                *(["trace"] if trace else []),
            ],
            env=environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        (root / "stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"native output failed: {root}")
        wave, actual_rate = read_sxa(root / "output.sxa")
        report = read(root / "runtime.json")
        if actual_rate != rate or wave.shape != (len(pcm), 1) or not report["finite"]:
            raise ValueError("native output timeline/finite failure")
        rendered.append(torch.from_numpy(wave[:, 0]))
        if trace:
            observed = unpack_trace(report)
            condition = observed["condition"]
            cutoff = condition[..., 1] * rate / 2
            mask = build_missing_band_mask(
                cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
            ).reshape(1, 1, -1, 129)
            b = {
                "x": observed["features"],
                "mask": mask,
                "active": condition[..., 2].bool(),
                "dry": batch["dry"][role : role + 1],
            }
            proxy = fused_stages(observed["prediction"], b["x"], prepare_fusion(b))["final"]
            difference = proxy[0, 128:].double() - rendered[-1][128:].double()
            maximum, rms = float(difference.abs().max()), float(difference.square().mean().sqrt())
            region = slice(BURN * 128, len(pcm) - 128)
            target, rates = (
                batch["clean_wave"][role : role + 1, region],
                batch["rate"][role : role + 1],
            )
            correspondence = {
                "pcm_max": maximum,
                "pcm_rms": rms,
                "passed": maximum <= 2**-14 and rms <= 2e-5,
                "proxy": scores(proxy[:, region], target, rates),
                "native": scores(rendered[-1][None, region], target, rates),
            }
    return torch.stack(rendered), correspondence


def summarize(records, config):
    comparisons = {}
    paths = sorted({r["path"] for r in records})
    for population in ("fit", "transfer"):
        for role in ("paired", "identity"):
            for corpus in ("all", "music_library", "speech_library"):
                selected = [
                    r
                    for r in records
                    if r["population"] == population
                    and r["role"] == role
                    and (corpus == "all" or r["corpus"] == corpus)
                ]
                for endpoint in config["endpoints"]:
                    for route in ("raw", "proxy", "native"):
                        for arm, control in (
                            ("old", "parent"),
                            ("realized_phase", "old"),
                            ("direct_phase", "old"),
                            ("realized_phase", "direct_phase"),
                        ):
                            a, b = f"{route}/{endpoint}/{arm}", f"{route}/{endpoint}/{control}"
                            if a in paths and b in paths:
                                key = f"{population}/{role}/{corpus}/{a}-vs-{control}"
                                comparisons[key] = paired_recordings(selected, a, b)
                    for arm in ("parent", *config["arms"]):
                        a = f"native/{endpoint}/{arm}"
                        for b in ("untreated", "native/identity_dsp"):
                            key = f"{population}/{role}/{corpus}/{a}-vs-{b}"
                            comparisons[key] = paired_recordings(selected, a, b)
    gates = {}
    for endpoint in config["endpoints"]:
        values = comparisons[f"transfer/paired/all/native/{endpoint}/realized_phase-vs-old"]
        gates[endpoint] = all(
            values[k].get("mean_delta", float("inf")) < 0 and not values[k]["undefined_sources"]
            for k in ("mr", "hf_sc", "realized_complex")
        )
    return {
        "comparisons": comparisons,
        "joint_transfer_gate": gates,
        "all_joint_transfer_gates_pass": all(gates.values()),
        "qualified": False,
        "scope": config["scope"],
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "proposals").exists():
        raise ValueError("existing proposals retained; no silent retry")
    prior = Path(config["prior"])
    old = read(prior / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        config["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), prior / "cache", capacity=100),
    )
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"] + config["transfer"]}
    frozen, updates = {}, {}
    for label, endpoint in config["endpoints"].items():
        saved = torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        if saved["step"] != 16384 or any(
            int(s["step"]) != 16384 for s in saved["optimizer"]["state"].values()
        ):
            raise ValueError("complete P74 endpoint and persistent Adam required")
        frozen[label + "/parent"] = endpoint
        for arm in config["arms"]:
            model, optimizer = continued(saved)
            history = []
            if arm == "old":
                canonical, canonical_optimizer = continued(saved)
            for step in range(2):
                selected = [items[i] for i in config["fit"][4 * step : 4 * step + 4]]
                result = update(model, selected, optimizer, arm)
                if arm == "old":
                    canonical_update(canonical, selected, canonical_optimizer)
                    exact(model.state_dict(), canonical.state_dict())
                    exact(optimizer.state_dict(), canonical_optimizer.state_dict())
                history.append(result)
            root = output / "proposals" / label / arm
            root.mkdir(parents=True)
            path = root / "endpoint.pth"
            torch.save(
                {
                    **saved,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "p76": {"parent": endpoint, "arm": arm, "updates": history},
                    "step": 16386,
                    "history": saved["history"] + history,
                },
                path,
            )
            frozen[label + "/" + arm] = {"path": str(path), "sha256": sha256_file(path)}
            updates[label + "/" + arm] = history
            print("frozen", label, arm, flush=True)
    registered({"endpoints": frozen, "updates": updates}, output / "frozen.json")
    records, graphs, parity = [], {}, {}
    with torch.no_grad():
        for label, entry in frozen.items():
            model = load(torch.load(entry["path"], map_location="cpu", weights_only=False)).eval()
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
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
                        route + "/" + label,
                        scores(wave, reference, batch["rate"]),
                    )
            write_json_report(records, output / "proxy-records.json")
            print("proxy scored", label, flush=True)
        graphs["identity_dsp"] = Path(old["controls"]["identity_dsp"])
        registered({str(p): sha256_file(p) for p in graphs.values()}, output / "native-graphs.json")
        for label, graph in graphs.items():
            for draw in config["native_draws"]:
                batch = items[draw][0].batch
                wave, correspondence = native(
                    Path(old["binary"]),
                    graph,
                    batch,
                    output / "native" / label / str(draw),
                    draw == config["fit"][0],
                )
                region = slice(BURN * 128, wave.shape[-1] - 128)
                add_records(
                    records,
                    config,
                    schedule,
                    rows,
                    draw,
                    "native/" + label,
                    scores(wave[:, region], batch["clean_wave"][:, region], batch["rate"]),
                )
                if correspondence is not None:
                    parity[label] = correspondence
            write_json_report(records, output / "records.json")
            write_json_report(parity, output / "correspondence.json")
            print("native scored", label, flush=True)
        for draw in config["native_draws"]:
            batch = items[draw][0].batch
            region = slice(BURN * 128, batch["dry"].shape[-1] - 128)
            add_records(
                records,
                config,
                schedule,
                rows,
                draw,
                "untreated",
                scores(batch["dry"][:, region], batch["clean_wave"][:, region], batch["rate"]),
            )
    write_json_report(records, output / "records.json")
    result = summarize(records, config)
    result["native_correspondence_pass"] = all(p["passed"] for p in parity.values())
    registered(result, output / "assessment.json")
    if not result["native_correspondence_pass"]:
        raise ValueError("native correspondence failure retained; quality comparison unqualified")
    verify(config["hashes"])
    verify({entry["path"]: entry["sha256"] for entry in frozen.values()})
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
    print("complete; joint transfer gates", result["joint_transfer_gate"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.audit.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
