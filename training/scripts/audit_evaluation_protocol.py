"""P81 analytic metric calibration and read-only P80 evidence reinterpretation.

No optimizer, model forward, native rerender, timing retry or new evaluation source.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_evidence import (
    VERSION,
    describe_comparison,
    extended_panel,
    spectral_dynamics,
    strict_paired_recordings,
)
from evaluation.restoration_panel import _band, paired_recordings
from scripts.audit_fused_spectrum import snapshot
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def calibrate():
    records = []
    for rate in (44100, 48000):
        t = np.arange(32768) / rate
        envelope = np.exp(-(((t - 0.18) / 0.018) ** 2)) + np.exp(-(((t - 0.5) / 0.03) ** 2))
        reference = (0.1 * envelope * np.sin(2 * np.pi * 10000 * t))[:, None]
        noise = np.random.default_rng(81).normal(0, 0.01, reference.shape)
        # Exact whole-excerpt high-pass, a calibration artifact, not deployment DSP.
        spectrum = np.fft.rfft(noise, axis=0)
        spectrum[np.fft.rfftfreq(len(noise), 1 / rate) < 8000] = 0
        noise = np.fft.irfft(spectrum, n=len(noise), axis=0)
        cases = {
            "reference": (reference, reference),
            "half_gain": (reference / 2, reference),
            "polarity": (-reference, reference),
            "wrong_frequency": (
                (0.1 * envelope * np.sin(2 * np.pi * 11000 * t))[:, None],
                reference,
            ),
            "shifted_transient": (np.roll(reference, 2400, axis=0), reference),
            "added_hf_noise": (reference + noise, reference),
            "silence": (reference * 0, reference * 0),
            "silent_noise": (noise, reference * 0),
            "weak_gain": (reference * 0.5e-6, reference * 1e-6),
        }
        for name, (wave, target) in cases.items():
            result = extended_panel(wave, target, rate)
            records.append({"rate": rate, "case": name, **result})
        values = {r["case"]: r["metrics"] for r in records if r["rate"] == rate}
        assert values["reference"]["hf_sc"] == values["reference"]["hf_spectral_dynamics"] == 0
        assert abs(values["half_gain"]["hf_sc"] - 0.5) < 1e-14
        assert values["half_gain"]["hf_shape"] < 1e-20
        assert values["polarity"]["hf_sc"] == 0
        assert abs(values["polarity"]["hf_complex"] - 2) < 1e-14
        assert values["wrong_frequency"]["hf_shape"] > 0.8
        assert values["shifted_transient"]["hf_spectral_dynamics"] > 0
        assert values["added_hf_noise"]["hf_sc"] > 0
        assert values["silent_noise"]["hf_sc"] is None
        assert values["silent_noise"]["hf_magnitude_error_rms"] > 0
        assert abs(values["weak_gain"]["hf_sc"] - 0.5) < 1e-14

    a = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    b = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    movement = {**_band(a, b), **spectral_dynamics(a, b)}
    assert movement["envelope"] == movement["dynamics"] == 0
    assert abs(movement["spectral_dynamics"] - np.sqrt(4 / 3)) < 1e-14
    shape = _band(np.tile([0.5, np.sqrt(0.75)], (1, 3, 1)), b)["shape"]
    assert abs(shape - 0.5) < 1e-14
    reference = (0.125 * (-1.0) ** np.arange(16384))[:, None]
    control = reference.astype(np.float32) / 2
    candidate = np.nextafter(control, reference.astype(np.float32))
    before, after = [
        extended_panel(x, reference, 48000)["metrics"]["hf_sc"] for x in (control, candidate)
    ]
    analytical_delta = -float(np.spacing(np.float32(0.0625))) / 0.125
    assert abs(after - before - analytical_delta) < 1e-14
    ruler = 8 * np.finfo(np.float32).eps * before
    assert 0 < before - after < ruler
    zero = np.zeros((16384, 1))
    impulse = zero.copy()
    impulse[0] = 0.25
    edge = extended_panel(impulse, zero, 48000)["metrics"]
    assert edge["hf_magnitude_error_rms"] == 0 and edge["full_pcm_error_rms"] == 0.25 / 128
    return {
        "records": records,
        "constant_energy_frequency_motion": movement,
        "squared_hellinger_known_answer": shape,
        "hann_endpoint": edge,
        "fp32_one_ulp": {
            "before_hf_sc": before,
            "after_hf_sc": after,
            "observed_delta": after - before,
            "analytical_delta": analytical_delta,
            "historical_8eps_ruler": float(ruler),
            "incremental_pcm_rms": float(
                np.sqrt(np.mean((candidate.astype(float) - control) ** 2))
            ),
            "scope": "Exact represented PCM and an analytic answer; not native renderer repeatability or audibility.",
        },
    }


def schema_controls():
    rows = [
        {"source": "a", "case": 0, "path": path, "scores": score}
        for path, score in (("new", {"mr": 1, "hf_sc": 1}), ("old", {"mr": 2}))
    ]
    old = paired_recordings(rows, "new", "old")
    assert "hf_sc" not in old
    failures = {}
    for name in ("missing_metric", "nan_with_undefined_partner"):
        if name == "nan_with_undefined_partner":
            rows[0]["scores"] = {"mr": float("nan")}
            rows[1]["scores"] = {"mr": None}
        try:
            strict_paired_recordings(rows, "new", "old")
        except ValueError as error:
            failures[name] = str(error)
        else:
            raise AssertionError("invalid data accepted")
    return {"historical_silently_dropped": ["hf_sc"], "strict_rejections": failures}


def run(root, output):
    if output.exists():
        raise ValueError("fresh review directory required; retain prior failures")
    checked_receipt(root / "complete.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = {**read(root / "complete.json")["files"]}
    hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "version": VERSION,
            "prior": str(root),
            "hashes": hashes,
            "hypothesis": "P71 envelope dynamics misses within-band motion; shape is squared Hellinger; schema intersection can drop metrics; P77 scalar FP32-epsilon cutoff is not a numerical error bound. Test known answers, then reinterpret all frozen P80 comparisons without changing old scores, endpoints or gates.",
            "calibration": "Seed81 fixed synthetic PCM at44100/48000: exact reference, gain, polarity, frequency error, shifted transients, HF noise, silence and weak scaling. Exact spectral-vector motion/Hellinger controls, a Hann endpoint impulse and analytic FP32 one-ULP Nyquist amplitude change. These test definitions, not perceptual severity or model quality.",
            "review": "Read-only all saved P80 comparison summaries; independently strict-reaggregate all native versus-old comparisons for fit/transfer, paired/identity and all/music/speech strata with original seed80/2000 recording bootstrap. All must exactly reproduce old numbers. Preserve every source, interval, undefined score and historical gate. Missing new spectral-dynamics scores remain missing, not fabricated from aggregate envelopes.",
            "decision": "Primary MR/HFSC interval direction, supporting shape/energy/dynamics and coherent/preservation directions reported separately. No epsilon-derived threshold, phase sign veto or automatic training/release approval. Intervals describe the reused panel, not numerical precision, audibility, multiplicity correction or generalization. No new independent score selection.",
        },
        output / "registration.json",
    )
    verify(hashes)
    registered(calibrate(), output / "calibration.json")
    registered(schema_controls(), output / "schema-controls.json")
    print("analytic calibration complete", flush=True)
    records, old = read(root / "records.json"), read(root / "assessment.json")
    config = read(root / "registration.json")
    comparisons = {key: describe_comparison(value) for key, value in old["comparisons"].items()}
    reproduced = []
    for population, role, corpus, origin, arm in itertools.product(
        ("fit", "transfer"),
        ("paired", "identity"),
        ("all", "music_library", "speech_library"),
        config["endpoints"],
        ("realized_phase", "circular_relation", "wrapped_relation"),
    ):
        selected = [
            r
            for r in records
            if r["population"] == population
            and r["role"] == role
            and (corpus == "all" or r["corpus"] == corpus)
        ]
        key = f"{population}/{role}/{corpus}/native/{origin}/{arm}-vs-old"
        current = strict_paired_recordings(
            selected, f"native/{origin}/{arm}", f"native/{origin}/old", seed=80
        )
        previous = {
            metric: {k: v for k, v in item.items() if k != "p77_numerical_scale"}
            for metric, item in old["comparisons"][key].items()
        }
        if current != previous:
            raise AssertionError(f"historical score changed: {key}")
        reproduced.append(key)
    primary = {
        key: value
        for key, value in comparisons.items()
        if key.startswith("transfer/paired/all/native/") and key.endswith("-vs-old")
    }
    registered(
        {
            "comparisons": comparisons,
            "primary_native_transfer": primary,
            "strictly_reproduced_native_comparisons": reproduced,
            "historical_joint_transfer_gate_unchanged": old["joint_transfer_gate"],
            "qualified": False,
            "scope": "Post-hoc diagnostic reinterpretation, not a newly preregistered candidate result. No changed score, new native output, perceptual result, training or promotion. New spectral-dynamics fields are absent from the old recording scores.",
        },
        output / "assessment.json",
    )
    verify(hashes)
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
    print(
        f"complete: {len(comparisons)} descriptions; {len(reproduced)} exact native comparisons",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh = not args.output.exists()
    try:
        run(args.root.resolve(), args.output.resolve())
    except BaseException:
        if fresh and args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
