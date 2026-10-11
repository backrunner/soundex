"""Frozen paired draws and input-only Rust detector traces for the P2 study."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from data.protocol import load_manifest, validate_manifest_rows
from data.sampling_weights import recording_weights, speech_mix_ratios
from evaluation.reporting import sha256_file
from evaluation.rust_bridge import write_sxa
from models.causal_objective import BURN, gate_envelope
from models.losses import build_missing_band_mask
from train import spectral_features

FRAMES = 256
SAMPLES = FRAMES * 128


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_sources(manifests):
    rows, hashes = {}, {}
    for path in manifests:
        values = load_manifest(path)
        validate_manifest_rows(values, path.parent, audit_files=False)
        hashes[str(path.resolve())] = sha256_file(path)
        for row in values:
            if row["row_id"] in rows:
                raise ValueError("duplicate row across manifests")
            rows[row["row_id"]] = {**row, "_root": str(path.parent.resolve())}
    if {r["corpus"] for r in rows.values()} != {"music_library", "speech_library"}:
        raise ValueError("P2 requires exactly reviewed music_library and speech_library")
    if len({r["recipe_hash"] for r in rows.values()}) != 1:
        raise ValueError("mixed recipes")
    # Recording and original-source hashes must not cross any split boundary.
    for field in ("track_id", "source_checksum"):
        seen = {}
        for row in rows.values():
            old = seen.setdefault(row[field], row["split"])
            if old != row["split"]:
                raise ValueError(f"split leakage in {field}")
    return rows, hashes


def make_schedule(rows, seed, draws):
    train = [r for r in rows.values() if r["split"] == "train" and not r["held_out"]]
    ratios, genre = speech_mix_ratios([r for r in train if r["corpus"] == "music_library"])
    if abs(ratios["speech_library"] - 1 / 254) > 1e-12:
        raise ValueError(
            "source ratio changed; re-register the study instead of silently renormalizing"
        )
    groups = {name: [r for r in train if r["corpus"] == name] for name in ratios}
    rng = np.random.default_rng(seed)
    corpus = rng.choice(list(ratios), size=draws, p=list(ratios.values()))
    schedules = {}
    for name, group in groups.items():
        weights = np.asarray(recording_weights(group), dtype=np.float64)
        schedules[name] = iter(
            rng.choice(len(group), size=int((corpus == name).sum()), p=weights / weights.sum())
        )
    result = []
    for name in corpus:
        row = groups[name][next(schedules[name])]
        start = int(rng.integers(max(1, int(row["length_samples"]) - SAMPLES + 1)))
        result.append(
            {"row_id": row["row_id"], "start": start, "gain_db": float(rng.uniform(-6, 3))}
        )
    return result, {
        "ratios": ratios,
        "genre": genre,
        "draws_by_corpus": {n: int((corpus == n).sum()) for n in groups},
        "available_recordings": {n: len({r["track_id"] for r in g}) for n, g in groups.items()},
        "drawn_recordings": len({rows[r["row_id"]]["track_id"] for r in result}),
        "supervised_audio_seconds": sum(
            max(
                0, min(SAMPLES, rows[r["row_id"]]["length_samples"] - r["start"]) - (BURN + 1) * 128
            )
            / rows[r["row_id"]]["sample_rate"]
            for r in result
        ),
    }


class PreparedClips:
    def __init__(self, rows, detector, cache):
        self.rows, self.detector, self.cache = rows, Path(detector).resolve(), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.detector_sha = sha256_file(self.detector)
        self.audited = set()

    def get(self, spec):
        row = self.rows[spec["row_id"]]
        for role in ("clean", "degraded"):
            path = Path(row["_root"]) / row[f"{role}_path"]
            if str(path) not in self.audited:
                if sha256_file(path) != row[f"{role}_checksum"]:
                    raise ValueError(f"changed {role} PCM: {row['row_id']}")
                self.audited.add(str(path))
        key = digest(
            {
                "version": "p2-clips-v2",
                "spec": spec,
                "recipe": row["recipe_hash"],
                "clean": row["clean_checksum"],
                "degraded": row["degraded_checksum"],
                "detector": self.detector_sha,
            }
        )
        path = self.cache / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as value:
                return {k: torch.from_numpy(value[k].copy()) for k in value.files}
        waves = []
        for role in ("degraded", "clean"):
            wave, rate = sf.read(
                Path(row["_root"]) / row[f"{role}_path"], dtype="float32", always_2d=True
            )
            if (
                rate != row["sample_rate"]
                or wave.shape != (row["length_samples"], 1)
                or not np.isfinite(wave).all()
            ):
                raise ValueError("invalid paired PCM geometry")
            start = spec["start"]
            if start < 0 or start > max(0, len(wave) - SAMPLES):
                raise ValueError("invalid paired crop")
            wave = wave[start : start + SAMPLES, 0] * (10 ** (spec["gain_db"] / 20))
            waves.append(np.pad(wave, (0, SAMPLES - len(wave))))
        rate = int(row["sample_rate"])
        with tempfile.TemporaryDirectory(prefix="soundex-p2-trace-") as directory:
            root = Path(directory)
            write_sxa(root / "input.sxa", waves[0], rate)
            # The detector collection API also emits clean labels; these are ignored.
            # Give it input twice so no reference audio can enter the input features.
            subprocess.run(
                [
                    str(self.detector),
                    str(root / "input.sxa"),
                    str(root / "input.sxa"),
                    str(root / "trace.json"),
                    "1",
                ],
                check=True,
                capture_output=True,
            )
            trace = json.loads((root / "trace.json").read_text())[:FRAMES]
        if len(trace) != FRAMES or any(
            r["hop"] != i or r["channel"] != 0 for i, r in enumerate(trace)
        ):
            raise ValueError("bad detector clock")
        cutoff = torch.tensor([r["cutoff_hz"] for r in trace])
        active = torch.tensor([r["active"] for r in trace])
        active[:BURN] = False
        mask = build_missing_band_mask(
            cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
        )[:, 0, 0][None]
        x, y = [
            spectral_features(F.pad(torch.from_numpy(w)[None], (128, 0)), 256, 128)[0]
            for w in waves
        ]
        result = {
            "x": x,
            "y": y,
            "mask": mask,
            "active": active,
            "gate": gate_envelope(active[None])[0],
            "rate": torch.tensor([rate], dtype=torch.float32),
            "cutoff": cutoff,
            "dry": F.pad(torch.from_numpy(waves[0]), (128, 0))[:SAMPLES],
            "clean_wave": F.pad(torch.from_numpy(waves[1]), (128, 0))[:SAMPLES],
        }
        staged = path.with_suffix(".partial.npz")
        np.savez(staged, **{k: v.numpy() for k, v in result.items()})
        staged.replace(path)
        return result

    def batch(self, specs, device="cpu"):
        values = [self.get(spec) for spec in specs]
        return {k: torch.stack([v[k] for v in values]).to(device) for k in values[0]}
