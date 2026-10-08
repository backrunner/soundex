# SPDX-License-Identifier: Apache-2.0
"""Attenuate valid floating-point masters without clipping or altering their timing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from data.source_quality import inspect_master


def attenuate_float_master(
    source: Path, target: Path, *, source_sha256: str, target_peak: float = 0.95
) -> dict[str, Any]:
    """Publish a gain-only WAV derivative and a measured, hash-bound edit receipt.

    Only float masters whose sole review flag is over-full-scale qualify. Apply
    one gain to every channel, retain all frames and verify every output sample
    against the mathematical transform. Source bytes stay unchanged. The caller
    stores this receipt and the source's own grant alongside the derivative.
    """
    source, target = source.resolve(), target.resolve()
    if source == target or target.exists():
        raise ValueError("a new derivative path is required")
    if not np.isfinite(target_peak) or not 0 < target_peak <= 0.95:
        raise ValueError("target peak must be finite and in (0, 0.95]")
    original = inspect_master(source, source_sha256)
    if original["subtype"] not in {"FLOAT", "DOUBLE"} or original["review_flags"] != [
        "over_full_scale"
    ]:
        raise ValueError("only otherwise-valid over-full-scale float masters qualify")
    gain = target_peak / original["peak"]
    subtype = original["subtype"]
    precision = np.finfo(np.float32 if subtype == "FLOAT" else np.float64)
    maximum_error = 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, suffix=".wav.part")
    os.close(descriptor)
    pending = Path(name)
    try:
        with (
            sf.SoundFile(source) as audio,
            sf.SoundFile(
                pending,
                mode="w",
                samplerate=audio.samplerate,
                channels=audio.channels,
                format="WAV",
                subtype=subtype,
            ) as output,
        ):
            for block in audio.blocks(blocksize=65536, dtype="float64", always_2d=True):
                output.write(block * gain)
        with sf.SoundFile(source) as audio, sf.SoundFile(pending) as output:
            if len(output) != len(audio):
                raise ValueError("derivative frame count changed")
            for block in audio.blocks(blocksize=65536, dtype="float64", always_2d=True):
                actual = output.read(len(block), dtype="float64", always_2d=True)
                error = float(np.max(np.abs(actual - block * gain), initial=0.0))
                maximum_error = max(maximum_error, error)
                if error > precision.eps * target_peak:
                    raise ValueError("derivative exceeds native float rounding tolerance")
        quality = inspect_master(pending)
        if quality["review_flags"]:
            raise ValueError("gain derivative still needs a separate signal review")
        checksum = hashlib.sha256()
        with source.open("rb") as audio:
            while chunk := audio.read(1 << 20):
                checksum.update(chunk)
        if checksum.hexdigest() != source_sha256:
            raise ValueError("source changed during derivative preparation")
        # Same-directory hard link publishes atomically without replacing a file
        # another writer may have created since the initial path check.
        os.link(pending, target)
        return {
            "edit": "uniform attenuation of all channels; no clipping, crops or resampling",
            "original_path": str(source),
            "original_sha256": source_sha256,
            "original_quality": original,
            "derivative_path": str(target),
            "derivative_sha256": quality["audio_sha256"],
            "common_gain": gain,
            "gain_db": float(20 * np.log10(gain)),
            "target_peak": target_peak,
            "maximum_absolute_rounding_error": maximum_error,
            "rounding_error_budget": float(precision.eps * target_peak),
            "source_quality": quality,
        }
    finally:
        pending.unlink(missing_ok=True)
