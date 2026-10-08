# SPDX-License-Identifier: Apache-2.0
"""Verify clean-reference encoding; container checks cannot prove mastering history."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

LOSSLESS_FORMATS = frozenset({"WAV", "WAVEX", "RF64", "FLAC", "AIFF", "W64", "CAF"})
PCM_SUBTYPES = frozenset({"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"})


def lossless_source_info(path: str | Path) -> sf._SoundFileInfo:
    """Reject lossy encodings and ambiguous channel layouts before degradation."""
    info = sf.info(path)
    if info.format not in LOSSLESS_FORMATS or info.subtype not in PCM_SUBTYPES:
        raise ValueError(
            f"clean targets require lossless PCM/FLAC audio, got "
            f"{info.format}/{info.subtype}: {path}"
        )
    if info.channels not in {1, 2} or info.frames == 0:
        raise ValueError(f"recordings require nonempty mono/stereo audio: {path}")
    return info
