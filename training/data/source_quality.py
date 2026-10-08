# SPDX-License-Identifier: Apache-2.0
"""Measure complete lossless recordings without inferring mastering history from spectra."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from data.input_audio import lossless_source_info


def inspect_master(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Verify bytes, decode every frame and report signal measurements and review flags.

    Spectral measurements sample 2048 frames per decode block. They describe the
    signal, not whether its production ever involved a lossy codec. Quiet acoustic
    music and intentional distortion need different interpretations; review flags
    are not an assertion that a recording is defective.
    """
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            sha.update(chunk)
    checksum = sha.hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("source checksum mismatch")
    info = lossless_source_info(path)
    pcm = hashlib.sha256(f"{info.samplerate}:{info.channels}:".encode())
    frames = 0
    peak = 0.0
    energy = np.zeros(info.channels, dtype=np.float64)
    sums = np.zeros(info.channels, dtype=np.float64)
    rail_samples = 0
    silent_frames = 0
    spectrum = np.zeros(1025, dtype=np.float64)
    spectrum_windows = 0
    longest_rail = 0
    trailing = np.zeros(info.channels, dtype=np.int64)
    window = np.hanning(2048)
    with sf.SoundFile(path) as audio:
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("non-finite decoded audio")
            pcm.update(block.astype("<f4", copy=False).tobytes())
            frames += len(block)
            magnitude = np.abs(block)
            peak = max(peak, float(magnitude.max(initial=0.0)))
            energy += np.square(block, dtype=np.float64).sum(axis=0)
            sums += block.sum(axis=0, dtype=np.float64)
            rail = magnitude >= 1.0 - 1.0 / 32768
            rail_samples += int(rail.sum())
            silent_frames += int(np.all(magnitude < 1e-5, axis=1).sum())
            for channel in range(info.channels):
                mask = rail[:, channel]
                transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
                starts = np.flatnonzero(transitions == 1)
                ends = np.flatnonzero(transitions == -1)
                if starts.size:
                    lengths = ends - starts
                    if starts[0] == 0:
                        lengths[0] += trailing[channel]
                    longest_rail = max(longest_rail, int(lengths.max()))
                    trailing[channel] = lengths[-1] if ends[-1] == len(block) else 0
                else:
                    trailing[channel] = 0
            if len(block) >= 2048:
                # Sum channel powers: mono fold-down could cancel antiphase stereo.
                segment = block[:2048].astype(np.float64)
                segment -= segment.mean(axis=0)
                spectrum += np.square(np.abs(np.fft.rfft(segment * window[:, None], axis=0))).sum(
                    axis=1
                )
                spectrum_windows += 1
    if frames != info.frames:
        raise ValueError("incomplete decoded recording")
    rms = np.sqrt(energy / frames)
    if peak > 8 or float(rms.max()) < 1e-6:
        raise ValueError("silent or implausibly scaled master")
    dc = sums / frames
    rail_fraction = rail_samples / (frames * info.channels)
    power = float(spectrum.sum())
    frequencies = np.fft.rfftfreq(2048, 1.0 / info.samplerate)
    bandwidth = (
        float(frequencies[min(int(np.searchsorted(np.cumsum(spectrum), power * 0.999)), 1024)])
        if power > 0
        else None
    )
    high_ratio = float(spectrum[frequencies >= 16000].sum() / power) if power > 0 else None
    flags: list[str] = []
    if peak > 1.0001:
        flags.append("over_full_scale")
    if rail_fraction > 0.001 and longest_rail >= 3:
        flags.append("sustained_full_scale_samples")
    if float(np.abs(dc).max()) > 0.01:
        flags.append("large_dc_offset")
    if float(rms.max()) < 1e-3:
        flags.append("very_quiet_recording")
    if silent_frames / frames > 0.5:
        flags.append("mostly_silence")
    return {
        "quality_schema": 1,
        "audio_sha256": checksum,
        "decoded_pcm_sha256": pcm.hexdigest(),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": frames,
        "duration_seconds": frames / info.samplerate,
        "peak": peak,
        "rms_per_channel": rms.tolist(),
        "dc_per_channel": dc.tolist(),
        "full_scale_sample_fraction": rail_fraction,
        "longest_full_scale_run": longest_rail,
        "silent_frame_fraction": silent_frames / frames,
        "spectral_windows": spectrum_windows,
        "bandwidth_99_9_hz": bandwidth,
        "power_above_16khz_fraction": high_ratio,
        "review_flags": flags,
        "mastering_history_verified_by_spectrum": False,
    }
