"""Strict SXA1 transport for the Rust production streaming evaluator."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MAGIC = b"SXA1"
_HEADER = struct.Struct("<4sIHHQ")


@dataclass(frozen=True)
class RustEvaluationOutput:
    audio: np.ndarray
    report: dict[str, Any]


class RustStreamEvaluator:
    """Invoke `soundex-stream-eval` without bypassing Rust DSP or streaming."""

    def __init__(self, binary: str | Path | None = None) -> None:
        repository = Path(__file__).resolve().parents[2]
        configured = binary or os.environ.get("SOUNDEX_STREAM_EVAL")
        self.binary = (
            Path(configured) if configured else repository / "target/debug/soundex-stream-eval"
        )
        self.repository = repository

    def ensure_available(self) -> None:
        """Build the locked Rust bridge only when no executable is present."""
        if self.binary.is_file() and os.access(self.binary, os.X_OK):
            return
        result = subprocess.run(
            ["cargo", "build", "--locked", "-p", "soundex-core", "--bin", "soundex-stream-eval"],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not self.binary.is_file():
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"failed to build Rust stream evaluator: {detail}")

    def process(
        self,
        model_path: str | Path,
        audio: np.ndarray,
        sample_rate: int,
        *,
        mode: str,
        chunk_frames: tuple[int, ...] = (),
        fft_size: int = 256,
        hop_size: int = 128,
        crossover_width_hz: float = 1000.0,
    ) -> RustEvaluationOutput:
        """Run one offline or chunked evaluation and return aligned audio/report."""
        self.ensure_available()
        with tempfile.TemporaryDirectory(prefix="soundex-eval-") as directory:
            root = Path(directory)
            input_path = root / "input.sxa"
            output_path = root / "output.sxa"
            report_path = root / "report.json"
            write_sxa(input_path, audio, sample_rate)
            command = [
                str(self.binary),
                str(Path(model_path).resolve()),
                str(input_path),
                str(output_path),
                str(report_path),
                mode,
            ]
            if mode == "chunked":
                if not chunk_frames or any(frame < 1 for frame in chunk_frames):
                    raise ValueError("chunked evaluation requires positive chunk frame counts")
                command.append(",".join(str(frame) for frame in chunk_frames))
            elif mode != "offline" or chunk_frames:
                raise ValueError("mode must be offline or chunked without an invalid pattern")
            environment = os.environ.copy()
            environment.update(
                {
                    "SOUNDEX_EVAL_FFT_SIZE": str(fft_size),
                    "SOUNDEX_EVAL_HOP_SIZE": str(hop_size),
                    "SOUNDEX_EVAL_CROSSOVER_WIDTH_HZ": str(crossover_width_hz),
                }
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Rust stream evaluation failed: {detail}")
            enhanced, output_rate = read_sxa(output_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        expected = _as_frames_channels(audio)
        if output_rate != sample_rate or enhanced.shape != expected.shape:
            raise RuntimeError(
                f"Rust stream shape/rate mismatch: {enhanced.shape}@{output_rate}, "
                f"expected {expected.shape}@{sample_rate}"
            )
        if int(report.get("schema_version", 0)) != 1 or report.get("mode") != mode:
            raise RuntimeError("Rust stream report schema or mode mismatch")
        if not bool(report.get("processor", {}).get("output_finite", False)):
            raise RuntimeError("Rust stream report did not certify finite output")
        return RustEvaluationOutput(audio=enhanced, report=report)


def write_sxa(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write strict little-endian interleaved FP32 SXA1 audio."""
    array = _as_frames_channels(audio)
    if sample_rate not in {44_100, 48_000}:
        raise ValueError("SXA1 sample rate must be 44100 or 48000")
    if array.shape[1] not in {1, 2}:
        raise ValueError("SXA1 supports one or two channels")
    if not np.isfinite(array).all():
        raise ValueError("SXA1 input contains non-finite samples")
    header = _HEADER.pack(_MAGIC, sample_rate, array.shape[1], 0, array.shape[0])
    payload = np.asarray(array, dtype="<f4", order="C").tobytes(order="C")
    Path(path).write_bytes(header + payload)


def read_sxa(path: str | Path) -> tuple[np.ndarray, int]:
    """Read SXA1 while rejecting unsupported flags, truncation, and NaNs."""
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("truncated SXA1 header")
    magic, sample_rate, channels, flags, frames = _HEADER.unpack_from(raw)
    if magic != _MAGIC or flags != 0:
        raise ValueError("invalid SXA1 magic or flags")
    if sample_rate not in {44_100, 48_000} or channels not in {1, 2}:
        raise ValueError("unsupported SXA1 sample rate or channel count")
    sample_count = frames * channels
    expected_bytes = _HEADER.size + sample_count * 4
    if len(raw) != expected_bytes:
        raise ValueError("SXA1 payload length does not match header")
    audio = np.frombuffer(raw, dtype="<f4", offset=_HEADER.size).reshape(frames, channels).copy()
    if not np.isfinite(audio).all():
        raise ValueError("SXA1 payload contains non-finite samples")
    return audio, sample_rate


def _as_frames_channels(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("audio must have shape [frames] or [frames, channels]")
    return array
