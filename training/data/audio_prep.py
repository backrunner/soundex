"""Audio degradation, alignment, segmentation, and manifest-row generation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from .protocol import CodecSpec, DataProtocolError, DataRecipe, DatasetPublisher, sha256_file
except ImportError:
    from protocol import CodecSpec, DataProtocolError, DataRecipe, DatasetPublisher, sha256_file

AudioArray = np.ndarray
Encoder = Callable[[AudioArray, int, CodecSpec], AudioArray]


@dataclass(frozen=True)
class AlignedPair:
    """A sample-aligned clean/degraded pair and measured offsets."""

    clean: AudioArray
    degraded: AudioArray
    offset_samples: int
    residual_offset_samples: int


def load_audio(path: Path, target_sr: int) -> AudioArray:
    """Load mono/stereo float32 audio, preserving channels while resampling."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if sample_rate == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is required to resample {path} from {sample_rate} to {target_sr} Hz"
        )
    with tempfile.TemporaryDirectory() as temporary:
        output_path = Path(temporary) / "resampled.wav"
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-ar",
                str(target_sr),
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
        )
        audio, actual_rate = sf.read(str(output_path), dtype="float32", always_2d=True)
    if actual_rate != target_sr:
        raise DataProtocolError(
            f"resample failed for {path}: got {actual_rate} Hz, expected {target_sr}"
        )
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.ascontiguousarray(audio, dtype=np.float32)


def encode_decode_ffmpeg(audio: AudioArray, sample_rate: int, spec: CodecSpec) -> AudioArray:
    """Round-trip audio through one explicitly configured FFmpeg encoder."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for codec degradation")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_path = root / "input.wav"
        encoded_path = root / f"encoded.{spec.container}"
        decoded_path = root / "decoded.wav"
        sf.write(str(input_path), audio, sample_rate, subtype="FLOAT")
        setting_flag = "-b:a" if spec.mode == "cbr" else "-q:a"
        setting_value = f"{int(spec.setting)}k" if spec.mode == "cbr" else f"{spec.setting:g}"
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-c:a",
                spec.encoder,
                setting_flag,
                setting_value,
                str(encoded_path),
            ]
        )
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(encoded_path),
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_f32le",
                str(decoded_path),
            ]
        )
        degraded, actual_rate = sf.read(str(decoded_path), dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise DataProtocolError(
            f"codec {spec.id} decoded at {actual_rate} Hz, expected {sample_rate}"
        )
    if degraded.shape[1] != audio.shape[1]:
        raise DataProtocolError(
            f"codec {spec.id} changed channel count from {audio.shape[1]} to {degraded.shape[1]}"
        )
    return np.ascontiguousarray(degraded, dtype=np.float32)


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        detail = message[-1] if message else f"exit status {result.returncode}"
        raise RuntimeError(f"FFmpeg failed: {detail}")


def ffmpeg_encoders_available() -> set[str]:
    """Return audio encoder names advertised by the local FFmpeg build."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return set()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, check=False, text=True
    )
    if result.returncode != 0:
        return set()
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("A"):
            encoders.add(fields[1])
    return encoders


def validate_recipe_encoders(recipe: DataRecipe) -> None:
    """Fail early when an enabled encoder is absent from FFmpeg."""
    available = ffmpeg_encoders_available()
    if not available:
        raise RuntimeError("ffmpeg with audio encoders is required for preprocessing")
    missing = sorted({spec.encoder for spec in recipe.codecs} - available)
    if missing:
        raise RuntimeError(f"FFmpeg is missing recipe encoders: {', '.join(missing)}")


def _mono_lowpass(audio: AudioArray, sample_rate: int, cutoff_hz: float = 8000.0) -> AudioArray:
    mono = audio.mean(axis=1, dtype=np.float64)
    if mono.size < 2:
        return mono.astype(np.float32)
    spectrum = np.fft.rfft(mono)
    frequencies = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    spectrum[frequencies > min(cutoff_hz, sample_rate * 0.45)] = 0.0
    return np.fft.irfft(spectrum, n=mono.size).astype(np.float32)


def _full_correlation(left: AudioArray, right: AudioArray) -> AudioArray:
    output_length = left.size + right.size - 1
    fft_length = 1 << max(output_length - 1, 0).bit_length()
    left_fft = np.fft.rfft(left, fft_length)
    reversed_right_fft = np.fft.rfft(right[::-1], fft_length)
    return np.fft.irfft(left_fft * reversed_right_fft, fft_length)[:output_length]


def estimate_delay_samples(
    clean: AudioArray,
    degraded: AudioArray,
    sample_rate: int,
    *,
    max_offset_samples: int,
    correlation_samples: int,
) -> int:
    """Estimate degraded delay with bounded low-band FFT cross-correlation."""
    if clean.ndim == 1:
        clean = clean[:, None]
    if degraded.ndim == 1:
        degraded = degraded[:, None]
    usable = min(clean.shape[0], degraded.shape[0], correlation_samples + max_offset_samples)
    if usable < 2:
        return 0
    clean_probe = _mono_lowpass(clean[:usable], sample_rate)
    degraded_probe = _mono_lowpass(degraded[:usable], sample_rate)
    clean_probe = clean_probe - clean_probe.mean()
    degraded_probe = degraded_probe - degraded_probe.mean()
    if np.linalg.norm(clean_probe) < 1e-10 or np.linalg.norm(degraded_probe) < 1e-10:
        return 0
    correlation = _full_correlation(degraded_probe, clean_probe)
    lags = np.arange(-clean_probe.size + 1, degraded_probe.size)
    allowed = np.abs(lags) <= max_offset_samples
    if not np.any(allowed):
        return 0
    allowed_indices = np.flatnonzero(allowed)
    best = allowed_indices[int(np.argmax(correlation[allowed]))]
    return int(lags[best])


def align_codec_pair(
    clean: AudioArray,
    degraded: AudioArray,
    sample_rate: int,
    recipe: DataRecipe,
) -> AlignedPair:
    """Estimate codec delay, apply it, and enforce residual tolerance."""
    if clean.ndim == 1:
        clean = clean[:, None]
    if degraded.ndim == 1:
        degraded = degraded[:, None]
    if clean.shape[1] != degraded.shape[1]:
        raise DataProtocolError("clean/degraded channel count differs before alignment")
    policy = recipe.alignment
    offset = estimate_delay_samples(
        clean,
        degraded,
        sample_rate,
        max_offset_samples=policy.max_offset_samples,
        correlation_samples=policy.correlation_samples,
    )
    if offset >= 0:
        degraded_aligned = degraded[offset:]
        clean_aligned = clean
    else:
        degraded_aligned = degraded
        clean_aligned = clean[-offset:]
    length = min(clean_aligned.shape[0], degraded_aligned.shape[0])
    clean_aligned = np.ascontiguousarray(clean_aligned[:length], dtype=np.float32)
    degraded_aligned = np.ascontiguousarray(degraded_aligned[:length], dtype=np.float32)
    residual = estimate_delay_samples(
        clean_aligned,
        degraded_aligned,
        sample_rate,
        max_offset_samples=max(policy.residual_tolerance_samples + 2, 4),
        correlation_samples=policy.correlation_samples,
    )
    if abs(residual) > policy.residual_tolerance_samples:
        raise DataProtocolError(
            f"residual codec alignment {residual} exceeds tolerance "
            f"{policy.residual_tolerance_samples}"
        )
    validate_pair_samples(clean_aligned, degraded_aligned)
    return AlignedPair(clean_aligned, degraded_aligned, offset, residual)


def validate_pair_samples(clean: AudioArray, degraded: AudioArray) -> None:
    """Reject incomplete, non-finite, silent, or implausibly scaled pairs."""
    if clean.shape != degraded.shape or clean.size == 0:
        raise DataProtocolError(f"clean/degraded shape mismatch: {clean.shape} vs {degraded.shape}")
    if not np.isfinite(clean).all() or not np.isfinite(degraded).all():
        raise DataProtocolError("clean/degraded pair contains non-finite samples")
    clean_rms = float(np.sqrt(np.mean(np.square(clean, dtype=np.float64))))
    if clean_rms < 1e-6:
        raise DataProtocolError("clean audio is silent")
    if float(np.max(np.abs(clean))) > 8.0 or float(np.max(np.abs(degraded))) > 8.0:
        raise DataProtocolError("clean/degraded peak exceeds safety bound")


def derive_channel_roles(
    clean: AudioArray,
    degraded: AudioArray,
    role_weights: Mapping[str, float],
) -> dict[str, tuple[AudioArray, AudioArray, float]]:
    """Create mono or left/right/mid/side examples after stereo coding."""
    if clean.shape != degraded.shape or clean.ndim != 2:
        raise DataProtocolError("channel-role input must be aligned [samples, channels]")
    if clean.shape[1] == 1:
        return {
            "mono": (
                np.ascontiguousarray(clean[:, 0]),
                np.ascontiguousarray(degraded[:, 0]),
                float(role_weights["mono"]),
            )
        }
    if clean.shape[1] != 2:
        raise DataProtocolError(f"expected mono or stereo source, got {clean.shape[1]} channels")
    scale = np.float32(1.0 / np.sqrt(2.0))
    return {
        "left": (clean[:, 0].copy(), degraded[:, 0].copy(), float(role_weights["left"])),
        "right": (clean[:, 1].copy(), degraded[:, 1].copy(), float(role_weights["right"])),
        "mid": (
            np.ascontiguousarray((clean[:, 0] + clean[:, 1]) * scale),
            np.ascontiguousarray((degraded[:, 0] + degraded[:, 1]) * scale),
            float(role_weights["mid"]),
        ),
        "side": (
            np.ascontiguousarray((clean[:, 0] - clean[:, 1]) * scale),
            np.ascontiguousarray((degraded[:, 0] - degraded[:, 1]) * scale),
            float(role_weights["side"]),
        ),
    }


def deterministic_segments(
    length: int,
    sample_rate: int,
    recipe: DataRecipe,
    seed_material: str,
) -> list[tuple[int, int]]:
    """Choose reproducible segment starts and lengths for one degradation row."""
    if length <= 0:
        return []
    digest = hashlib.sha256(f"{recipe.seed}:{seed_material}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    generator = np.random.default_rng(seed)
    minimum, maximum = recipe.segment.duration_seconds
    segments: list[tuple[int, int]] = []
    for _ in range(recipe.segment.count_per_track):
        duration = float(generator.uniform(minimum, maximum))
        segment_length = min(length, max(1, round(duration * sample_rate)))
        max_start = length - segment_length
        start = int(generator.integers(0, max_start + 1)) if max_start > 0 else 0
        segments.append((start, segment_length))
    return segments


def measure_cutoff_hz(audio: AudioArray, sample_rate: int) -> float:
    """Measure a robust full-segment bandwidth from overlapping spectral frames."""
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size < 2 or sample_rate <= 0:
        return 0.0
    window_length = min(samples.size, 8192)
    hop = max(window_length // 2, 1)
    final_start = samples.size - window_length
    starts = list(range(0, final_start + 1, hop))
    if starts[-1] != final_start:
        starts.append(final_start)
    window = np.hanning(window_length + 1)[:-1]
    envelope = np.zeros(window_length // 2 + 1, dtype=np.float64)
    for start in starts:
        spectrum = np.fft.rfft(samples[start : start + window_length] * window)
        np.maximum(envelope, np.abs(spectrum), out=envelope)
    peak = float(envelope.max(initial=0.0))
    if peak <= 1e-10:
        return 0.0
    active = np.flatnonzero(envelope > peak * 1e-3)
    if active.size == 0:
        return 0.0
    return float(active[-1] * sample_rate / window_length)


def process_mixture_file(
    mixture_path: Path,
    publisher: DatasetPublisher,
    *,
    corpus: str,
    corpus_version: str,
    track_id: str,
    split: str,
    recipe: DataRecipe,
    encoder: Encoder = encode_decode_ffmpeg,
) -> list[dict[str, object]]:
    """Generate all configured degradation rows for one full-track mixture."""
    mixture_path = Path(mixture_path).resolve()
    source_checksum = sha256_file(mixture_path)
    role_weights = dict(recipe.channel_roles)
    rows: list[dict[str, object]] = []
    for sample_rate in recipe.sample_rates:
        clean = load_audio(mixture_path, sample_rate)
        if clean.size == 0:
            continue
        source_channels = clean.shape[1]
        for codec in recipe.codecs:
            if split == "train" and codec.held_out:
                continue
            degraded = encoder(clean, sample_rate, codec)
            aligned = align_codec_pair(clean, degraded, sample_rate, recipe)
            roles = derive_channel_roles(aligned.clean, aligned.degraded, role_weights)
            for role, (clean_role, degraded_role, role_weight) in roles.items():
                seed_material = f"{track_id}:{split}:{sample_rate}:{codec.id}:{role}"
                for segment_index, (start, length) in enumerate(
                    deterministic_segments(len(clean_role), sample_rate, recipe, seed_material)
                ):
                    clean_segment = np.ascontiguousarray(clean_role[start : start + length])
                    degraded_segment = np.ascontiguousarray(degraded_role[start : start + length])
                    validate_pair_samples(clean_segment, degraded_segment)
                    row_material = f"{seed_material}:{segment_index}:{start}:{length}"
                    row_id = hashlib.sha256(row_material.encode()).hexdigest()[:24]
                    audio_dir = publisher.staging / "audio" / split
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    clean_relative = Path("audio") / split / f"{row_id}_clean.wav"
                    degraded_relative = Path("audio") / split / f"{row_id}_degraded.wav"
                    clean_path = publisher.staging / clean_relative
                    degraded_path = publisher.staging / degraded_relative
                    sf.write(str(clean_path), clean_segment, sample_rate, subtype="FLOAT")
                    sf.write(str(degraded_path), degraded_segment, sample_rate, subtype="FLOAT")
                    rows.append(
                        {
                            "row_id": row_id,
                            "schema_version": recipe.schema_version,
                            "recipe_hash": recipe.hash,
                            "corpus": corpus,
                            "corpus_version": corpus_version,
                            "track_id": track_id,
                            "split": split,
                            "source_id": track_id,
                            "source_path": str(mixture_path),
                            "source_checksum": source_checksum,
                            "sample_rate": sample_rate,
                            "source_channels": source_channels,
                            "channel_role": role,
                            "channel_weight": role_weight,
                            "codec_id": codec.id,
                            "codec": codec.codec,
                            "encoder": codec.encoder,
                            "codec_mode": codec.mode,
                            "codec_setting": codec.setting,
                            "codec_options": json.dumps(
                                {"mode": codec.mode, "setting": codec.setting}, sort_keys=True
                            ),
                            "codec_weight": codec.weight,
                            "sampling_weight": codec.weight * role_weight,
                            "held_out": codec.held_out,
                            "clean_path": clean_relative.as_posix(),
                            "degraded_path": degraded_relative.as_posix(),
                            "start_sample": start,
                            "length_samples": length,
                            "alignment_offset_samples": aligned.offset_samples,
                            "residual_alignment_samples": aligned.residual_offset_samples,
                            "measured_cutoff_hz": measure_cutoff_hz(degraded_segment, sample_rate),
                            "clean_checksum": sha256_file(clean_path),
                            "degraded_checksum": sha256_file(degraded_path),
                        }
                    )
    return rows
