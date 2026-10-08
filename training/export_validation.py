"""Mandatory deterministic PyTorch-to-ONNX Runtime parity validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from artifact_contract import (
    DEFAULT_FFT_SIZE,
    DEFAULT_HOP_SIZE,
    INPUT_NAME,
    OUTPUT_NAME,
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    tensor_shape,
)
from models.generator import SoundExGenerator
from parity_metrics import validate_features

OutputTransform = Callable[[str, np.ndarray], np.ndarray]


def validate_ort_parity(
    model: SoundExGenerator,
    model_path: str | Path,
    *,
    output_transform: OutputTransform | None = None,
) -> tuple[float, float]:
    """Compare PyTorch and CPU ORT over deterministic deployment edge cases."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].name != INPUT_NAME or session.get_outputs()[0].name != OUTPUT_NAME:
        raise ExportValidationError("ONNX Runtime tensor names do not match the artifact contract")
    metadata = session.get_modelmeta().custom_metadata_map
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ExportValidationError(f"ONNX Runtime cannot read metadata key {missing[0]}")

    maximum = 0.0
    maximum_mean = 0.0
    try:
        fft_size = int(metadata["soundex.fft_size"])
        hop_size = int(metadata["soundex.hop_size"])
    except ValueError as error:
        raise ExportValidationError("ONNX Runtime frame metadata is not numeric") from error

    for case_name, input_tensor in deterministic_parity_inputs(
        fft_size=fft_size,
        hop_size=hop_size,
    ):
        input_array = input_tensor.numpy()
        with torch.no_grad():
            expected = model(input_tensor).numpy()
        actual = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        if output_transform is not None:
            actual = output_transform(case_name, actual)
        if actual.shape != expected.shape:
            raise ExportValidationError(
                f"ORT case {case_name!r} returned shape {actual.shape}, expected {expected.shape}"
            )
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ExportValidationError(f"ORT case {case_name!r} produced non-finite values")
        metrics = validate_features(expected, actual, label=f"ORT case {case_name!r}")
        case_max = metrics["raw_max_absolute_error"]
        case_mean = metrics["raw_mean_error"]
        maximum = max(maximum, case_max)
        maximum_mean = max(maximum_mean, case_mean)
    return maximum, maximum_mean


def deterministic_parity_inputs(
    *, fft_size: int = DEFAULT_FFT_SIZE, hop_size: int = DEFAULT_HOP_SIZE
) -> list[tuple[str, torch.Tensor]]:
    """Cover silence, real STFT, random spectra, floor, and phase wrapping."""
    floor = -200.0
    static_shape = tensor_shape(fft_size, hop_size)
    frequency_bins = static_shape[-1]
    silence = torch.zeros((1, *static_shape), dtype=torch.float32)
    silence[:, 0].fill_(floor)

    time = torch.arange(fft_size, dtype=torch.float32) / 44_100.0
    waveform = 0.999 * torch.sin(2.0 * math.pi * 997.0 * time)
    window = torch.hann_window(fft_size, periodic=True)
    spectrum = torch.stft(
        waveform.unsqueeze(0),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        center=False,
        return_complex=True,
    )
    sine = torch.stack(
        (
            20.0 * spectrum.abs().clamp_min(1e-10).log10(),
            torch.angle(spectrum),
        ),
        dim=1,
    ).transpose(2, 3)

    generator = torch.Generator().manual_seed(0x5EED)
    random_spectra = torch.empty((2, *static_shape), dtype=torch.float32)
    random_spectra[:, 0].uniform_(floor, 6.0, generator=generator)
    random_spectra[:, 1].uniform_(-math.pi, math.pi, generator=generator)

    floor_values = torch.zeros((1, *static_shape), dtype=torch.float32)
    floor_values[:, 0].fill_(floor)
    floor_values[:, 1] = torch.linspace(-math.pi, math.pi, frequency_bins)

    phase_edges = torch.full((2, *static_shape), -30.0, dtype=torch.float32)
    phase_edges[0, 1].fill_(math.pi - 1e-6)
    phase_edges[1, 1].fill_(-math.pi + 1e-6)
    cases = [
        ("silence", silence),
        ("near_full_scale_sine", sine.contiguous()),
        ("random_finite_spectra", random_spectra),
        ("db_floor", floor_values),
        ("phase_edges", phase_edges),
    ]
    generator = torch.Generator().manual_seed(501)
    for sample_rate in (44_100, 48_000):
        time = torch.arange(fft_size, dtype=torch.float32) / sample_rate
        impulse = torch.zeros(fft_size)
        impulse[fft_size // 2] = 0.9
        waveforms = {
            "music_tones": sum(
                amplitude * torch.sin(2 * math.pi * frequency * time)
                for amplitude, frequency in ((0.2, 440), (0.1, 880), (0.05, 6000))
            ),
            "quiet": 1e-5 * torch.sin(2 * math.pi * 997 * time),
            "noise": torch.randn(fft_size, generator=generator) * 0.2,
            "impulse": impulse,
        }
        for name, waveform in waveforms.items():
            spectrum = torch.fft.rfft(waveform * window)
            features = torch.stack(
                (20 * spectrum.abs().clamp_min(1e-10).log10(), torch.angle(spectrum))
            )[None, :, None, :]
            cases.append((f"{name}_{sample_rate}", features.contiguous()))
    return cases
