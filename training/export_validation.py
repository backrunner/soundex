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
    MAX_ABSOLUTE_ERROR,
    MAX_MEAN_ERROR,
    OUTPUT_NAME,
    REQUIRED_METADATA_KEYS,
    ExportValidationError,
    tensor_shape,
)
from models.generator import SoundExGenerator

OutputTransform = Callable[[str, np.ndarray], np.ndarray]


def validate_ort_parity(
    model: SoundExGenerator,
    model_path: str | Path,
    *,
    output_transform: OutputTransform | None = None,
) -> tuple[float, float]:
    """Compare PyTorch and CPU ORT over deterministic deployment edge cases."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
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
        difference = np.abs(actual - expected)
        case_max = float(difference.max())
        case_mean = float(difference.mean())
        maximum = max(maximum, case_max)
        maximum_mean = max(maximum_mean, case_mean)
        if case_max >= MAX_ABSOLUTE_ERROR or case_mean >= MAX_MEAN_ERROR:
            raise ExportValidationError(
                f"ORT case {case_name!r} mismatch: max={case_max:.3e}, mean={case_mean:.3e}"
            )
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
    return [
        ("silence", silence),
        ("near_full_scale_sine", sine.contiguous()),
        ("random_finite_spectra", random_spectra),
        ("db_floor", floor_values),
        ("phase_edges", phase_edges),
    ]
