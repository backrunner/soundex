"""Missing-band-aware losses for SoundEx spectral bandwidth extension."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_missing_band_mask(
    cutoff_hz: torch.Tensor,
    sample_rate: torch.Tensor,
    *,
    fft_size: int,
    crossover_width_hz: float,
) -> torch.Tensor:
    """Return Rust-equivalent raised-cosine blend weights `[B, 1, 1, F]`."""
    if fft_size < 2 or not fft_size & (fft_size - 1) == 0:
        raise ValueError("fft_size must be a power of two >= 2")
    if crossover_width_hz < 0.0 or not math.isfinite(crossover_width_hz):
        raise ValueError("crossover_width_hz must be finite and non-negative")
    cutoff = cutoff_hz.to(dtype=torch.float32).reshape(-1, 1)
    rates = sample_rate.to(device=cutoff.device, dtype=torch.float32).reshape(-1, 1)
    if (
        cutoff.shape != rates.shape
        or torch.any(rates <= 0)
        or torch.any(cutoff < 0)
        or torch.any(cutoff > rates / 2.0)
        or not torch.isfinite(cutoff).all()
        or not torch.isfinite(rates).all()
    ):
        raise ValueError(
            "cutoff_hz must be finite and Nyquist-bounded; sample_rate must be finite and positive"
        )

    frequencies = (
        torch.arange(fft_size // 2 + 1, device=cutoff.device, dtype=cutoff.dtype)
        * rates
        / float(fft_size)
    )
    half_width = crossover_width_hz / 2.0
    start = (cutoff - half_width).clamp_min(0.0)
    end = (cutoff + half_width).maximum(start)
    position = ((frequencies - start) / (end - start).clamp_min(1.0)).clamp(0.0, 1.0)
    transition = 0.5 * (1.0 - torch.cos(torch.pi * position))
    weights = torch.where(
        frequencies <= start,
        torch.zeros_like(transition),
        torch.where(frequencies >= end, torch.ones_like(transition), transition),
    )
    return weights[:, None, None, :]


def blend_deployment_features(
    degraded: torch.Tensor,
    predicted: torch.Tensor,
    missing_band_mask: torch.Tensor,
) -> torch.Tensor:
    """Blend model output with degraded features using Rust magnitude/phase semantics."""
    _validate_feature_pair(predicted, degraded)
    mask = _feature_weight(missing_band_mask, predicted[:, 0])
    original_magnitude = torch.pow(10.0, degraded[:, 0].clamp(-200.0, 100.0) / 20.0)
    generated_magnitude = torch.pow(10.0, predicted[:, 0].clamp(-200.0, 100.0) / 20.0)
    magnitude = original_magnitude * (1.0 - mask) + generated_magnitude * mask
    phase = _blend_phase_like_runtime(degraded[:, 1], predicted[:, 1], mask)
    magnitude_db = 20.0 * magnitude.clamp_min(1e-10).log10()
    return torch.stack((magnitude_db, phase), dim=1)


def _blend_phase_like_runtime(
    original: torch.Tensor,
    predicted: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Match Rust's unit-phasor interpolation, including its antipodal fallback."""
    blended_real = torch.cos(original) * (1.0 - weight) + torch.cos(predicted) * weight
    blended_imag = torch.sin(original) * (1.0 - weight) + torch.sin(predicted) * weight
    norm_squared = blended_real.square() + blended_imag.square()
    stable_phasor = norm_squared > torch.finfo(original.dtype).eps
    # Keep the mathematically undefined zero-vector atan2 out of backward,
    # rather than depending on the backend's zero-gradient convention.
    phasor_phase = torch.atan2(
        torch.where(stable_phasor, blended_imag, torch.zeros_like(blended_imag)),
        torch.where(stable_phasor, blended_real, torch.ones_like(blended_real)),
    )

    delta = torch.atan2(torch.sin(predicted - original), torch.cos(predicted - original))
    fallback = original + weight * delta
    fallback = torch.atan2(torch.sin(fallback), torch.cos(fallback))
    intermediate = torch.where(
        stable_phasor,
        phasor_phase,
        fallback,
    )
    return torch.where(
        weight == 0.0,
        original,
        torch.where(weight == 1.0, predicted, intermediate),
    )


def causal_overlap_add(features: torch.Tensor, *, fft_size: int, hop_size: int) -> torch.Tensor:
    """Synthesize `[B, 2, T, F]` with periodic Hann and squared-window OLA."""
    if features.ndim != 4 or features.shape[1] != 2:
        raise ValueError("features must have shape [B, 2, T, F]")
    if features.shape[-1] != fft_size // 2 + 1:
        raise ValueError("feature frequency bins do not match fft_size")
    if hop_size <= 0 or hop_size > fft_size or fft_size % hop_size:
        raise ValueError("hop_size must divide fft_size")
    magnitude = torch.pow(10.0, features[:, 0].clamp(-200.0, 100.0) / 20.0)
    spectrum = torch.polar(magnitude, features[:, 1])
    frames = torch.fft.irfft(spectrum, n=fft_size, dim=-1)
    window = torch.hann_window(
        fft_size,
        periodic=True,
        dtype=features.dtype,
        device=features.device,
    )
    frames = frames * window
    frame_count = features.shape[2]
    output_length = fft_size + max(frame_count - 1, 0) * hop_size
    output = features.new_zeros((features.shape[0], output_length))
    weights = features.new_zeros(output_length)
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        output[:, start : start + fft_size] = (
            output[:, start : start + fft_size] + frames[:, frame_index]
        )
        weights[start : start + fft_size] = weights[start : start + fft_size] + window.square()
    return torch.where(weights > 1e-12, output / weights.clamp_min(1e-12), 0.0)


def deployment_waveform_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    degraded: torch.Tensor,
    missing_band_mask: torch.Tensor,
    *,
    fft_size: int,
    hop_size: int,
    waveform_region: str = "full",
    normalization: str = "none",
    normalization_floor: float = 1e-4,
    relative_floor: float = 0.01,
) -> torch.Tensor:
    """Return L1 for the differentiable crossover, phase blend, and causal OLA proxy.

    Runtime loudness matching, limiting, and the sample-domain gate ramp are stateful
    Rust stages and are intentionally not approximated here. Release quality is
    therefore decided by evaluation through the actual Rust causal stream.
    """
    blended = blend_deployment_features(degraded, predicted, missing_band_mask)
    predicted_waveform = causal_overlap_add(blended, fft_size=fft_size, hop_size=hop_size)
    target_waveform = causal_overlap_add(target, fft_size=fft_size, hop_size=hop_size)
    if waveform_region == "steady_state":
        # Only supervise samples with the full overlap support of a continuous stream.
        # A cropped sequence's single-window Hann tails are artificial boundaries.
        edge = fft_size - hop_size
        if predicted_waveform.shape[-1] <= 2 * edge:
            raise ValueError("steady_state waveform loss needs more causal sequence frames")
        if edge:
            predicted_waveform = predicted_waveform[:, edge:-edge]
            target_waveform = target_waveform[:, edge:-edge]
    elif waveform_region != "full":
        raise ValueError("waveform_region must be 'full' or 'steady_state'")
    if normalization == "none":
        return F.l1_loss(predicted_waveform, target_waveform)
    if normalization != "baseline_error":
        raise ValueError("unsupported waveform normalization")
    errors = (predicted_waveform - target_waveform).abs().mean(dim=-1)
    # A value of one means the same waveform error as leaving this source dry.
    # Detached per-recording scales cannot be gamed by the generator. Relative
    # and absolute floors bound the gradient on near-identical and silent input.
    baseline_waveform = causal_overlap_add(degraded, fft_size=fft_size, hop_size=hop_size)
    if waveform_region == "steady_state" and edge:
        baseline_waveform = baseline_waveform[:, edge:-edge]
    baseline_error = (baseline_waveform - target_waveform).abs().mean(dim=-1)
    target_rms = target_waveform.square().mean(dim=-1).sqrt()
    scale = baseline_error.maximum(target_rms * relative_floor).clamp_min(normalization_floor)
    return (errors / scale.detach()).mean()


class GeneratorLoss(nn.Module):
    """Observable missing/high-band, identity, waveform, and GAN loss terms."""

    def __init__(
        self,
        *,
        high_band_weight: float = 1.0,
        phase_weight: float = 0.2,
        low_band_identity_weight: float = 0.1,
        crossover_weight: float = 0.05,
        waveform_weight: float = 0.1,
        adversarial_weight: float = 0.1,
        feature_matching_weight: float = 0.5,
        phase_relative_floor_db: float = -60.0,
        phase_absolute_floor_db: float = -120.0,
        fft_size: int = 1024,
        hop_size: int = 512,
        waveform_region: str = "full",
        objective_version: int = 1,
        magnitude_scale_db: float = 20.0,
        waveform_normalization_floor: float = 1e-4,
        waveform_relative_floor: float = 0.01,
        high_spectral_weight: float = 1.0,
        high_energy_weight: float = 0.5,
        high_temporal_weight: float = 0.1,
        reconstruction_high_weight: float = 0.5,
        high_relative_floor: float = 0.01,
        magnitude_relative_floor_db: float = -80.0,
        magnitude_absolute_floor_db: float = -120.0,
        reconstruction_high_complex_weight: float = 0.2,
        reconstruction_low_complex_weight: float = 1.0,
        spectral_consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.high_band_weight = high_band_weight
        self.phase_weight = phase_weight
        self.low_band_identity_weight = low_band_identity_weight
        self.crossover_weight = crossover_weight
        self.waveform_weight = waveform_weight
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.phase_relative_floor_db = phase_relative_floor_db
        self.phase_absolute_floor_db = phase_absolute_floor_db
        self.fft_size = fft_size
        self.hop_size = hop_size
        if waveform_region not in {"full", "steady_state"}:
            raise ValueError("waveform_region must be 'full' or 'steady_state'")
        self.waveform_region = waveform_region
        if objective_version not in {1, 2, 3, 4}:
            raise ValueError("objective_version must be 1, 2, 3 or 4")
        for name, value in {
            "magnitude_scale_db": magnitude_scale_db,
            "waveform_normalization_floor": waveform_normalization_floor,
            "waveform_relative_floor": waveform_relative_floor,
            "high_relative_floor": high_relative_floor,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {
            "high_band_weight": high_band_weight,
            "phase_weight": phase_weight,
            "low_band_identity_weight": low_band_identity_weight,
            "crossover_weight": crossover_weight,
            "waveform_weight": waveform_weight,
            "adversarial_weight": adversarial_weight,
            "feature_matching_weight": feature_matching_weight,
            "high_spectral_weight": high_spectral_weight,
            "high_energy_weight": high_energy_weight,
            "high_temporal_weight": high_temporal_weight,
            "reconstruction_high_weight": reconstruction_high_weight,
            "reconstruction_high_complex_weight": reconstruction_high_complex_weight,
            "reconstruction_low_complex_weight": reconstruction_low_complex_weight,
            "spectral_consistency_weight": spectral_consistency_weight,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.objective_version = objective_version
        self.magnitude_scale_db = magnitude_scale_db
        self.waveform_normalization_floor = waveform_normalization_floor
        self.waveform_relative_floor = waveform_relative_floor
        self.high_spectral_weight = high_spectral_weight
        self.high_energy_weight = high_energy_weight
        self.high_temporal_weight = high_temporal_weight
        self.reconstruction_high_weight = reconstruction_high_weight
        self.high_relative_floor = high_relative_floor
        self.reconstruction_high_complex_weight = reconstruction_high_complex_weight
        self.reconstruction_low_complex_weight = reconstruction_low_complex_weight
        self.spectral_consistency_weight = spectral_consistency_weight
        for name, value in {
            "magnitude_relative_floor_db": magnitude_relative_floor_db,
            "magnitude_absolute_floor_db": magnitude_absolute_floor_db,
        }.items():
            if not math.isfinite(value) or value > 0:
                raise ValueError(f"{name} must be finite and non-positive")
        self.magnitude_relative_floor_db = magnitude_relative_floor_db
        self.magnitude_absolute_floor_db = magnitude_absolute_floor_db

    @classmethod
    def from_config(
        cls,
        objective: dict[str, float | str],
        *,
        fft_size: int,
        hop_size: int,
    ) -> GeneratorLoss:
        """Build from the versioned `training.objective` mapping."""
        return cls(
            high_band_weight=float(objective["high_band_magnitude_weight"]),
            phase_weight=float(objective["phase_weight"]),
            low_band_identity_weight=float(objective["low_band_identity_weight"]),
            crossover_weight=float(objective["crossover_continuity_weight"]),
            waveform_weight=float(objective["waveform_weight"]),
            adversarial_weight=float(objective["adversarial_weight"]),
            feature_matching_weight=float(objective["feature_matching_weight"]),
            phase_relative_floor_db=float(objective["phase_relative_floor_db"]),
            phase_absolute_floor_db=float(objective["phase_absolute_floor_db"]),
            fft_size=fft_size,
            hop_size=hop_size,
            waveform_region=str(objective.get("waveform_region", "full")),
            objective_version=int(objective.get("version", 1)),
            magnitude_scale_db=float(objective.get("magnitude_scale_db", 20.0)),
            waveform_normalization_floor=float(objective.get("waveform_normalization_floor", 1e-4)),
            waveform_relative_floor=float(objective.get("waveform_relative_floor", 0.01)),
            high_spectral_weight=float(objective.get("high_band_spectral_weight", 1.0)),
            high_energy_weight=float(objective.get("high_band_energy_weight", 0.5)),
            high_temporal_weight=float(objective.get("high_band_temporal_weight", 0.1)),
            reconstruction_high_weight=float(objective.get("reconstruction_high_weight", 0.5)),
            high_relative_floor=float(objective.get("high_relative_floor", 0.01)),
            magnitude_relative_floor_db=float(objective.get("magnitude_relative_floor_db", -80.0)),
            magnitude_absolute_floor_db=float(objective.get("magnitude_absolute_floor_db", -120.0)),
            reconstruction_high_complex_weight=float(
                objective.get("reconstruction_high_complex_weight", 0.2)
            ),
            reconstruction_low_complex_weight=float(
                objective.get("reconstruction_low_complex_weight", 1.0)
            ),
            spectral_consistency_weight=float(objective.get("spectral_consistency_weight", 0.1)),
        )

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor,
        missing_band_mask: torch.Tensor,
        disc_pred_features: list[list[torch.Tensor]] | None = None,
        disc_target_features: list[list[torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for aligned `[B, 2, T, F]` dB/phase features."""
        _validate_feature_pair(predicted, target)
        _validate_feature_pair(predicted, degraded)
        mask = _feature_weight(missing_band_mask, predicted[:, 0])
        mean = _recording_weighted_mean if self.objective_version >= 2 else _weighted_mean
        high_band_magnitude = mean((predicted[:, 0] - target[:, 0]).abs(), mask)
        if self.objective_version >= 3:
            # The same detached reference floor is used on both spectra. Weak
            # bins must not reward invented hiss merely for beating a -200 dB
            # numerical floor. Linear spectral/energy losses supervise excess.
            floor = (
                (target[:, 0].amax(dim=-1, keepdim=True) + self.magnitude_relative_floor_db)
                .clamp_min(self.magnitude_absolute_floor_db)
                .detach()
            )
            high_band_magnitude = mean(
                (predicted[:, 0].maximum(floor) - target[:, 0].maximum(floor)).abs(), mask
            )

        target_magnitude = torch.pow(10.0, target[:, 0].clamp(-200.0, 100.0) / 20.0)
        peak = target_magnitude.amax(dim=-1, keepdim=True)
        relative_energy = target_magnitude / peak.clamp_min(1e-10)
        relative_floor = 10.0 ** (self.phase_relative_floor_db / 20.0)
        meaningful = (relative_energy >= relative_floor) & (
            target[:, 0] >= self.phase_absolute_floor_db
        )
        phase_weights = mask * relative_energy * meaningful.to(mask.dtype)
        phase_delta = predicted[:, 1] - target[:, 1]
        phase = mean(1.0 - torch.cos(phase_delta), phase_weights)

        low_band_identity = mean((predicted[:, 0] - degraded[:, 0]).abs(), 1.0 - mask)
        residual = predicted[:, 0] - degraded[:, 0]
        residual_gradient = (residual[..., 1:] - residual[..., :-1]).abs()
        transition = 4.0 * mask * (1.0 - mask)
        transition_edges = torch.maximum(transition[..., 1:], transition[..., :-1])
        crossover = mean(residual_gradient, transition_edges)
        waveform = deployment_waveform_loss(
            predicted,
            target,
            degraded,
            missing_band_mask,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            waveform_region=self.waveform_region,
        )
        waveform_relative = (
            deployment_waveform_loss(
                predicted,
                target,
                degraded,
                missing_band_mask,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                waveform_region=self.waveform_region,
                normalization="baseline_error",
                normalization_floor=self.waveform_normalization_floor,
                relative_floor=self.waveform_relative_floor,
            )
            if self.objective_version >= 2
            else waveform
        )

        zero = predicted.new_zeros(())
        adversarial = zero
        feature_matching = zero
        if disc_pred_features is not None and disc_target_features is not None:
            if len(disc_pred_features) != len(disc_target_features):
                raise ValueError("discriminator feature scale counts must match")
            for scale_features in disc_pred_features:
                adversarial = adversarial - scale_features[-1].mean()
            adversarial = adversarial / max(len(disc_pred_features), 1)

            feature_count = 0
            for pred_features, target_features in zip(
                disc_pred_features, disc_target_features, strict=True
            ):
                for pred_feature, target_feature in zip(
                    pred_features[:-1], target_features[:-1], strict=True
                ):
                    feature_matching = feature_matching + F.l1_loss(
                        pred_feature, target_feature.detach()
                    )
                    feature_count += 1
            feature_matching = feature_matching / max(feature_count, 1)

        db_scale = self.magnitude_scale_db if self.objective_version >= 2 else 1.0
        phase_scale = 2.0 if self.objective_version >= 2 else 1.0
        total = (
            self.high_band_weight * high_band_magnitude / db_scale
            + self.phase_weight * phase / phase_scale
            + self.low_band_identity_weight * low_band_identity / db_scale
            + self.crossover_weight * crossover / db_scale
            + self.waveform_weight * waveform_relative
            + self.adversarial_weight * adversarial
            + self.feature_matching_weight * feature_matching
        )
        result = {
            "total": total,
            "high_band_magnitude": high_band_magnitude,
            "phase": phase,
            "low_band_identity": low_band_identity,
            "crossover_continuity": crossover,
            "waveform": waveform,
            "adversarial": adversarial,
            "feature_matching": feature_matching,
        }
        if self.objective_version >= 2:
            result["waveform_relative"] = waveform_relative
        if self.objective_version >= 3:
            inpainting = inpainting_fidelity_terms(
                predicted,
                target,
                degraded,
                mask,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                relative_floor=self.high_relative_floor,
                absolute_floor=self.waveform_normalization_floor,
            )
            result.update(inpainting)
            result["total"] = total + (
                self.high_spectral_weight * inpainting["high_band_spectral_convergence"]
                + self.high_energy_weight * inpainting["high_band_energy"] / db_scale
                + self.high_temporal_weight * inpainting["high_band_temporal"]
                + self.reconstruction_high_weight
                * inpainting["reconstruction_high_spectral_convergence"]
            )
        if self.objective_version == 4:
            consistency = reconstruction_complex_terms(
                predicted,
                target,
                degraded,
                mask,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                relative_floor=self.high_relative_floor,
                absolute_floor=self.waveform_normalization_floor,
            )
            result.update(consistency)
            result["total"] = result["total"] + (
                self.reconstruction_high_complex_weight * consistency["reconstruction_high_complex"]
                + self.reconstruction_low_complex_weight * consistency["reconstruction_low_complex"]
                + self.spectral_consistency_weight * consistency["spectral_consistency"]
            )
        return result


def _relative_high_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
    relative_floor: float,
    absolute_floor: float,
) -> torch.Tensor:
    dimensions = tuple(range(1, predicted.ndim))
    error = ((predicted - target) * mask.sqrt()).flatten(1).norm(dim=1)
    high = (target * mask.sqrt()).flatten(1).norm(dim=1)
    full = reference.square().mean(dim=dimensions).sqrt() * math.sqrt(target[0].numel())
    scale = high.maximum(full * relative_floor).clamp_min(absolute_floor).detach()
    return (error / scale).mean()


def inpainting_fidelity_terms(
    predicted: torch.Tensor,
    target: torch.Tensor,
    degraded: torch.Tensor,
    mask: torch.Tensor,
    *,
    fft_size: int,
    hop_size: int,
    relative_floor: float,
    absolute_floor: float,
) -> dict[str, torch.Tensor]:
    """V3: high-band shape, energy, reference-relative dynamics and reconstruction.

    All scales are per recording and detached. Signed frame differences preserve
    real attacks/releases rather than encouraging a temporally constant output.
    The reconstruction term measures *reanalysed* causal OLA after crossover and
    phase blend. This proxy still excludes Rust's loudness, limiter and gate state.
    """
    if predicted.shape[2] < 2:
        raise ValueError("inpainting objective requires at least two causal frames")
    magnitude = torch.pow(10.0, predicted[:, 0].clamp(-200.0, 100.0) / 20.0)
    reference = torch.pow(10.0, target[:, 0].clamp(-200.0, 100.0) / 20.0)
    spectral = _relative_high_error(
        magnitude, reference, mask, reference, relative_floor, absolute_floor
    )
    denominator = mask.sum(dim=-1).clamp_min(1e-12)
    actual_power = (magnitude.square() * mask).sum(dim=-1) / denominator
    reference_power = (reference.square() * mask).sum(dim=-1) / denominator
    power_floor = (
        (reference.square().mean(dim=-1) * relative_floor**2).clamp_min(absolute_floor**2).detach()
    )
    energy_delta = (
        10.0
        * ((actual_power + power_floor).log10() - (reference_power + power_floor).log10()).abs()
    )
    energy = (energy_delta * (mask.sum(dim=-1) > 0)).mean()
    temporal = _relative_high_error(
        magnitude[:, 1:] - magnitude[:, :-1],
        reference[:, 1:] - reference[:, :-1],
        torch.minimum(mask[:, 1:], mask[:, :-1]),
        reference,
        relative_floor,
        absolute_floor,
    )
    blended = blend_deployment_features(degraded, predicted, mask)
    waves = [
        causal_overlap_add(value, fft_size=fft_size, hop_size=hop_size)
        for value in (blended, target)
    ]
    edge = fft_size - hop_size
    if edge:
        waves = [value[:, edge:-edge] for value in waves]
    if waves[0].shape[-1] < fft_size:
        raise ValueError("inpainting reconstruction requires more fully supported causal frames")
    window = torch.hann_window(
        fft_size, periodic=True, device=predicted.device, dtype=predicted.dtype
    )
    spectra = [
        torch.fft.rfft(value.unfold(-1, fft_size, hop_size) * window).abs() for value in waves
    ]
    reconstructed_mask = mask.mean(dim=1, keepdim=True).expand_as(spectra[0])
    reconstruction = _relative_high_error(
        spectra[0], spectra[1], reconstructed_mask, spectra[1], relative_floor, absolute_floor
    )
    return {
        "high_band_spectral_convergence": spectral,
        "high_band_energy": energy,
        "high_band_temporal": temporal,
        "reconstruction_high_spectral_convergence": reconstruction,
    }


def reconstruction_complex_terms(
    predicted: torch.Tensor,
    target: torch.Tensor,
    degraded: torch.Tensor,
    mask: torch.Tensor,
    *,
    fft_size: int,
    hop_size: int,
    relative_floor: float,
    absolute_floor: float,
) -> dict[str, torch.Tensor]:
    """V4: supervise actual complex reconstruction and retained-band corruption.

    Compare matching fully supported STFT frames; never include artificial cropped
    Hann tails. Consistency follows the predicted complex spectrum, while its
    detached normalization always comes from the clean reference, not prediction.
    This remains a crossover/phase/OLA proxy, excluding online detector and gain.
    """
    blended = blend_deployment_features(degraded, predicted, mask)
    edge = fft_size - hop_size
    if edge == 0:
        first, last = 0, predicted.shape[2]
    else:
        first = edge // hop_size
        last = predicted.shape[2] - first
    if last <= first:
        raise ValueError("complex reconstruction requires fully supported causal frames")
    window = torch.hann_window(
        fft_size, periodic=True, device=predicted.device, dtype=predicted.dtype
    )
    spectra = []
    for features in (blended, target, degraded):
        wave = causal_overlap_add(features, fft_size=fft_size, hop_size=hop_size)
        if edge:
            wave = wave[:, edge:-edge]
        if wave.shape[-1] < fft_size:
            raise ValueError("complex reconstruction requires more causal frames")
        spectra.append(torch.fft.rfft(wave.unfold(-1, fft_size, hop_size) * window))
    amplitude = torch.pow(10.0, blended[:, 0, first:last].clamp(-200.0, 100.0) / 20.0)
    desired = torch.polar(amplitude, blended[:, 1, first:last])
    aligned_mask = mask.expand_as(predicted[:, 0])[:, first:last]
    if desired.shape != spectra[0].shape:
        raise ValueError("complex reconstruction frame alignment differs")

    def relative(
        delta: torch.Tensor, reference: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        error = (delta * weight.sqrt()).flatten(1).norm(dim=1)
        band = (reference * weight.sqrt()).flatten(1).norm(dim=1)
        full = reference.abs().square().mean(dim=(1, 2)).sqrt() * math.sqrt(reference[0].numel())
        scale = band.maximum(full * relative_floor).clamp_min(absolute_floor).detach()
        return (error / scale).mean()

    return {
        "reconstruction_high_complex": relative(spectra[0] - spectra[1], spectra[1], aligned_mask),
        "reconstruction_low_complex": relative(
            spectra[0] - spectra[2], spectra[2], 1.0 - aligned_mask
        ),
        "spectral_consistency": relative(spectra[0] - desired, spectra[1], aligned_mask),
    }


class DiscriminatorLoss(nn.Module):
    """Hinge discriminator loss."""

    def forward(
        self,
        disc_pred_features: list[list[torch.Tensor]],
        disc_target_features: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        if not disc_pred_features or len(disc_pred_features) != len(disc_target_features):
            raise ValueError("Discriminator feature lists must be non-empty and have equal length")

        loss = disc_pred_features[0][0].new_zeros(())
        for pred_features, target_features in zip(
            disc_pred_features, disc_target_features, strict=True
        ):
            loss = loss + F.relu(1.0 - target_features[-1]).mean()
            loss = loss + F.relu(1.0 + pred_features[-1]).mean()
        return loss / len(disc_pred_features)


def _validate_feature_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 2:
        raise ValueError("feature tensors must have matching [B, 2, T, F] shapes")


def _feature_weight(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 4 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if weight.ndim != 3:
        raise ValueError("missing_band_mask must have shape [B, 1, T|1, F] or [B, T|1, F]")
    try:
        return torch.broadcast_to(weight.to(reference), reference.shape)
    except RuntimeError as error:
        raise ValueError("missing_band_mask is not broadcastable to feature shape") from error


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denominator = weight.sum()
    if float(denominator.detach()) == 0.0:
        return value.new_zeros(())
    return (value * weight).sum() / denominator


def _recording_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Normalize each source independently; band width must not alter draw mass."""
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions)
    return (numerator / denominator.clamp_min(1e-12)).mean()
