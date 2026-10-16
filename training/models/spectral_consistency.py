"""P16 training-only reference-aware complex STFT consistency auxiliary risk."""

import torch

from models.complex_synthesis import analyze, complex_features, synthesize
from models.envelope_objective import BANDS
from models.factorized_envelope import ABSOLUTE_FLOOR, RELATIVE_FLOOR
from models.fit_diagnostic import BURN
from models.linear_objective import SMOOTHING
from models.losses import blend_deployment_features
from models.rendered_objective import amplitude_features


def reference_features(batch, *, phase_reference="input"):
    """Clean target with explicit phase policy; historical default preserves input phase.

    P20 uses clean phase so valid phase restoration is not penalized. Identity uses
    the actual input. This reference never enters inference. Warmup and blending
    follow the same path as predictions.
    """
    if phase_reference not in ("input", "clean"):
        raise ValueError("unknown phase reference")
    amplitude = torch.pow(10.0, batch["y"][:, 0] / 20)
    original = torch.pow(10.0, batch["x"][:, 0] / 20)
    amplitude = torch.where(batch["identity"][:, None, None].bool(), original, amplitude)
    reference = amplitude_features(amplitude, batch["x"])
    if phase_reference == "clean":
        phase = torch.where(
            batch["identity"][:, None, None].bool(), batch["x"][:, 1], batch["y"][:, 1]
        )
        reference = torch.stack(
            (reference[:, 0], torch.cat((batch["x"][:, 1, :BURN], phase[:, BURN:]), dim=1)), dim=1
        )
    return reference


def consistency_loss(predicted, batch, burn=89, *, phase_reference="input", normalization="band"):
    """Penalize (P-I)(Zpred-Zref), not changes over time or distance to silence.

    P is analysis after the existing prelimit OLA. Subtracting the reference
    inconsistency avoids treating valid target transients as an error to smooth.
    The original DSP-rendered objective remains the main risk and evaluation score.
    """
    if normalization not in ("band", "global_band"):
        raise ValueError("unknown consistency normalization")
    effective = batch["mask"] * batch["active"][:, None, :, None]
    blended = blend_deployment_features(batch["x"], predicted, effective)
    with torch.no_grad():
        target = blend_deployment_features(
            batch["x"], reference_features(batch, phase_reference=phase_reference), effective
        )
        reference = complex_features(target)
    delta = complex_features(blended) - reference
    projected = analyze(synthesize(delta))
    error = (projected - delta[:, : projected.shape[1]])[:, burn:]
    magnitude = reference[:, burn : projected.shape[1]].abs()
    floor = (magnitude.square().mean((1, 2)).sqrt() * RELATIVE_FLOOR).clamp_min(ABSOLUTE_FLOOR)
    frequency = torch.arange(129, device=predicted.device)[None, :] * batch["rate"] / 256
    per_example = predicted.new_zeros(len(predicted))
    for low, high in BANDS:
        mask = ((frequency >= low) & (frequency < high))[:, None, :]
        count = mask.sum((1, 2)).clamp_min(1) * error.shape[1]
        denominator = (magnitude.square() * mask).sum((1, 2)) / count + floor.square()
        if normalization == "global_band":
            denominator = denominator + magnitude.square().mean((1, 2))
        power = (error.abs().square() * mask).sum((1, 2)) / count
        weight = (0.25 if low == 0 else 1.0) / 4.25
        per_example = per_example + weight * (
            (power / denominator + SMOOTHING**2).sqrt() - SMOOTHING
        )
    identity = batch["identity"].bool()
    if not identity.any() or identity.all():
        raise ValueError("both paired and identity examples required")
    return 0.5 * (per_example[~identity].mean() + per_example[identity].mean())
