"""Reference-assisted HF component swaps for diagnosis only, never inference."""

import torch

from models.complex_synthesis import complex_features
from models.envelope_objective import BANDS
from models.fit_diagnostic import BURN

VARIANTS = ("candidate", "reference_phase", "reference_amplitude", "reference_high")


def high_frequency_swaps(prediction, batch):
    """Replace only >=4 kHz after real model warmup; keep all lower bins unchanged.

    The hard 4 kHz boundary is a fixed measurement intervention, not a runtime
    crossover proposal. References never enter the model or its history/state.
    """
    if prediction.shape != batch["x"].shape or prediction.shape != batch["y"].shape:
        raise ValueError("aligned input/prediction/reference required")
    if prediction.shape[2] <= BURN + 10:
        raise ValueError("insufficient supervised frames")
    hz = torch.arange(129, device=prediction.device)[None, :] * batch["rate"] / 256
    mask = (hz >= 4000)[:, None, :].expand_as(prediction[:, 0]).clone()
    mask[:, :BURN] = False
    reference = torch.where(batch["identity"][:, None, None, None].bool(), batch["x"], batch["y"])
    result = {"candidate": prediction}
    for name, components in (
        ("reference_phase", (1,)),
        ("reference_amplitude", (0,)),
        ("reference_high", (0, 1)),
    ):
        features = prediction.clone()
        for component in components:
            features[:, component] = torch.where(
                mask, reference[:, component], prediction[:, component]
            )
        result[name] = features
    return result


def complex_error_parts(features, target, rate, burn=89):
    """Exact amplitude + amplitude-weighted circular-phase complex-error split.

    Values are unnormalized FFT coefficient powers, NOT PCM dBFS. Empty/zero
    reference denominators have null relative errors, without fitted thresholds.
    """
    p, t = complex_features(features)[:, burn:], complex_features(target)[:, burn:]
    pm, tm = p.abs(), t.abs()
    amplitude = (pm - tm).square()
    # Unit-phasor distance is stable at +/-pi and nonnegative at equal phases.
    dp = features[:, 1, burn:] - target[:, 1, burn:]
    phase = pm * tm * ((dp.cos() - 1).square() + dp.sin().square())
    total = (p - t).abs().square()
    hz = torch.arange(129, device=p.device)[None, :] * rate / 256
    result = []
    for i in range(len(p)):
        bands = {}
        for low, high in (*BANDS, (4000, 24001)):
            mask = (hz[i] >= low) & (hz[i] < high)
            ref = float(tm[i, :, mask].square().mean())
            values = {
                name: float(v[i, :, mask].mean())
                for name, v in (
                    ("amplitude_power", amplitude),
                    ("phase_power", phase),
                    ("complex_power", total),
                )
            }
            bands[f"{low}_{high}"] = {
                **values,
                "reference_power": ref,
                "complex_relative_power": values["complex_power"] / ref if ref > 0 else None,
                "decomposition_abs_error": abs(
                    values["complex_power"] - values["amplitude_power"] - values["phase_power"]
                ),
            }
        result.append(bands)
    return result
