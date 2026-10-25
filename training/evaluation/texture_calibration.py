"""P59 critic diagnostics, including energy-preserving false-detail controls."""

import math
from collections import defaultdict

import torch

from evaluation.longform_review import aliases
from models.spectral_texture import masked_mean

BANDS = ((8000, 12000), (12000, 16000), (16000, 24001))
CONTROLS = (
    "prediction",
    "observed",
    "energy_matched",
    "phase_scrambled",
    "time_shifted",
    "roundtrip",
)


def isolated_draws(source, rows, universe, reserved):
    """Filter entire draws by connected hash/work aliases before any scoring."""
    blocked = set().union(*(aliases(r) for r in reserved))
    previous = -1
    while len(blocked) != previous:
        previous = len(blocked)
        for row in universe:
            keys = aliases(row)
            if keys & blocked:
                blocked.update(keys)
    return [
        i
        for i, draw in enumerate(source["draws"])
        if all(not aliases(rows[source["specs"][j]["row_id"]]) & blocked for j in draw)
    ]


def controls(prediction, reference, observed, rates, seed):
    """Oracle diagnostics only; never labels or deployable restoration outputs.

    Whole-excerpt rectangular FFT bands introduce circular boundary semantics.
    The phase/time controls preserve reference FFT magnitudes and low frequencies.
    Energy matching uses the reference and cannot count as a learned improvement.
    """
    if (
        prediction.shape != reference.shape
        or observed.shape != reference.shape
        or reference.ndim != 2
    ):
        raise ValueError("aligned physical channel views required")
    rates = rates.reshape(-1)
    if len(rates) != len(reference) or not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical rates required")
    length = reference.shape[-1]
    target = torch.fft.rfft(reference.double())
    estimated = torch.fft.rfft(prediction.double())
    bins = target.shape[-1]
    hz = torch.arange(bins, device=reference.device)[None] * rates[:, None] / length
    sides = torch.full_like(hz, 2, dtype=torch.float64)
    sides[:, 0] = 1
    if length % 2 == 0:
        sides[:, -1] = 1
    matched = estimated.clone()
    for low, high in BANDS:
        mask = (hz >= low) & (hz < high)
        a = (target.abs().square() * sides * mask).sum(-1, keepdim=True)
        b = (estimated.abs().square() * sides * mask).sum(-1, keepdim=True)
        # Exactly absent predicted bands stay absent; record any residual mismatch.
        scale = (a / b.clamp_min(1e-30)).sqrt()
        matched = torch.where(mask, estimated * scale, matched)
    rng = torch.Generator(device=reference.device).manual_seed(seed)
    phase = torch.rand(
        target.shape, generator=rng, device=reference.device, dtype=torch.float64
    ) * (2 * torch.pi)
    high = hz >= 8000
    high[:, 0] = False
    if length % 2 == 0:
        high[:, -1] = False  # Nyquist is real, not an arbitrary phase angle.
    scrambled = torch.where(high, torch.polar(target.abs(), phase), target)
    shift = length // 4
    rotation = torch.exp(
        -2j
        * torch.pi
        * torch.arange(bins, device=reference.device, dtype=torch.float64)
        * shift
        / length
    )
    shifted = torch.where(high, target * rotation, target)

    def wave(spectrum):
        return torch.fft.irfft(spectrum, n=length).to(reference.dtype)

    return {
        "prediction": prediction,
        "observed": observed,
        "energy_matched": wave(matched),
        "phase_scrambled": wave(scrambled),
        "time_shifted": wave(shifted),
        "roundtrip": wave(target),
    }


def pcm_diagnostics(wave, reference, rates):
    error = wave.double() - reference.double()
    length = wave.shape[-1]
    a, b = torch.fft.rfft(wave.double()) / length, torch.fft.rfft(reference.double()) / length
    hz = torch.arange(a.shape[-1], device=wave.device)[None] * rates.reshape(-1, 1) / length
    sides = torch.full_like(hz, 2, dtype=torch.float64)
    sides[:, 0] = 1
    if length % 2 == 0:
        sides[:, -1] = 1
    result = {"pcm_error_rms": error.square().mean(-1).sqrt()}
    for low, high in ((0, 8000), *BANDS):
        mask = (hz >= low) & (hz < high)
        for label, spec in (("error", a - b), ("reference", b), ("candidate", a)):
            power = (spec.abs().square() * sides * mask).sum(-1)
            result[f"band{low}_{label}_rms"] = power.sqrt()
    return result


@torch.no_grad()
def measure(critic, wave, reference, observed, rates):
    real = critic.features(reference, observed, rates)
    fake = critic.features(wave, observed, rates)
    real_logit = torch.stack([masked_mean(r[1], r[2]) for r in real]).mean(0)
    fake_logit = torch.stack([masked_mean(f[1], f[2]) for f in fake]).mean(0)
    feature = critic.reference_loss(wave, reference, observed, rates)
    return {
        "real_logit": real_logit,
        "candidate_logit": fake_logit,
        "margin": real_logit - fake_logit,
        "feature_loss": feature,
        "balanced_logistic_loss": torch.nn.functional.softplus(-real_logit)
        + torch.nn.functional.softplus(fake_logit),
        **pcm_diagnostics(wave, reference, rates),
    }


def aggregate(records):
    """Source-equal means within corpus; no codec/crop pseudoreplication."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        key = "/".join(row[k] for k in ("generator", "critic", "role", "corpus", "control"))
        groups[key][row["source"]].append(row["scores"])
    result = {}
    for key, sources in sorted(groups.items()):
        means = {
            s: {k: sum(x[k] for x in rows) / len(rows) for k in rows[0]}
            for s, rows in sorted(sources.items())
        }
        result[key] = {
            "sources": len(sources),
            "cases": sum(map(len, sources.values())),
            "source_mean": {
                k: sum(v[k] for v in means.values()) / len(means)
                for k in next(iter(means.values()))
            },
            "per_source": means,
            "positive_margin_sources": sum(x["margin"] > 0 for x in means.values()),
        }
    return result


def assess(groups, generators):
    """Predeclared prerequisite, not an audio-quality or convergence gate.

    Every corpus/seed must distinguish all four diagnostic conditions on
    calibration, with a positive mean margin and a source majority beyond the
    roundtrip numerical control. Prediction logistic loss must improve over
    initialization and the balanced constant classifier. Preserve all failures.
    """
    results = {}
    for generator in generators:
        for corpus in ("music_library", "speech_library"):
            prefix = f"{generator}/trained/calibration/{corpus}/"
            rounding = groups[prefix + "roundtrip"]["per_source"]
            cases = {}
            for name in ("prediction", "energy_matched", "phase_scrambled", "time_shifted"):
                g = groups[prefix + name]
                if g["per_source"].keys() != rounding.keys():
                    raise ValueError("unmatched calibration sources")
                differences = [
                    v["margin"] - abs(rounding[s]["margin"]) for s, v in g["per_source"].items()
                ]
                wins = sum(v > 0 for v in differences)
                cases[name] = {
                    "mean_margin_over_roundtrip": sum(differences) / len(differences),
                    "sources_above_roundtrip": wins,
                    "sources": len(differences),
                    "passes": sum(differences) > 0 and wins > len(differences) / 2,
                }
            trained = groups[prefix + "prediction"]
            initial = groups[f"{generator}/initial/calibration/{corpus}/prediction"]
            learned = trained["source_mean"]["balanced_logistic_loss"] < min(
                initial["source_mean"]["balanced_logistic_loss"], 2 * math.log(2)
            )
            results[generator + "/" + corpus] = {
                "controls": cases,
                "prediction_logistic_loss_improved": learned,
                "passes": learned and all(v["passes"] for v in cases.values()),
            }
    return {
        "groups": results,
        "prerequisite_passes": all(v["passes"] for v in results.values()),
        "qualified_audio": False,
    }
