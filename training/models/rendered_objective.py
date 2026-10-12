"""P10: supervise the same input-phase amplitude path through DSP and reanalysis.

CPU FFT/render is autograd-connected to MPS amplitude predictions. Identity labels
are training-only; identity preservation references the input's existing DSP path,
while original-clean restoration metrics are always reported separately.
"""

import torch

from models.causal_objective import render
from models.factorized_envelope import BURN
from models.linear_objective import SMOOTHING, linear_objective


def amplitude_features(amplitude, original, burn=BURN):
    features = torch.stack((20 * amplitude.clamp_min(1e-10).log10(), original[:, 1]), dim=1)
    return torch.cat((original[:, :, :burn], features[:, :, burn:]), dim=2)


def render_amplitude(amplitude, batch, burn=BURN):
    return render(
        amplitude_features(amplitude, batch["x"], burn),
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )[0]


def reanalyze(wave):
    # Native output already includes synthesis delay; do not add another pad.
    window = torch.hann_window(256, device=wave.device, dtype=wave.dtype)
    return torch.fft.rfft(wave[:, :-128].unfold(-1, 256, 128) * window).abs().clamp_min(1e-10)


def waveform_norm(prediction, reference):
    return (
        (prediction - reference).square().mean(1) / (reference.square().mean(1) + 1e-14)
        + SMOOTHING**2
    ).sqrt() - SMOOTHING


class RenderedObjective:
    uses_batch = True

    def __init__(self, identity_weight=1.0):
        if identity_weight not in (1.0, 7.0):
            raise ValueError("registered identity weights are 1 or 7")
        self.identity_weight = identity_weight

    def __call__(self, prediction, target, batch, burn=BURN):
        if "identity" not in batch:
            raise ValueError("explicit training identity labels required")
        raw = linear_objective(prediction, target, batch["rate"], burn=burn)
        # Explicit transfers retain gradients, avoiding backend-specific FFT math.
        cpu = {
            k: batch[k].to("cpu")
            for k in ("x", "mask", "active", "gate", "dry", "clean_wave", "rate", "identity")
        }
        p, t = prediction.to("cpu"), target.to("cpu")
        output = render_amplitude(p, cpu, burn)
        with torch.no_grad():
            original = torch.pow(10.0, cpu["x"][:, 0] / 20)
            baseline = render_amplitude(original, cpu, burn)
            baseline_amplitude = reanalyze(baseline)
        amplitude = reanalyze(output)
        start = (burn + 9) * 128
        stop = output.shape[-1] - 128
        if stop <= start or amplitude.shape[1] <= burn + 10:
            raise ValueError("not enough fully supported scoring frames")
        wave_restore = waveform_norm(output[:, start:stop], cpu["clean_wave"][:, start:stop])
        wave_preserve = waveform_norm(output[:, start:stop], baseline[:, start:stop])
        reconstruction, preservation = [], []
        for i in range(len(p)):
            rate = cpu["rate"][i : i + 1]
            reconstruction.append(
                linear_objective(
                    amplitude[i : i + 1], t[i : i + 1, : amplitude.shape[1]], rate, burn=burn + 9
                )["total"]
            )
            preservation.append(
                linear_objective(
                    amplitude[i : i + 1], baseline_amplitude[i : i + 1], rate, burn=burn + 9
                )["total"]
            )
        reconstruction, preservation = torch.stack(reconstruction), torch.stack(preservation)
        restore = reconstruction + 0.25 * wave_restore
        preserve = preservation + 0.25 * wave_preserve
        identity = cpu["identity"].bool()
        # Expected weighted sample count, not batch-dependent renormalization:
        # fixed draws contain 1/8 identity, so weight 7 balances the two risks.
        normalizer = 0.875 + 0.125 * self.identity_weight
        total = torch.where(identity, self.identity_weight * preserve, restore).mean() / normalizer
        return {
            **raw,
            "raw_total": raw["total"],
            "total": total.to(prediction.device),
            "rendered_reconstruction": reconstruction.mean().to(prediction.device),
            "rendered_waveform": wave_restore.mean().to(prediction.device),
            "dsp_preservation": preserve.mean().to(prediction.device),
        }
