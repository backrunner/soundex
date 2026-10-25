"""P61 decomposition of frozen critic objectives and paired PCM gradients."""

import torch
from torch.nn import functional as F

from evaluation.texture_calibration import controls
from models.context_texture import shifted_high
from models.multires_spectrum import spectral_loss
from models.spectral_texture import masked_mean


def task_losses(critic, predicted, reference, observed, rates, seed, step):
    """Per-example real-plus-fake tasks and the exact P60 mixture weights."""
    predicted, reference, observed = (v.detach() for v in (predicted, reference, observed))
    variants = controls(predicted, reference, observed, rates, seed)
    candidates = (
        predicted,
        variants["phase_scrambled"],
        shifted_high(reference, rates, (1 / 8, 1 / 6, 1 / 3)[step % 3]),
    )
    weights = reference.new_tensor((0.5, 0.25, 0.25)).expand(len(reference), -1).clone()
    rounding = (variants["roundtrip"].double() - reference.double()).square().mean(-1)
    for index in (1, 2):
        weights[:, index] *= (candidates[index].double() - reference.double()).square().mean(
            -1
        ) > rounding
    weights /= weights.sum(-1, keepdim=True)
    real = critic.features(reference, observed, rates)
    positive = torch.stack([masked_mean(F.softplus(-r[1]), r[2]) for r in real]).mean(0)
    values = []
    for candidate in candidates:
        features = critic.features(candidate, observed, rates)
        values.append(
            positive + torch.stack([masked_mean(F.softplus(f[1]), f[2]) for f in features]).mean(0)
        )
    return torch.stack(values, dim=1), weights


def flattened_gradient(value, parameters):
    gradients = torch.autograd.grad(value, parameters, retain_graph=True)
    result = torch.cat([g.detach().flatten() for g in gradients])
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite diagnostic gradient")
    return result


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 0 else None


def copied_adam(parameters, gradient, checkpoint, *, persistent):
    """Actual float32 Adam displacement on a copy, with optional saved moments."""
    parameter = torch.nn.Parameter(parameters.clone())
    optimizer = torch.optim.Adam([parameter], lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    if persistent:
        saved = checkpoint["optimizer"]
        ids = saved["param_groups"][0]["params"]
        states = [saved["state"][i] for i in ids]
        steps = [float(state["step"]) for state in states]
        if len(set(steps)) != 1:
            raise ValueError("unequal critic optimizer clocks")
        optimizer.state[parameter] = {
            "step": states[0]["step"].clone(),
            "exp_avg": torch.cat([s["exp_avg"].flatten() for s in states]).clone(),
            "exp_avg_sq": torch.cat([s["exp_avg_sq"].flatten() for s in states]).clone(),
        }
    parameter.grad = gradient.clone()
    torch.nn.utils.clip_grad_norm_([parameter], 5, error_if_nonfinite=True)
    optimizer.step()
    return (parameter.detach() - parameters).double()


def pcm_feature_probe(critic, predicted, reference, observed, rates):
    """Raw PCM gradient geometry only, not a realizable generator update.

    Use the existing MR objective independently of discriminator classification.
    A 1%-of-current-error RMS step is fixed for every sample/critic. It probes
    the local direction; it is neither a generator LR nor an acceptance rule.
    """
    wave = predicted.detach().clone().requires_grad_()
    feature = critic.reference_loss(wave, reference, observed, rates)
    mr = spectral_loss(wave, reference, rates)["per_example"]
    (fg,) = torch.autograd.grad(feature.sum(), wave, retain_graph=True)
    (mg,) = torch.autograd.grad(mr.sum(), wave)
    fg, mg = fg.detach(), mg.detach()
    error_rms = (wave.detach() - reference).double().square().mean(-1).sqrt()
    norm = fg.double().square().mean(-1).sqrt()
    scale = torch.where(norm > 0, 0.01 * error_rms / norm.clamp_min(1e-30), 0)
    candidate = predicted + (-fg.double() * scale[:, None]).to(predicted.dtype)
    with torch.no_grad():
        after = spectral_loss(candidate, reference, rates)["per_example"]
        feature_after = critic.reference_loss(candidate, reference, observed, rates)
    rows = []
    for i in range(len(wave)):
        rows.append(
            {
                "feature_mr_gradient_cosine": cosine(fg[i], mg[i]),
                "feature_gradient_norm": float(fg[i].double().norm()),
                "mr_before": float(mr[i].detach()),
                "mr_after": float(after[i]),
                "feature_before": float(feature[i].detach()),
                "feature_after": float(feature_after[i]),
                "pcm_error_rms_before": float(error_rms[i]),
                "pcm_error_rms_after": float(
                    (candidate[i].double() - reference[i].double()).square().mean().sqrt()
                ),
                "step_rms": float(
                    (candidate[i].double() - predicted[i].double()).square().mean().sqrt()
                ),
            }
        )
    return rows
