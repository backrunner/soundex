"""P51 affine feature separation, with an equally suppressed scalar control.

These fit-only logit proxies are not lossless reconstruction or PCM objectives.
SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.restoration_head import predict_budget

RIDGE = 1e-6
PENALTY = 1.0


@torch.no_grad()
def features(model, item):
    """Capture the actual deployed affine-head inputs after real causal warmup.

    Frames 89 onward match direct supervision. Warmup and frames 80..88 do not
    enter statistics; this is distinct from the preceding OLA guard frame.
    """
    captured = []
    handle = model.adapter.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        predict_budget(model, item[0].batch, item[1])
    finally:
        handle.remove()
    if len(captured) != 2 or captured[0].shape[1] != 80:
        raise ValueError("expected separate warmup and tail feature calls")
    latent = torch.cat(captured, dim=1)[:, 89:]
    if latent.shape[1] < 1 or not torch.isfinite(latent).all():
        raise ValueError("finite supervised features required")
    identity = item[0].batch["identity"].bool()
    count = len(identity) // 2
    if identity.tolist() != [False] * count + [True] * count:
        raise ValueError("aligned paired/identity layout required")
    return torch.cat((latent.double(), torch.ones_like(latent[..., :1]).double()), -1)


def grams(z):
    pair, clean = z.chunk(2)
    return {
        role: flat.T @ flat / len(flat)
        for role, values in (("paired", pair), ("identity", clean))
        for flat in (values.reshape(-1, z.shape[-1]),)
    }


def energy(weights, gram):
    return float(((weights @ gram) * weights).sum())


def separate(weights, paired, identity, penalty=PENALTY):
    """Preserve paired logits while penalizing clean logits, using fit Grams only.

    Minimize E_p ||(V'-V)u||^2 + penalty E_i ||V'u||^2
    + RIDGE ||V'-V||_F^2 after diagonal RMS scaling z=D*u.
    The ridge anchors unsupported directions to the original head. Fit/review
    scores and original clean target spectra are not inputs to this solver.
    """
    if penalty < 0 or not all(torch.isfinite(x).all() for x in (weights, paired, identity)):
        raise ValueError("finite inputs and nonnegative penalty required")
    scale = ((paired.diag() + identity.diag()) / 2).clamp_min(1e-12).sqrt()
    gp = paired / scale[:, None] / scale[None, :]
    gi = identity / scale[:, None] / scale[None, :]
    eye = torch.eye(len(scale), dtype=weights.dtype, device=weights.device)
    original = weights * scale
    rhs = original @ (gp + RIDGE * eye)
    system = gp + penalty * gi + RIDGE * eye
    projected = torch.linalg.solve(system, rhs.T).T / scale
    # Deployment coefficients are float32. Match the actually saved coefficients,
    # not a higher-precision solution that is never used for inference.
    projected = projected.float().double()
    before, after = energy(weights, identity), energy(projected, identity)
    if before <= 0 or after < 0 or not torch.isfinite(projected).all():
        raise ValueError("positive identity logit energy required")
    scalar = (after / before) ** 0.5
    scaled = (weights * scalar).float().double()
    residual = ((projected * scale) @ system - rhs).norm() / rhs.norm().clamp_min(1e-30)
    report = {
        "penalty": penalty,
        "ridge": RIDGE,
        "scalar": scalar,
        "relative_normal_equation_residual_after_float32": float(residual),
        "identity_logit_energy_before": before,
        "identity_logit_energy_projected": after,
        "identity_logit_energy_scalar": energy(scaled, identity),
        "paired_logit_distortion_projected": energy(projected - weights, paired),
        "paired_logit_distortion_scalar": energy(scaled - weights, paired),
        "paired_logit_energy_before": energy(weights, paired),
        "paired_logit_energy_projected": energy(projected, paired),
        "paired_logit_energy_scalar": energy(scaled, paired),
        "paired_plus_identity_gram_trace": float((paired + identity).trace()),
        "condition_number": float(torch.linalg.cond(system)),
    }
    return projected, scaled, report


def coefficients(model):
    return torch.cat(
        (model.adapter.weight.detach().double(), model.adapter.bias.detach().double()[:, None]), 1
    )


@torch.no_grad()
def install(model, weights):
    model.adapter.weight.copy_(weights[:, :-1])
    model.adapter.bias.copy_(weights[:, -1])


def paired_geometry(z):
    pair, clean = z[..., :-1].chunk(2)
    # Aligned frame cosines and contrast are descriptive, not a classifier ceiling.
    cosine = torch.nn.functional.cosine_similarity(pair, clean, dim=-1)
    difference = (pair - clean).square().mean((1, 2))
    return {
        "cosine_mean_per_clip": cosine.mean(-1).tolist(),
        "contrast_mean_square_per_clip": difference.tolist(),
        "identity_feature_mean_square_per_clip": clean.square().mean((1, 2)).tolist(),
    }


@torch.no_grad()
def response_bands(model, z, rates):
    """Input-unit-normalized adapter power, explicitly NOT spectral/PCM dBFS.

    Includes tanh, physical support and imaginary endpoint masks; excludes the
    input-dependent scale/power units, base decoder, phase rotation and OLA.
    Original-reference PCM and spectral losses are reviewed separately.
    """
    logits = model.adapter(z[..., :-1].to(model.adapter.weight.dtype))
    adjustment = 2 * torch.tanh(model.residual_slope * logits)
    condition = rates.reshape(-1, 1, 1) / 48000
    mask = model.support_mask(condition)
    real = adjustment[..., :129] * mask
    imag = adjustment[..., 129:] * mask * model.interior
    power = real.double().square() + imag.double().square()
    hz = torch.arange(129)[None, :] * rates.reshape(-1, 1) / 256
    result = {"affine_logit_mean_square": logits.double().square().mean((1, 2)).tolist()}
    for low, high in ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001)):
        band = ((hz >= low) & (hz < high))[:, None, :]
        result[f"normalized_power_{low}_{high}"] = (
            (power * band).sum(-1).mean(-1) / band.sum(-1).squeeze(1).clamp_min(1)
        ).tolist()
    return result
