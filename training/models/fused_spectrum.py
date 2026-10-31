"""P67 original-reference MR after the diagnostic native Model-gain DSP.

No DSP target, reference clipping, future model context or inference-time loss.
Training examples are independent mono channel views, not physical stereo pairs.
SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.causal_objective import gate_envelope, limit
from models.complex_synthesis import complex_features, synthesize
from models.direct_spectrum_objective import direct_spectrum_loss
from models.losses import blend_deployment_features
from models.multires_spectrum import multires_spectrum_loss, spectral_loss

VERSION = "soundex-fused-spectrum-p67-v1"
WARMUP = 80
BURN = 89


def prepare_fusion(batch):
    """Input-only DSP decisions; keep detector condition/model history separate.

    The inherited detector condition is inactive for 64 hops. The native model
    replay suppresses enhancement for 80 hops, without suppressing model history.
    Gate construction uses host double internally; measured FP32 parity bounds
    still apply. Do not use this independent-mono gate for physical stereo.
    """
    active = batch["active"].detach().bool().clone()
    active[:, :WARMUP] = False
    return {
        "weights": batch["mask"].detach() * active[:, None, :, None],
        "gate": gate_envelope(active).to(batch["x"]),
        "dry": batch["dry"].detach(),
    }


def fused_stages(prediction, observed, plan):
    weights = plan["weights"]
    fused = blend_deployment_features(observed, prediction, weights)
    # Rust copies retained bins without a dB -> linear -> dB round trip.
    fused = torch.where(weights == 0, observed, fused)
    prelimit = synthesize(complex_features(fused))
    wet = limit(prelimit, "smooth-knee")
    gate = plan["gate"]
    final = (limit(plan["dry"], "smooth-knee") * (1 - gate) + wet * gate).clamp(-0.95, 0.95)
    return {"features": fused, "prelimit": prelimit, "wet": wet, "final": final}


def fused_spectrum_loss(prediction, batch, plan=None):
    if prediction.shape != batch["x"].shape or prediction.shape != batch["y"].shape:
        raise ValueError("paired feature geometry mismatch")
    plan = prepare_fusion(batch) if plan is None else plan
    wave = fused_stages(prediction, batch["x"], plan)["final"]
    target = batch["clean_wave"].detach()
    if wave.shape != target.shape:
        raise ValueError("native and original-reference timeline mismatch")
    region = slice(BURN * 128, wave.shape[-1] - 128)
    return spectral_loss(wave[:, region], target[:, region], batch["rate"])


def balanced_risk(per_example, identity):
    identity = identity.bool()
    if not identity.any() or identity.all():
        raise ValueError("paired and identity roles required")
    return 0.5 * (per_example[~identity].mean() + per_example[identity].mean())


def phase_objective(prediction, batch, *, include_fused, plan=None):
    """Preserve P63/P65 direct+raw terms; add one fixed, same-scale MR auxiliary.

    No waveform distance, rescaled reference, auxiliary-budget teacher, dynamic
    weighting or hidden parameter-step scaling. Different totals are not scores
    to compare; all endpoints are reviewed under both fixed objectives.
    """
    direct = direct_spectrum_loss(prediction, batch)["per_example"]
    raw = multires_spectrum_loss(prediction, batch)["per_example"]
    risk = direct.log1p() + raw.log1p()
    if include_fused:
        risk = risk + fused_spectrum_loss(prediction, batch, plan)["per_example"].log1p()
    return balanced_risk(risk, batch["identity"])
