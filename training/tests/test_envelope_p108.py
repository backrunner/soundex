import numpy as np
import pytest
import torch

from evaluation import envelope_fill as fill
from evaluation.missing_band import score
from models.envelope_predictor import (
    EnvelopePredictor,
    envelope_loss,
    features,
    stft_power,
    targets,
)

RATE = 44100


def lowpass(x, hz):
    spec = np.fft.rfft(x, axis=0)
    spec[np.fft.rfftfreq(len(x), 1 / RATE) >= hz] = 0
    return np.fft.irfft(spec, len(x), axis=0)


@pytest.fixture
def pair():
    rng = np.random.default_rng(1)
    n = RATE * 2
    env = 0.4 + 0.6 * np.sin(2 * np.pi * 2 * np.arange(n) / RATE) ** 2
    ref = 0.05 * rng.standard_normal((n, 2)) * env[:, None]
    return ref, lowpass(ref, 15000)


@pytest.mark.parametrize("arm", ("band", "bin"))
def test_loss_zero_at_reference_and_finite_on_silence(arm, pair):
    ref, _ = pair
    p, _ = stft_power(ref[:, 0])
    t, active = targets(p, RATE, arm)
    _, lref = features(p, RATE)
    target = torch.tensor(t[None])
    pred = (target - torch.tensor(lref)[None, :, None]).clone().requires_grad_(True)
    total, level, change = envelope_loss(
        pred, target, torch.tensor(active[None]), torch.tensor(lref[None]), warmup=8
    )
    assert float(total) == 0 and float(level) == 0 and float(change) == 0
    silent = torch.zeros_like(pred, requires_grad=True)
    total, _, _ = envelope_loss(
        silent, target, torch.zeros_like(torch.tensor(active[None])), torch.tensor(lref[None]), 8
    )
    total.backward()
    assert float(total) == 0 and torch.isfinite(silent.grad).all()


def test_loss_gradcheck_and_warmup_excluded():
    torch.manual_seed(0)
    pred = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(1, 6, 3, dtype=torch.float64)
    ref = torch.zeros(1, 6, dtype=torch.float64)
    active = torch.ones(1, 6, 3, dtype=torch.bool)
    torch.autograd.gradcheck(lambda p: envelope_loss(p + 0.37, target, active, ref, 2)[0], (pred,))
    # Changing only warmup frames must not change the loss.
    other = pred.detach().clone()
    other[:, :2] += 5
    a = envelope_loss(pred.detach(), target, active, ref, 2)[0]
    b = envelope_loss(other, target, active, ref, 2)[0]
    # Only the first supervised change pair can see frame 1; exclude it explicitly.
    active2 = active.clone()
    active2[:, 2] = False
    assert float(envelope_loss(pred.detach(), target, active2, ref, 2)[1]) == pytest.approx(
        float(envelope_loss(other, target, active2, ref, 2)[1])
    )
    assert np.isfinite(float(a)) and np.isfinite(float(b))


def test_loss_is_gain_equivariant():
    target = torch.zeros(1, 10, 4)
    active = torch.ones(1, 10, 4, dtype=torch.bool)
    ref = torch.zeros(1, 10)
    level = envelope_loss(torch.full((1, 10, 4), -0.6), target, active, ref, 0)[1]
    assert float(level) == pytest.approx(6.0)


def test_detector_tracks_cutoff_and_resets_on_silence(pair):
    _, dry = pair
    _, spec = stft_power(dry[:, 0])
    tail = len(spec) * 128 - len(dry)
    frames = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.zeros(128), dry[:, 0], np.zeros(tail)]), 256
    )[::128]
    bw, needs = fill.detector(spec, frames, RATE)
    assert 14000 < np.median(bw[20:]) < 16500
    assert needs[20:].all()
    silent = np.zeros((len(spec), 256))
    _, needs = fill.detector(np.zeros_like(spec), silent, RATE)
    assert not needs.any()


def test_wola_roundtrip_is_exact(pair):
    ref, _ = pair
    _, spec = stft_power(ref[:, 0])
    assert np.allclose(fill.wola(spec, len(ref)), ref[:, 0], atol=1e-12)


def test_full_band_input_is_bypassed(pair):
    ref, _ = pair
    out, info = fill.render(ref, RATE, "oracle_band", reference=ref)
    assert all(i["active_frames"] == 0 for i in info)
    assert np.allclose(out, ref, atol=1e-12)


def test_reference_envelope_fill_restores_level(pair):
    ref, dry = pair
    out, _ = fill.render(dry, RATE, "oracle_band", reference=ref)
    before, after = score(dry, ref, dry, RATE), score(out, ref, dry, RATE)
    assert before["missing_erb_bias_db"] < -60
    assert abs(after["missing_erb_bias_db"]) < 3
    assert after["missing_erb_rmse_db"] < 6
    assert after["preserved_complex_rel"] < before["preserved_complex_rel"] + 0.01


@pytest.mark.parametrize("arm", ("band", "bin"))
def test_untrained_model_renders_finite_and_causal(arm, pair):
    _, dry = pair
    torch.manual_seed(0)
    model = EnvelopePredictor(arm).eval()
    out, _ = fill.render(dry, RATE, arm, model=model)
    assert np.isfinite(out).all() and out.shape == dry.shape
    # Causality: changing the future does not change earlier output.
    late = dry.copy()
    late[RATE:] *= 0.1
    out2, _ = fill.render(late, RATE, arm, model=model)
    assert np.allclose(out[: RATE - 512], out2[: RATE - 512], atol=1e-12)


def test_deficit_fill_adds_predicted_energy_and_nothing_at_floor(pair, monkeypatch):
    ref, dry = pair
    from models.envelope_predictor import band_matrix

    p_ref, _ = stft_power(ref[:, 0])
    p_dry, _ = stft_power(dry[:, 0])
    deficit = np.log10(np.maximum(p_ref @ band_matrix(RATE) - p_dry @ band_matrix(RATE), 0) + 1e-30)
    monkeypatch.setattr(fill, "predict", lambda model, power, rate: deficit)
    out, _ = fill.fill_channel(dry[:, 0], RATE, "band_deficit", model=object())
    s = score(out[:, None], ref[:, :1], dry[:, :1], RATE)
    assert abs(s["missing_erb_bias_db"]) < 3
    floor = np.full_like(deficit, -30.0)
    monkeypatch.setattr(fill, "predict", lambda model, power, rate: floor)
    out, _ = fill.fill_channel(dry[:, 0], RATE, "band_deficit", model=object())
    assert np.allclose(out, dry[:, 0], atol=1e-9)
