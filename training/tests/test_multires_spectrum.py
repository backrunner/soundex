import pytest
import torch

from models.multires_spectrum import FFT_SIZES, multires_spectrum_loss, spectral_loss
from scripts.run_spectrum_priority_study import protocol


def waves(level=0.05, length=8192):
    torch.manual_seed(32)
    return torch.randn(2, length, dtype=torch.float64) * level, torch.tensor([[44100], [48000]])


@pytest.mark.parametrize("level", [0.0, 1e-7, 0.05])
def test_reference_zero_finite_gradient_and_detached_target(level):
    reference, rates = waves(level)
    reference.requires_grad_()
    prediction = reference.detach().clone().requires_grad_()
    result = spectral_loss(prediction, reference, rates)
    assert result["per_example"].abs().max() < 1e-14
    result["per_example"].sum().backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().max() < 1e-12
    assert reference.grad is None


def test_magnitude_supervision_does_not_require_identical_waveform():
    reference, rates = waves()
    result = spectral_loss(-reference, reference, rates)
    assert result["per_example"].abs().max() < 1e-14
    assert (-reference - reference).square().mean() > 0


def test_equal_energy_wrong_frequency_is_penalized_at_all_resolutions():
    time = torch.arange(8192, dtype=torch.float64) / 48000
    reference = (0.1 * torch.sin(2 * torch.pi * 12000 * time))[None]
    wrong = (0.1 * torch.sin(2 * torch.pi * 15000 * time))[None]
    torch.testing.assert_close(reference.square().mean(), wrong.square().mean())
    result = spectral_loss(wrong, reference, torch.tensor([48000]))
    for resolution in result["resolutions"]:
        assert resolution["components"]["amplitude"].item() > 0.1
        assert resolution["components"]["log_shape"].item() > 0.01


def test_transient_dynamics_follow_reference_not_smoothness():
    reference, rates = waves()
    reference[:, :4096] *= 0.01
    prediction = reference.clone()
    prediction[:, 3072:5120] *= torch.linspace(0.1, 1, 2048)
    assert spectral_loss(reference, reference, rates)["components"]["dynamics"].max() == 0
    assert spectral_loss(prediction, reference, rates)["components"]["dynamics"].min() > 0


@pytest.mark.parametrize("level", [1e-7, 0.05])
def test_gradients_match_finite_difference(level):
    reference, rates = waves(level)
    prediction = (reference * 0.7).requires_grad_()

    def score():
        return spectral_loss(prediction, reference, rates)["per_example"].sum()

    score().backward()
    index = (0, 4096)
    analytical = prediction.grad[index].item()
    with torch.no_grad():
        value, epsilon = prediction[index].item(), level * 1e-4
        prediction[index] = value + epsilon
        plus = score().item()
        prediction[index] = value - epsilon
        minus = score().item()
        prediction[index] = value
    assert analytical == pytest.approx((plus - minus) / (2 * epsilon), rel=0.002, abs=1e-6)
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad * reference).sum() < 0


def test_noise_in_silence_is_penalized_and_channel_views_are_independent():
    noise, rates = waves(1e-7)
    result = spectral_loss(noise, torch.zeros_like(noise), rates)
    assert result["per_example"].min() > 0
    for i in range(2):
        solo = spectral_loss(noise[i : i + 1], torch.zeros_like(noise[i : i + 1]), rates[i : i + 1])
        torch.testing.assert_close(solo["per_example"][0], result["per_example"][i])


def feature_batch():
    wave, rates = waves(length=128 * 129)
    wave[:, :128] = 0
    spectrum = torch.fft.rfft(wave.unfold(-1, 256, 128) * torch.hann_window(256, dtype=wave.dtype))
    features = torch.stack((20 * spectrum.abs().clamp_min(1e-10).log10(), spectrum.angle()), dim=1)
    return {
        "x": features.clone(),
        "y": features.clone(),
        "clean_wave": wave[:, : 128 * 128],
        "rate": rates,
        "identity": torch.tensor([False, True]),
    }


def test_raw_ola_alignment_reference_and_dsp_independence():
    batch = feature_batch()
    exact = multires_spectrum_loss(batch["y"], batch)
    assert exact["total"].abs() < 1e-12
    prediction = batch["x"].clone()
    prediction[0, 0, 80:, 48:] -= 6
    prediction.requires_grad_()
    original = multires_spectrum_loss(prediction, batch)
    original["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :, :80].abs().max() == 0
    assert prediction.grad[0, 0, 90:, 48:].sum() < 0
    batch.update(mask=torch.zeros_like(prediction), active=torch.zeros(2, 128))
    torch.testing.assert_close(
        multires_spectrum_loss(prediction, batch)["total"], original["total"], atol=0, rtol=0
    )
    batch["clean_wave"] = torch.roll(batch["clean_wave"], 1, -1)
    assert multires_spectrum_loss(prediction, batch)["total"] != original["total"]


def test_short_support_rate_and_role_errors_are_explicit():
    wave, rates = waves(length=2048)
    with pytest.raises(ValueError, match="two complete"):
        spectral_loss(wave, wave, rates)
    wave, rates = waves()
    with pytest.raises(ValueError, match="physical sample rates"):
        spectral_loss(wave, wave, torch.tensor([32000, 48000]))
    batch = feature_batch()
    batch["identity"].fill_(True)
    with pytest.raises(ValueError, match="roles"):
        multires_spectrum_loss(batch["x"], batch)


def test_protocol_keeps_architecture_and_budgets_matched_and_routes_explicit():
    parent = dict(
        specs=[],
        weight=0.13,
        rate=1e-4,
        direct_loss={},
        optimization="same",
        seeds=[20261012, 20261013],
    )
    config = protocol(parent)
    assert config["steps"] == 1024 and len(config["seeds"]) == 2
    assert len({str(v) for v in config["model_configs"].values()}) == 1
    assert config["multires_loss"]["fft_sizes"] == list(FFT_SIZES)
    assert config["loss_routes"]["control"] == {"raw": 1, "budget": 1}
    assert config["loss_routes"]["mr_add"] == {"raw": 1, "multires": 1, "budget": 1}
    assert config["loss_routes"]["mr_primary"] == {"multires": 1, "complex_guard": 1, "budget": 1}
    assert protocol(parent, preflight=True)["steps"] == 4
