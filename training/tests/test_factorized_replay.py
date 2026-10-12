import torch

from scripts.replay_factorized_study import aligned_band_metrics, hybrid_features
from train import spectral_features


def test_replay_keeps_shared_startup_and_chosen_phase():
    original = torch.randn(2, 2, 96, 129)
    amplitude = torch.rand(2, 96, 129) + 0.01
    phase = torch.randn(2, 96, 129)
    result = hybrid_features(amplitude, phase, original)
    torch.testing.assert_close(result[:, :, :80], original[:, :, :80], atol=0, rtol=0)
    torch.testing.assert_close(result[:, 1, 80:], phase[:, 80:], atol=0, rtol=0)
    torch.testing.assert_close(torch.pow(10.0, result[:, 0, 80:] / 20), amplitude[:, 80:])
    torch.testing.assert_close(hybrid_features(amplitude[:1], phase[:1], original[:1]), result[:1])


def test_aligned_metrics_preserve_synthesis_timeline_and_fft_grid():
    torch.manual_seed(31)
    wave = torch.randn(1, 32768) * 0.01
    padded = torch.nn.functional.pad(wave, (128, 0))
    features = spectral_features(padded, 256, 128)
    raw, output = aligned_band_metrics(
        features, features, padded[:, :32768], torch.tensor([[48000.0]])
    )
    for metrics in (raw, output):
        assert metrics["high_sc"] < 1e-6
        assert metrics["high_flux"] < 1e-6
        assert metrics["high_energy_error_db"] < 1e-6
