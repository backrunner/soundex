import torch

from scripts.replay_factorized_study import hybrid_features


def test_replay_keeps_shared_startup_and_chosen_phase():
    original = torch.randn(2, 2, 96, 129)
    amplitude = torch.rand(2, 96, 129) + 0.01
    phase = torch.randn(2, 96, 129)
    result = hybrid_features(amplitude, phase, original)
    torch.testing.assert_close(result[:, :, :80], original[:, :, :80], atol=0, rtol=0)
    torch.testing.assert_close(result[:, 1, 80:], phase[:, 80:], atol=0, rtol=0)
    torch.testing.assert_close(torch.pow(10.0, result[:, 0, 80:] / 20), amplitude[:, 80:])
    torch.testing.assert_close(hybrid_features(amplitude[:1], phase[:1], original[:1]), result[:1])
