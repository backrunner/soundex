"""Feature and context tests shared by training and deployment."""

import pytest
import torch

from models.generator import SoundExGenerator
from train import (
    build_lr_scheduler,
    infer_independent_frames,
    select_causal_sequence,
    select_context,
    spectral_features,
)


def test_context_must_be_one_frame() -> None:
    features = torch.randn(2, 4, 8, 65)

    with pytest.raises(ValueError, match="context_frames must be 1"):
        select_context(features, 8)


def test_validation_context_is_deterministic_center_frame() -> None:
    features = torch.arange(2 * 4 * 7 * 3, dtype=torch.float32).reshape(2, 4, 7, 3)

    selected = select_context(features, 1, randomize=False)

    torch.testing.assert_close(selected, features[:, :, 3:4])


def test_spectral_features_are_single_contract_scale() -> None:
    fft_size = 64
    impulse = torch.zeros(1, fft_size)
    impulse[:, 0] = 1.0

    features = spectral_features(impulse, fft_size=fft_size, hop_size=16)

    assert features.shape == (1, 2, 1, fft_size // 2 + 1)
    assert torch.isfinite(features).all()


def test_causal_sequence_validation_selection_is_fixed() -> None:
    features = torch.arange(2 * 4 * 9 * 3, dtype=torch.float32).reshape(2, 4, 9, 3)

    selected = select_causal_sequence(features, 4, randomize=False)

    torch.testing.assert_close(selected, features[:, :, 2:6])


def test_sequence_inference_matches_independent_t1_calls() -> None:
    generator = SoundExGenerator(channels=[2, 4], bottleneck_blocks=1, expand_ratio=2).eval()
    features = torch.randn(2, 2, 3, 17)

    actual = infer_independent_frames(generator, features)
    expected = torch.stack(
        [generator(features[:, :, frame : frame + 1]) for frame in range(3)],
        dim=2,
    ).squeeze(3)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_warmup_cosine_scheduler_resumes_the_same_update_trajectory() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(optimizer, total_steps=8, warmup_steps=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    for _ in range(3):
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    expected = []
    for _ in range(3):
        expected.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=1.0)
    resumed_scheduler = build_lr_scheduler(resumed_optimizer, total_steps=8, warmup_steps=2)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    actual = []
    for _ in range(3):
        actual.append(resumed_optimizer.param_groups[0]["lr"])
        resumed_optimizer.step()
        resumed_scheduler.step()

    assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="warmup_steps"):
        build_lr_scheduler(resumed_optimizer, total_steps=8, warmup_steps=8)
