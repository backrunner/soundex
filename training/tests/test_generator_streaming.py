"""Streaming-contract tests for the SoundEx generator."""

import torch
import torch.nn as nn

from models.generator import SoundExGenerator


def small_generator() -> SoundExGenerator:
    """Return a fast model with the production topology."""
    torch.manual_seed(7)
    model = SoundExGenerator(channels=[4, 8, 16, 16], bottleneck_blocks=1)
    return model.eval()


def test_generator_preserves_exact_shape_for_odd_frequency_width() -> None:
    model = small_generator()
    input_features = torch.randn(2, 2, 1, 65)

    with torch.no_grad():
        output = model(input_features)

    assert output.shape == input_features.shape


def test_time_positions_are_equivalent_to_independent_frames() -> None:
    model = small_generator()
    input_features = torch.randn(2, 2, 5, 65)

    with torch.no_grad():
        together = model(input_features)
        independent = torch.cat(
            [model(input_features[:, :, index : index + 1]) for index in range(5)],
            dim=2,
        )

    torch.testing.assert_close(together, independent, rtol=1e-6, atol=1e-6)


def test_future_and_past_frames_cannot_change_current_output() -> None:
    model = small_generator()
    original = torch.randn(1, 2, 5, 65)
    changed = original.clone()
    changed[:, :, :2] += 100.0
    changed[:, :, 3:] -= 100.0

    with torch.no_grad():
        original_output = model(original)[:, :, 2]
        changed_output = model(changed)[:, :, 2]

    torch.testing.assert_close(original_output, changed_output, rtol=0.0, atol=0.0)


def test_all_convolutions_have_unit_time_kernel_and_stride() -> None:
    model = SoundExGenerator()

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            assert module.kernel_size[0] == 1
            assert module.stride[0] == 1


def test_production_model_is_below_parameter_budget() -> None:
    assert SoundExGenerator().count_parameters() < 2_000_000
