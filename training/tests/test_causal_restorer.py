"""Temporal/state invariants for the new backbone, independently of quality."""

import pytest
import torch

from models.causal_restorer import CausalRestorer


@pytest.mark.parametrize("kind", ["gru", "tcn"])
def test_sequence_matches_irregular_chunks_and_future_cannot_change_past(kind):
    torch.manual_seed(310)
    model = CausalRestorer(kind, width=16).eval()
    x = torch.randn(2, 2, 73, 129)
    x[:, 0] = x[:, 0] * 10 - 25
    rate = torch.tensor([[44100.0], [48000.0]])
    initial = model.initial_state(2)
    expected, final = model.forward_complex(x, initial, rate)
    state, parts = initial, []
    offset = 0
    for length in [1, 9, 32, 7, 24]:
        result, state = model.forward_complex(x[:, :, offset : offset + length], state, rate)
        parts.append(result)
        offset += length
    torch.testing.assert_close(torch.cat(parts, dim=2), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state, final, atol=2e-6, rtol=2e-5)
    changed = x.clone()
    changed[:, :, 40:] += 10
    actual, _ = model.forward_complex(changed, initial, rate)
    torch.testing.assert_close(actual[:, :, :40], expected[:, :, :40], atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["gru", "tcn"])
def test_channel_state_is_independent_reset_and_circular_phase(kind):
    torch.manual_seed(311)
    model = CausalRestorer(kind, width=16).eval()
    x = torch.randn(2, 2, 8, 129)
    x[:, 0] -= 20
    rate = torch.full((2, 1), 48000.0)
    batched, state = model.forward_complex(x, model.initial_state(2), rate)
    for channel in range(2):
        single, single_state = model.forward_complex(
            x[channel : channel + 1], model.initial_state(1), rate[channel : channel + 1]
        )
        torch.testing.assert_close(single, batched[channel : channel + 1], atol=1e-6, rtol=2e-5)
        torch.testing.assert_close(single_state, state[channel : channel + 1], atol=2e-6, rtol=2e-5)
    rotated = x.clone()
    rotated[:, 1] += 2 * torch.pi
    circular, _ = model.forward_complex(rotated, model.initial_state(2), rate)
    torch.testing.assert_close(circular, batched, atol=2e-6, rtol=2e-5)
    reset, _ = model.forward_complex(x, model.initial_state(2), rate)
    torch.testing.assert_close(reset, batched, atol=0, rtol=0)
    assert torch.count_nonzero(batched[:, 1, :, [0, -1]]) == 0


@pytest.mark.parametrize("kind", ["gru", "tcn"])
def test_silence_gradients_and_default_budget(kind):
    model = CausalRestorer(kind)
    assert sum(p.numel() for p in model.parameters()) <= 2_000_000
    x = torch.zeros(1, 2, 3, 129)
    x[:, 0] = -200
    output, _ = model(x, model.initial_state(1), torch.tensor([[48000.0]]))
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
