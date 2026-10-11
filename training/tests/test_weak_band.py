"""Loss semantics, gradients, identity controls and architecture expressivity."""

import numpy as np
import pytest
import torch

from data.weak_band_study import add_identity_controls, audit_groups, identity_specs
from models.causal_restorer import CausalRestorer
from models.weak_band_objective import excess_band_energy, objective
from tests.test_causal_objective import make_batch


@pytest.mark.parametrize("rate", [44100, 48000])
def test_excess_energy_identity_silence_and_missing_band(rate):
    torch.set_num_threads(1)
    target = torch.ones(1, 3, 129, dtype=torch.complex128) * 1e-4
    target[..., :20] = 1
    rates = torch.tensor([[rate]], dtype=torch.float64)
    identity = target.clone().requires_grad_()
    loss = excess_band_energy(identity, target, rates, 256)
    loss.backward()
    assert loss == 0 and bool((identity.grad == 0).all())
    added = target.clone()
    added[..., 100:110] = 0.01
    assert excess_band_energy(added, target, rates, 256) > 0.1
    # Actual reference energy is not forbidden by a uniform high-band cutoff.
    assert excess_band_energy(added, added, rates, 256) == 0
    silence = torch.zeros_like(target).requires_grad_()
    zero = excess_band_energy(silence, torch.zeros_like(target), rates, 256)
    zero.backward()
    assert zero == 0 and bool(torch.isfinite(silence.grad).all())
    # Check analytic derivatives against finite differences away from ReLU kinks.
    probe = (target * 2).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda p: excess_band_energy(p, target, rates, 256),
        (probe,),
        eps=1e-7,
        atol=1e-4,
        rtol=1e-3,
    )


def test_reconstruction_beats_dry_and_noise_and_gradients_remain_finite(tmp_path):
    n = np.arange(32768)
    dry = (0.2 * np.sin(n * 2 * np.pi * 1700 / 48000)).astype(np.float32)
    clean = dry + (0.02 * np.sin(n * 2 * np.pi * 15000 / 48000)).astype(np.float32)
    batch = make_batch(dry, 48000, tmp_path)
    target = make_batch(clean, 48000, tmp_path)
    batch["y"], batch["clean_wave"] = target["y"], target["clean_wave"]
    oracle = objective(batch["y"], batch)["total"]
    assert oracle < objective(batch["x"], batch)["total"]
    noisy = batch["y"].clone()
    noisy[:, 0, :, 100:] += 20
    noisy.requires_grad_()
    result = objective(noisy, batch)
    assert result["total"] > oracle
    result["total"].backward()
    assert bool(torch.isfinite(noisy.grad).all())


@pytest.mark.parametrize("kind", ["gru", "tcn"])
def test_architecture_can_represent_identity_and_high_band_insertion(kind):
    model = CausalRestorer(kind, width=8).eval()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
    x = torch.zeros(1, 2, 8, 129)
    x[:, 0] = -60
    x[:, 0, :, :32] = -10
    rate = torch.tensor([[48000.0]])
    expected = torch.stack((torch.pow(10.0, x[:, 0] / 20), torch.zeros_like(x[:, 0])), 1)
    actual, _ = model.forward_complex(x, model.initial_state(1), rate)
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        model.decoder[-1].bias[100] = 0.1
    restored, _ = model.forward_complex(x, model.initial_state(1), rate)
    assert bool((restored[:, 0, :, 100] > actual[:, 0, :, 100]).all())
    torch.testing.assert_close(restored[:, :, :, :32], actual[:, :, :, :32])


def test_identity_schedule_and_split_group_leakage():
    draws = [{"row_id": str(i), "start": i, "gain_db": 1.0} for i in range(32)]
    mixed = add_identity_controls(draws, 4)
    assert [i for i, (a, b) in enumerate(zip(draws, mixed, strict=True)) if a != b] == [
        0,
        8,
        16,
        24,
    ]
    assert all(
        a["start"] == b["start"] and a["gain_db"] == b["gain_db"]
        for a, b in zip(draws, mixed, strict=True)
    )
    assert identity_specs(draws)[0]["row_id"] == "0:identity"
    rows = {
        str(i): {
            "corpus": "music_library",
            "track_id": str(i),
            "split": split,
            "source_metadata": {"split_group": "same-work"},
        }
        for i, split in enumerate(["train", "validation"])
    }
    with pytest.raises(ValueError, match="work-group split leakage"):
        audit_groups(rows)


def test_band_power_matches_parseval_for_a_known_sine():
    size = 256
    n = torch.arange(size, dtype=torch.float64)
    wave = 0.1 * torch.sin(2 * torch.pi * 80 * n / size)
    spec = torch.fft.rfft(wave * torch.hann_window(size, dtype=torch.float64))[None, None]
    actual = excess_band_energy(2 * spec, spec, torch.tensor([[48000.0]]), size)
    # 15 kHz sine lies inside a single band; RMS^2 is amplitude^2/2.
    power = 0.1**2 / 2
    expected = np.log1p(3 * power / (power + power * 1e-6)) / 7
    assert float(actual) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_review_selection_is_frozen_per_recording_and_excludes_train():
    from data.weak_band_study import review_specs

    rows = {
        "z": {"row_id": "z", "track_id": "a", "split": "test", "held_out": True},
        "b": {"row_id": "b", "track_id": "a", "split": "test", "held_out": True},
        "a": {"row_id": "a", "track_id": "x", "split": "train", "held_out": False},
    }
    assert review_specs(rows) == [{"row_id": "b", "start": 0, "gain_db": 0.0}]
    assert review_specs(dict(reversed(list(rows.items())))) == review_specs(rows)


def test_frozen_binary_isolated_from_build_and_rejects_mutation(tmp_path):
    from evaluation.reporting import sha256_file
    from scripts.run_weak_band_study import freeze_binary

    source = tmp_path / "built"
    source.write_bytes(b"registered")
    expected = sha256_file(source)
    frozen = freeze_binary(source, tmp_path / "frozen", expected)
    source.write_bytes(b"rebuilt")
    assert frozen.read_bytes() == b"registered"
    assert freeze_binary(source, frozen, expected) == frozen
    with pytest.raises(ValueError, match="source binary changed"):
        freeze_binary(source, tmp_path / "new", expected)
    frozen.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="frozen binary changed"):
        freeze_binary(source, frozen, expected)
