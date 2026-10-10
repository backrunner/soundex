"""Refinement must preserve the parent, frame contract and deployable behavior."""

from copy import deepcopy

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export_onnx import _build_generator, export_onnx
from models.generator import SoundExGenerator
from models.spectral_refiner import SpectralRefiner
from train import initialize_generator_from_checkpoint


@pytest.mark.parametrize("representation", ["polar", "gain_shape"])
def test_identity_rng_causality_and_learning(representation):
    kwargs = dict(channels=[4, 8, 16, 16], bottleneck_blocks=1)
    torch.manual_seed(17)
    parent = SoundExGenerator(**kwargs).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(17)
    model = SoundExGenerator(
        **kwargs, spectral_refiner=dict(bins=129, width=32, representation=representation)
    ).eval()
    assert torch.equal(rng, torch.get_rng_state())
    features = torch.randn(2, 2, 4, 129)
    torch.testing.assert_close(model(features), parent(features), rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.spectral_refiner.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(features).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.spectral_refiner.parameters())
        optimizer.step()
    assert model.spectral_refiner.encoder[0].weight.grad.abs().sum() > 0
    independent = torch.cat([model(features[:, :, t : t + 1]) for t in range(4)], dim=2)
    torch.testing.assert_close(model(features), independent, rtol=1e-5, atol=1e-6)
    perturbed = features.clone()
    perturbed[1, :, -1] += 20
    torch.testing.assert_close(model(perturbed)[0], model(features)[0], rtol=0, atol=0)
    torch.testing.assert_close(
        model(perturbed)[:, :, :-1], model(features)[:, :, :-1], rtol=0, atol=0
    )


def test_circular_encoding_attenuates_quiet_phase_and_bounds_correction():
    model = SpectralRefiner(bins=129, width=32, representation="gain_shape")
    features = torch.zeros(2, 2, 3, 129)
    features[:, 0, :, 1:] = -120
    wrapped = features.clone()
    wrapped[:, 1] += 2 * torch.pi
    torch.testing.assert_close(model.encode_features(features), model.encode_features(wrapped))
    encoded = model.encode_features(features).reshape(2, 3, 129, 3)
    assert encoded[..., 1:, 1:].abs().max() <= 0.001001
    with torch.no_grad():
        model.head.weight.fill_(100)
        model.head.bias.fill_(100)
    output = model(features)
    assert torch.isfinite(output).all()
    assert output[:, 0].abs().max() <= 12
    assert output[:, 1].abs().max() <= 0.5
    with pytest.raises(ValueError, match="FFT geometry"):
        model(torch.zeros(1, 2, 1, 257))


def test_explicit_migration_and_learned_export(tmp_path, checkpoint_factory):
    path, parent = checkpoint_factory()
    config = deepcopy(parent["resolved_config"])
    spec = dict(bins=129, width=128, representation="gain_shape")
    config["model"]["generator"]["spectral_refiner"] = spec
    generator_config = config["model"]["generator"]
    model = SoundExGenerator(
        channels=generator_config["channels"],
        bottleneck_blocks=generator_config["bottleneck_blocks"],
        expand_ratio=generator_config["expand_ratio"],
        spectral_refiner=spec,
    )
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(path, model, config, parent["data"])
    receipt = initialize_generator_from_checkpoint(
        path, model, config, parent["data"], initialize_spectral_refiner=True
    )
    assert len(receipt["zero_initialized_state_keys"]) == 2
    features = torch.randn(2, 2, 1, 129)
    model.eval()
    torch.testing.assert_close(model(features), _build_generator(parent)(features), rtol=0, atol=0)
    model.spectral_refiner.head.bias.data.fill_(0.1)
    with pytest.raises(ValueError, match="exactly zero"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_spectral_refiner=True
        )
    export_path, checkpoint = checkpoint_factory("refiner.pth", config=config)
    assert checkpoint["model"]["architecture"]["minor"] == 3
    # Export a learned, nonzero branch; zero-only export can hide broken operators.
    checkpoint["model"]["generator_state"]["spectral_refiner.head.weight"].normal_(std=0.002)
    torch.save(checkpoint, export_path)
    result = export_onnx(export_path, tmp_path / "refiner.onnx")
    assert result.size_bytes < 8 * 1024**2
    session = ort.InferenceSession(str(result.path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        model = _build_generator(checkpoint)
        for batch in (1, 2):
            x = torch.randn(batch, 2, 1, 129)
            expected = model(x).numpy()
            actual = session.run(None, {"input_features": x.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("width", [128, 512])
def test_capacity_remains_within_budget(width):
    model = SoundExGenerator(
        spectral_refiner=dict(bins=129, width=width, representation="gain_shape")
    )
    assert 806276 < model.count_parameters() <= 2_000_000
