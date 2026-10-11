"""Finite causal history must agree across training, ONNX and Rust deployment."""

from copy import deepcopy

import numpy as np
import onnxruntime as ort
import pytest
import torch

from artifact_contract import ExportValidationError, frame_contract_from_metadata
from checkpoint import CheckpointSchemaError, feature_contract_from_config
from evaluation.parity import generate_parity_evidence
from export_onnx import _build_generator, export_onnx
from models.generator import SoundExGenerator
from train import infer_independent_frames, initialize_generator_from_checkpoint


def config_with_history(config, history):
    config = deepcopy(config)
    config["model"]["generator"]["spectral_refiner"] = dict(
        bins=129, width=32, representation="gain_shape", history_frames=history
    )
    config["training"]["context_frames"] = history + 1
    config["training"]["sequence_warmup_frames"] = 8
    return config


def test_temporal_migration_is_explicit_identity_and_learns(checkpoint_factory, resolved_config):
    config = config_with_history(resolved_config, 0)
    path, parent = checkpoint_factory(config=config)
    # A learned parent head is essential: a zero head can hide a broken temporal path.
    parent["model"]["generator_state"]["spectral_refiner.head.weight"].normal_(std=0.01)
    torch.save(parent, path)
    config = config_with_history(config, 8)
    target_path, target = checkpoint_factory("temporal.pth", config=config)
    model = _build_generator(target)
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(path, model, config, parent["data"])
    receipt = initialize_generator_from_checkpoint(
        path, model, config, parent["data"], initialize_temporal_refiner=True
    )
    assert len(receipt["added_state_keys"]) == 4
    assert len(receipt["zero_initialized_state_keys"]) == 2
    x = torch.randn(2, 2, 9, 129)
    x[:, 0] = x[:, 0] * 10 - 40
    # Encoding nine rows changes GEMM grouping; permit FP32 rounding, not a
    # learned correction at initialization (1e-6 dB / rad absolute ceiling).
    torch.testing.assert_close(model(x), _build_generator(parent)(x[:, :, -1:]), rtol=0, atol=1e-6)
    optimizer = torch.optim.Adam(model.spectral_refiner.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    assert model.spectral_refiner.temporal_filter.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="exactly zero"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_temporal_refiner=True
        )
    changed_history = x.clone()
    changed_history[:, :, 0] += 20
    assert not torch.allclose(model(x), model(changed_history), rtol=0, atol=1e-7)
    assert target["model"]["architecture"]["minor"] == 4
    assert target_path.exists()


def test_windows_are_causal_and_match_explicit_streaming():
    model = SoundExGenerator(
        channels=[4, 8, 16, 16],
        bottleneck_blocks=1,
        spectral_refiner=dict(bins=129, width=32, representation="gain_shape", history_frames=8),
    ).eval()
    with torch.no_grad():
        model.spectral_refiner.head.weight.normal_(std=0.03)
        model.spectral_refiner.temporal_mix.weight.normal_(std=0.03)
    x = torch.randn(2, 2, 19, 129)
    x[:, 0] = x[:, 0] * 15 - 50
    actual = infer_independent_frames(model, x)
    history = x.new_zeros(2, 2, 8, 129)
    history[:, 0] = -200
    expected = []
    for t in range(x.shape[2]):
        window = torch.cat((history, x[:, :, t : t + 1]), dim=2)
        expected.append(model(window))
        history = window[:, :, 1:]
    torch.testing.assert_close(actual, torch.cat(expected, dim=2), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        infer_independent_frames(model, x, warmup_frames=8), actual[:, :, 8:]
    )
    future = x.clone()
    future[:, :, 11:] += 50
    torch.testing.assert_close(
        infer_independent_frames(model, future)[:, :, :11], actual[:, :, :11]
    )
    other_channel = x.clone()
    other_channel[1] += 20
    torch.testing.assert_close(infer_independent_frames(model, other_channel)[0], actual[0])
    # Only the preceding eight observations can affect the last prediction.
    distant = x.clone()
    distant[:, :, :-9] += 50
    torch.testing.assert_close(infer_independent_frames(model, distant)[:, :, -1], actual[:, :, -1])


def test_nonzero_temporal_export_parity_and_versioning(
    tmp_path, checkpoint_factory, resolved_config
):
    config = config_with_history(resolved_config, 8)
    path, checkpoint = checkpoint_factory(config=config)
    state = checkpoint["model"]["generator_state"]
    state["spectral_refiner.head.weight"].normal_(std=0.01)
    state["spectral_refiner.temporal_mix.weight"].normal_(std=0.03)
    torch.save(checkpoint, path)
    result = export_onnx(path, tmp_path / "temporal.onnx")
    session = ort.InferenceSession(str(result.path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["soundex.artifact_schema"] == "1.4"
    assert session.get_inputs()[0].shape == ["batch", 2, 9, 129]
    assert session.get_outputs()[0].shape == ["batch", 2, 1, 129]
    assert frame_contract_from_metadata(metadata) == (256, 128)
    model = _build_generator(checkpoint)
    with torch.no_grad():
        for batch in (1, 2):
            x = torch.randn(batch, 2, 9, 129)
            np.testing.assert_allclose(
                session.run(None, {"input_features": x.numpy()})[0],
                model(x).numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
    metadata["soundex.artifact_schema"] = "1.3"
    with pytest.raises(ExportValidationError, match=r"schema 1\.4"):
        frame_contract_from_metadata(metadata)
    generate_parity_evidence(
        checkpoint_path=path, model_path=result.path, output_path=tmp_path / "parity.json"
    )


def test_context_contract_and_rng_budget(resolved_config):
    config = config_with_history(resolved_config, 8)
    config["training"]["context_frames"] = 1
    with pytest.raises(CheckpointSchemaError, match="context_frames"):
        feature_contract_from_config(config)
    config = config_with_history(resolved_config, 8)
    config["training"]["sequence_warmup_frames"] = 7
    with pytest.raises(CheckpointSchemaError, match="full observed history"):
        feature_contract_from_config(config)
    kwargs = dict(bins=129, width=128, representation="gain_shape")
    torch.manual_seed(9)
    frame = SoundExGenerator(spectral_refiner=kwargs)
    rng = torch.get_rng_state()
    torch.manual_seed(9)
    temporal = SoundExGenerator(spectral_refiner={**kwargs, "history_frames": 8})
    assert torch.equal(torch.get_rng_state(), rng)
    assert temporal.count_parameters() <= 2_000_000
    for name, value in frame.state_dict().items():
        torch.testing.assert_close(temporal.state_dict()[name], value, rtol=0, atol=0)


def test_validation_audit_uses_real_history_without_future_or_cold_padding():
    from scripts.audit_validation_parity import observed_windows

    features = torch.arange(20 * 2 * 129).reshape(20, 2, 1, 129).float()
    positions = np.array([8, 19])
    windows = observed_windows(features, positions, 9)
    assert windows.shape == (2, 2, 9, 129)
    torch.testing.assert_close(windows[0], features[:9, :, 0].permute(1, 0, 2))
    torch.testing.assert_close(windows[1], features[11:, :, 0].permute(1, 0, 2))
    torch.testing.assert_close(observed_windows(features, positions, 1), features[positions])
    with pytest.raises(ValueError, match="complete observed history"):
        observed_windows(features, np.array([7]), 9)


def test_temporal_validation_rejects_missing_warmup_before_reading_audio():
    from train import validate

    class Temporal(torch.nn.Identity):
        context_frames = 9

    with pytest.raises(ValueError, match="real warmup"):
        validate(Temporal(), None, None, torch.device("cpu"), {}, 8)


def test_temporal_only_optimization_preserves_every_parent_parameter_and_buffer():
    from train import generator_trainable_parameters

    model = SoundExGenerator(
        channels=[4, 8, 16, 16],
        bottleneck_blocks=1,
        spectral_refiner=dict(bins=129, width=32, representation="gain_shape", history_frames=8),
    ).eval()
    with torch.no_grad():
        model.spectral_refiner.head.weight.normal_(std=0.03)
    config = {
        "trainable_scope": "temporal_only",
        "batch_norm_statistics": "frozen",
        "objective": {},
    }
    parameters = generator_trainable_parameters(model, config)
    assert len(parameters) == 4
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(2, 2, 9, 129)
    x[:, 0] = x[:, 0] * 15 - 40
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("spectral_refiner.temporal_"):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert not torch.equal(
        model.spectral_refiner.temporal_mix.weight, before["spectral_refiner.temporal_mix.weight"]
    )
    assert model.spectral_refiner.temporal_filter.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="frozen BN"):
        generator_trainable_parameters(model, {**config, "batch_norm_statistics": "update"})
    all_parameters = generator_trainable_parameters(model, {"trainable_scope": "all"})
    assert len(all_parameters) == len(list(model.parameters()))
