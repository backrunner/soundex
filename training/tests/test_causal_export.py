"""Explicit-state artifacts are checked before publishing and rejected if mislabeled."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import onnx
import pytest
import torch

import export_causal as exporter
from models.causal_restorer import CausalRestorer


def checkpoint(root, kind):
    model = CausalRestorer(kind, width=16)
    path = root / f"{kind}.pth"
    torch.save(
        {
            "format": "soundex-causal-study-v1",
            "kind": kind,
            "width": 16,
            "model": model.state_dict(),
            "config": {"synthetic_test": True},
            "provenance": {"data_recipe_sha256": "a" * 64, "manifest_set_sha256": "b" * 64},
        },
        path,
    )
    return path


@pytest.mark.parametrize("kind", ["gru", "tcn"])
def test_stateful_export_roundtrip_and_rust_sequence(tmp_path, kind):
    source = checkpoint(tmp_path, kind)
    destination = tmp_path / "model.onnx"
    metadata = exporter.export_causal(source, destination)
    assert metadata["soundex.artifact_schema"] == "2.0"
    assert metadata["soundex.stateless"] == "false"
    repository = Path(__file__).resolve().parents[2]
    binary = repository / "target/debug/soundex-state-parity"
    assert binary.is_file(), "prebuild soundex-state-parity before running export tests"
    frame = [-30.0] * 129 + [0.0] * 129
    inputs = tmp_path / "input.json"
    inputs.write_text(
        json.dumps({"sample_rate": 48000, "channels": 1, "resets": [70], "frames": [frame] * 72})
    )
    result = tmp_path / "output.json"
    command = [str(binary), str(destination), str(inputs), str(result)]
    subprocess.run(command, check=True, capture_output=True)
    values = json.loads(result.read_text())
    assert values["runs"] == 72
    assert values["frames"][0] == values["frames"][70]
    assert values["frames"][1] == values["frames"][71]
    # Reject semantic and tensor/state mismatch, without accepting a legacy alias.
    for key, bad in [
        ("soundex.state_size", "32769"),
        ("soundex.stateless", "true"),
        ("soundex.warmup_frames", "0"),
        ("soundex.sample_rate_input_units", "kHz"),
        ("soundex.parameter_count", "2000001"),
        ("soundex.artifact_schema", "1.4"),
    ]:
        graph = onnx.load(destination)
        changed = dict(metadata)
        changed[key] = bad
        onnx.helper.set_model_props(graph, changed)
        path = tmp_path / "invalid.onnx"
        onnx.save(graph, path)
        invalid = subprocess.run(
            [str(binary), str(path), str(inputs), str(result)], capture_output=True
        )
        assert invalid.returncode != 0, (key, bad)


def test_parity_failure_does_not_publish_or_replace(tmp_path, monkeypatch):
    source = checkpoint(tmp_path, "gru")
    destination = tmp_path / "existing.onnx"
    destination.write_bytes(b"existing artifact")

    def reject(*_):
        raise ValueError("injected sequential parity failure")

    monkeypatch.setattr(exporter, "verify_stateful_export", reject)
    with pytest.raises(ValueError, match="sequential parity"):
        exporter.export_causal(source, destination)
    assert destination.read_bytes() == b"existing artifact"
    assert not destination.with_suffix(".partial.onnx").exists()
