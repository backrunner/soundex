import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.envelope_probe import EnvelopeProbe
from models.factorized_envelope import make_factor_model
from models.residual_energy import make_energy_model
from scripts.analyze_envelope_study import KEYS, export_probe, paired_comparison


def test_comparison_weights_recordings_and_requires_pairs():
    rows = []
    for corpus in ("music_library", "speech_library"):
        for index, track, error in ((0, "a", 1.0), (1, "a", 1.0), (2, "b", 3.0)):
            rows.append(
                {
                    "row_id": f"{corpus}-{index}",
                    "track_id": track,
                    "corpus": corpus,
                    "metrics": dict.fromkeys(KEYS, error),
                }
            )
    zero = json.loads(json.dumps(rows))
    for row in zero:
        row["metrics"] = dict.fromkeys(KEYS, 0.0)
    result = paired_comparison({"rows": rows}, {"rows": zero})
    assert result["music_library"]["total"]["candidate_minus_control"] == 2
    with pytest.raises(ValueError, match="unpaired"):
        paired_comparison({"rows": rows}, {"rows": zero[:-1]})


@pytest.mark.parametrize(
    "family", ["envelope-probe-v1", "factor-envelope-v1", "energy-envelope-v1"]
)
def test_amplitude_probe_export_contract_and_budget(tmp_path, family):
    model = (
        EnvelopeProbe() if family == "envelope-probe-v1" else make_factor_model("factorized")
    ).eval()
    arm = "factorized"
    if family == "energy-envelope-v1":
        arm = "rms_residual"
        model = make_energy_model(arm).eval()
    if family != "envelope-probe-v1":
        with torch.no_grad():
            model.energy_head.weight.normal_(std=0.01)
            model.decoder[-1].weight.normal_(std=0.01)
    checkpoint = tmp_path / "probe.pth"
    torch.save({"format": family, "arm": arm, "model": model.state_dict()}, checkpoint)
    session, result = export_probe(checkpoint, tmp_path / "probe.onnx")
    assert result["parameters"] < 2_000_000
    assert result["bytes"] < 8 * 1024 * 1024
    assert not result["production_compatible"]
    state = np.zeros((1, 9072), dtype=np.float32)
    data = {
        "magnitude": np.zeros((1, 1, 129), dtype=np.float32),
        "context": np.zeros((1, 1, 1538), dtype=np.float32),
        "condition": np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32),
        "state": state,
    }
    out = session.run(None, data)
    assert np.isfinite(out[0]).all() and np.max(out[0]) < 1e-9
    assert session.get_modelmeta().custom_metadata_map["soundex.production_compatible"] == "false"
    memory = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts/measure_probe_memory.py"),
                "--model",
                str(tmp_path / "probe.onnx"),
            ],
            text=True,
        )
    )
    assert memory["onnx_sha256"] == result["onnx_sha256"]
    assert memory["after_inference_peak_mib"] >= memory["baseline_import_peak_mib"]
