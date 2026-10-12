import json

import numpy as np
import torch

from models.envelope_probe import EnvelopeProbe
from scripts.analyze_envelope_study import KEYS, export_probe, paired_comparison


def test_comparison_weights_recordings_and_requires_pairs():
    import pytest

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


def test_amplitude_probe_export_contract_and_budget(tmp_path):
    model = EnvelopeProbe().eval()
    checkpoint = tmp_path / "probe.pth"
    torch.save({"format": "envelope-probe-v1", "model": model.state_dict()}, checkpoint)
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
