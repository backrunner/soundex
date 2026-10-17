import json

import numpy as np
import pytest

from evaluation.replay_parity import parity_report
from models.causal_objective import render
from models.fit_diagnostic import FixedFitObjective
from tests.test_fit_diagnostic import paired_batch


def test_original_parity_gate_is_not_relaxed():
    wave = np.zeros(32)
    assert parity_report(wave, wave, slice(None))["training_frontend_parity_passed"]
    with pytest.raises(ValueError, match="native parity failure"):
        parity_report(wave + 2.1e-5, wave, slice(None))
    with pytest.raises(ValueError, match="invalid aligned replay PCM"):
        parity_report(np.full(32, np.nan), wave, slice(None))


def test_runtime_observation_diagnostic_retains_training_failure(tmp_path, monkeypatch):
    _, batch = paired_batch()
    batch = {k: v[:1] for k, v in batch.items()}
    objective = FixedFitObjective(batch)
    prediction = batch["x"].clone()
    observed = prediction.clone()
    observed[:, 1] += 0.02
    native = render(
        prediction,
        observed,
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )[0][0].numpy()
    trace = {
        "schema_version": 1,
        "fft_size": 256,
        "hop_size": 128,
        "channels": 1,
        "sample_rate": 48000,
        "layout": "hop,channel,magnitude_db_then_phase,bin",
        "features": observed.permute(2, 0, 1, 3).reshape(-1).tolist(),
    }

    def fake_run(*args, **kwargs):
        (tmp_path / "trace.json").write_text(json.dumps(trace))

    monkeypatch.setattr("evaluation.replay_parity.subprocess.run", fake_run)
    kwargs = dict(
        observed_binary=tmp_path / "observer",
        input_path=tmp_path / "input.sxa",
        trace_path=tmp_path / "trace.json",
        prediction=prediction,
        batch=batch,
    )
    proxy = objective.render(prediction)[0].numpy()
    result = parity_report(native, proxy, objective.region, **kwargs)
    assert not result["training_frontend_parity_passed"]
    assert result["proxy_max_error"] > 2e-5
    assert result["runtime_observed_parity_passed"]
    assert result["runtime_observed_proxy_max_error"] == 0
    with pytest.raises(ValueError, match="runtime-observed synthesis parity failure"):
        parity_report(native + 3e-5, proxy, objective.region, **kwargs)
    trace["sample_rate"] = 44100
    with pytest.raises(ValueError, match="unexpected runtime trace contract"):
        parity_report(native, proxy, objective.region, **kwargs)
