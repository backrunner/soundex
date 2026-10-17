"""Actual diagnostic Rust frontend/ORT replay; requires its release binary."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.head_export import cartesian, export_graph
from evaluation.rust_bridge import read_sxa, write_sxa
from models.envelope_probe import context_features
from models.restoration_head import RestorationHead
from scripts.audit_context_deployment import unpack_trace
from scripts.bench_context_deployment import summarize

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "target/release/soundex-context-replay"
if not BINARY.exists():
    BINARY = REPO / "target/debug/soundex-context-replay"


@pytest.mark.skipif(
    not BINARY.exists(), reason="build soundex-context-replay for native integration"
)
@pytest.mark.parametrize("rate", [44100, 48000])
def test_real_context_head_preserves_channels_clock_and_past_context(tmp_path, rate):
    torch.manual_seed(120)
    model = RestorationHead(
        "complex", width=8, residual_slope=1 / 64, generation_budget=True
    ).eval()
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.02)
    path = tmp_path / "probe.onnx"
    export_graph(model, path)
    x = (0.2 * np.sin(np.arange(128 * 96) * 0.17)).astype(np.float32)
    stereo = np.stack((np.zeros_like(x), x), axis=1)
    outputs = []
    for label, pcm, channels in (("mono", x, 1), ("stereo", stereo, 2)):
        src, dst, report = [
            tmp_path / f"{label}.{suffix}" for suffix in ("in.sxa", "out.sxa", "json")
        ]
        write_sxa(src, pcm, rate)
        subprocess.run(
            [str(BINARY), str(path), str(src), str(dst), str(report), "1", "trace"], check=True
        )
        audio, actual_rate = read_sxa(dst)
        assert actual_rate == rate
        assert audio.shape == (len(x), channels)
        trace = unpack_trace(json.loads(report.read_text()), channels)
        expected_context = context_features(
            torch.from_numpy(pcm.reshape(-1, channels).T.copy()), "past"
        )
        torch.testing.assert_close(trace["context"], expected_context, atol=3e-6, rtol=2e-5)
        with torch.no_grad():
            prediction, _ = model(
                trace["features"],
                trace["context"],
                trace["condition"],
                model.initial_state(channels),
            )
        np.testing.assert_allclose(
            cartesian(trace["prediction"].numpy())[:, 80:],
            cartesian(prediction.numpy())[:, 80:],
            atol=2e-5,
            rtol=1e-4,
        )
        outputs.append(audio)
    np.testing.assert_allclose(outputs[0][:, 0], outputs[1][:, 1], atol=2e-6, rtol=1e-5)
    assert np.max(np.abs(outputs[1][:, 0])) < 1e-9


def test_trace_rejects_nonfinite_and_wrong_clock():
    with pytest.raises(ValueError, match="invalid runtime report"):
        unpack_trace({"traces": [], "channels": 1, "hop_samples": 256, "finite": True})


def test_cost_separates_service_backlog_pacing_and_warmup():
    report = {
        "timings": [[0, 0, 0, 100.0]] * 100 + [[0.1, 1.8, 0.1, v] for v in (6.0, 1.0, 1.0)],
        "sample_rate": 48000,
        "timing_columns": ["frontend_ms", "ort_ms", "dsp_ms", "total_ms"],
        "load_ms": 42,
        "paced_finish_lateness_ms": [0.0] * 100 + [0.1, 0, 0],
        "qos_applied": True,
        "finite": True,
        "output_peak": 0.4,
        "active_hops": 103,
        "inference_hops": 103,
    }
    result = summarize(report, "1.20 real 0.80 user 0.20 sys\n 123456 maximum resident set size")
    assert result["measured_hops"] == 3
    assert result["stages"]["total_ms"]["over_5ms"] == 1
    assert result["stages"]["total_ms"]["service_over_hop"] == 1
    assert result["stages"]["total_ms"]["virtual_deadline_misses"] >= 2
    assert result["paced_late_hops"] == 1
    assert result["process_peak_rss_bytes"] == 123456
    assert result["cpu_user_seconds"] == 0.8
    json.dumps(result, allow_nan=False)
