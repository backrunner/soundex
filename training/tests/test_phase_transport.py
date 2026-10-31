"""Independent controls for phase-only support accounting."""

import numpy as np
import pytest

from evaluation.phase_transport import correction_support


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1.0])
def test_parseval_and_effective_partition(scale):
    x = np.zeros((4, 2, 129))
    x[:, 0] = -200
    if scale:
        x[:, 0, 64] = 20 * np.log10(128 * scale)
    y = x.copy()
    y[:, 1, 64] = 0.2 if scale else 0
    w = np.zeros((4, 129))
    w[2, 64] = 0.5
    w[3, 64] = 1
    result = correction_support(x, y, w, [False, True, True, True], 48000, burn=0)
    expected = 2 * scale**2 * np.sin(0.1) ** 2
    assert result["full"]["windowed_frame_power"] == pytest.approx(
        expected, rel=64 * np.finfo(float).eps, abs=1e-30
    )
    for share in result["full"]["shares"].values():
        assert share == (pytest.approx(0.25) if scale else None)


def test_no_correction_and_wrong_magnitude_or_bypass_fail_closed():
    x = np.zeros((2, 2, 129))
    x[:, 0] = -200
    w = np.zeros((2, 129))
    assert (
        correction_support(x, x, w, [False, True], 44100, burn=0)["full"]["windowed_frame_power"]
        == 0
    )
    changed = x.copy()
    changed[1, 0, 64] += 1
    with pytest.raises(ValueError, match="identical"):
        correction_support(x, changed, w, [False, True], 44100, burn=0)
    w[0, 1] = 1
    with pytest.raises(ValueError, match="effective"):
        correction_support(x, x, w, [False, True], 44100, burn=0)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_native_stage_observation_preserves_output_and_channels(tmp_path, rate):
    import subprocess
    from pathlib import Path

    import torch

    from evaluation.adapter_runtime import environment
    from evaluation.head_export import export_graph
    from evaluation.rust_bridge import read_sxa, write_sxa
    from models.band_tcn import make_prototype
    from scripts.run_consistency_followup import read

    binary = Path(__file__).resolve().parents[2] / "target/release/soundex-context-replay"
    if not binary.exists():
        pytest.skip("build the stage-enabled native replay for integration")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        graph = tmp_path / "band.onnx"
        export_graph(make_prototype("band", nonzero=True), graph)
        signal = (0.03 * np.sin(np.arange(12800) * 0.17)).astype(np.float32)
        waves = []
        for name, pcm in (
            ("mono", signal[:, None]),
            ("stereo", np.stack((np.zeros_like(signal), signal), 1)),
        ):
            source = tmp_path / f"{name}.sxa"
            write_sxa(source, pcm, rate)
            outputs = []
            for mode in ("", "stages"):
                path = tmp_path / f"{name}-{mode or 'plain'}"
                subprocess.run(
                    [
                        str(binary),
                        str(graph),
                        str(source),
                        str(path.with_suffix(".sxa")),
                        str(path.with_suffix(".json")),
                        "1",
                        mode,
                    ],
                    env=environment(),
                    capture_output=True,
                    check=True,
                )
                output, _ = read_sxa(path.with_suffix(".sxa"))
                outputs.append(output)
                if mode:
                    traces = read(path.with_suffix(".json"))["traces"]
                    assert len(traces) == 100
                    for channel in range(pcm.shape[1]):
                        prelimit = np.asarray([t["dsp"][channel]["phase_ola"] for t in traces])
                        limited = np.asarray([t["dsp"][channel]["wet_limited"] for t in traces])
                        np.testing.assert_array_equal(prelimit, limited)
            np.testing.assert_array_equal(*outputs)
            waves.append(outputs[0])
        assert not np.any(waves[1][:, 0])
        np.testing.assert_allclose(waves[0][:, 0], waves[1][:, 1], atol=2e-6, rtol=1e-5)
    finally:
        torch.set_num_threads(previous)
