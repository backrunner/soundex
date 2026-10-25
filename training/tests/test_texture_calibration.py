import pytest
import torch

from evaluation.texture_calibration import (
    aggregate,
    assess,
    controls,
    isolated_draws,
    pcm_diagnostics,
)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_false_detail_controls_preserve_energy_and_low_band(rate):
    torch.manual_seed(59)
    reference = torch.randn(2, 4096, dtype=torch.float64) * 0.03
    rates = torch.tensor([rate, rate])
    variants = controls(reference * 0.25, reference, reference * 0.5, rates, 59)
    for name in ("phase_scrambled", "time_shifted"):
        value = variants[name]
        torch.testing.assert_close(
            torch.fft.rfft(value).abs(), torch.fft.rfft(reference).abs(), atol=1e-11, rtol=1e-11
        )
        assert not torch.allclose(value, reference)
        d = pcm_diagnostics(value, reference, rates)
        assert d["band0_error_rms"].max() < 1e-15
    matched = pcm_diagnostics(variants["energy_matched"], reference, rates)
    for low in (8000, 12000, 16000):
        torch.testing.assert_close(
            matched[f"band{low}_candidate_rms"], matched[f"band{low}_reference_rms"]
        )
    again = controls(reference * 0.25, reference, reference * 0.5, rates, 59)
    assert torch.equal(again["phase_scrambled"], variants["phase_scrambled"])


def test_silence_and_missing_generation_remain_finite():
    wave = torch.zeros(2, 2048)
    variants = controls(wave, wave, wave, torch.tensor([44100, 48000]), 0)
    assert all(torch.equal(v, wave) for v in variants.values())
    reference = torch.randn_like(wave) * 1e-8
    variants = controls(wave, reference, wave, torch.tensor([44100, 48000]), 0)
    assert all(torch.isfinite(v).all() for v in variants.values())
    assert torch.equal(variants["energy_matched"], wave)


def test_band_rms_is_calibrated_pcm_not_fft_power():
    n = 4096
    wave = torch.sin(2 * torch.pi * 1024 * torch.arange(n, dtype=torch.float64) / n)[None]
    d = pcm_diagnostics(wave, wave * 0, torch.tensor([48000]))
    assert float(d["band12000_error_rms"][0]) == pytest.approx(2**-0.5)
    assert float(d["pcm_error_rms"][0]) == pytest.approx(2**-0.5)


def test_source_equal_aggregation_and_shortcut_rejection():
    records = []
    names = ("prediction", "energy_matched", "phase_scrambled", "time_shifted", "roundtrip")
    for corpus in ("music_library", "speech_library"):
        for critic in ("initial", "trained"):
            for name in names:
                for source, count in (("a", 3), ("b", 1)):
                    for _ in range(count):
                        records.append(
                            {
                                "generator": "g",
                                "critic": critic,
                                "role": "calibration",
                                "corpus": corpus,
                                "control": name,
                                "source": source,
                                "scores": {
                                    "margin": 0
                                    if name == "roundtrip"
                                    else (1 if source == "a" else 3),
                                    "balanced_logistic_loss": 1.5 if critic == "initial" else 1.0,
                                },
                            }
                        )
    groups = aggregate(records)
    assert groups["g/trained/calibration/music_library/prediction"]["source_mean"]["margin"] == 2
    assert assess(groups, ["g"])["prerequisite_passes"]
    # Prediction classification alone does not pass when an energy-matched
    # false-detail control fools the critic on one corpus.
    target = groups["g/trained/calibration/music_library/energy_matched"]
    for row in target["per_source"].values():
        row["margin"] = -1
    assert not assess(groups, ["g"])["prerequisite_passes"]


def test_source_filter_excludes_connected_aliases_and_preserves_whole_draws():
    rows = {
        "a": {"source_checksum": "a"},
        "b": {"source_checksum": "b", "work_id": "work"},
        "c": {"source_checksum": "c"},
        "d": {"source_checksum": "d"},
    }
    bridge = {"source_checksum": "a", "work_id": "work"}
    source = {"specs": [{"row_id": x} for x in rows], "draws": [[0, 2], [1, 3], [2, 3]]}
    assert isolated_draws(source, rows, [*rows.values(), bridge], [rows["a"]]) == [2]
