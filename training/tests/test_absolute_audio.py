import numpy as np
import pytest

from evaluation.absolute_audio import band_powers, compare_pcm, pcm_stats


@pytest.mark.parametrize("rate", (44100, 48000))
def test_periodogram_units_and_band_placement(rate):
    size = 4096
    frequency = 1200 * rate / size
    wave = np.sin(2 * np.pi * frequency * np.arange(size * 4) / rate)
    powers = band_powers(wave, rate)
    assert sum(powers.values()) == pytest.approx(0.5, abs=1e-12)
    assert powers["12000_16000"] == pytest.approx(0.5, abs=1e-12)
    assert pcm_stats(wave)["rms_dbfs"] == pytest.approx(-3.01029995664)
    assert sum(band_powers(np.ones(size * 2), rate).values()) == pytest.approx(1.0)


def test_residual_and_silence_keep_absolute_scale():
    wave = np.sin(2 * np.pi * 1000 * np.arange(16384) / 48000)
    result = compare_pcm(wave * 0.501, wave * 0.5, 48000)
    assert result["error"]["error_to_reference_rms"] == pytest.approx(0.002)
    identical = compare_pcm(wave, wave, 48000)
    assert identical["error"]["rms"] == 0
    assert identical["error"]["rms_dbfs"] is None
    silent = compare_pcm(np.zeros(8192), np.zeros(8192), 48000)
    assert silent["error"]["error_to_reference_rms"] is None


def test_invalid_audio_rejected():
    with pytest.raises(ValueError):
        pcm_stats([float("nan")])
    with pytest.raises(ValueError):
        compare_pcm(np.zeros(8192), np.zeros(8193), 48000)
    with pytest.raises(ValueError):
        band_powers(np.zeros(10), 48000)
