import numpy as np
import pytest

from evaluation import envelope_oracle as oracle
from evaluation.missing_band import calibrated_stft, missing_mask, missing_region, score

RATE = 44100


def lowpass(x, hz):
    spec = np.fft.rfft(x, axis=0)
    spec[np.fft.rfftfreq(len(x), 1 / RATE) >= hz] = 0
    return np.fft.irfft(spec, len(x), axis=0)


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    n = RATE * 2
    # Amplitude-modulated broadband noise: real temporal variation in every band.
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * np.arange(n) / RATE) ** 2
    ref = 0.05 * rng.standard_normal((n, 2)) * envelope[:, None]
    return ref, lowpass(ref, 16000)


def test_parseval_units():
    wave = np.sin(2 * np.pi * 1000 * np.arange(RATE) / RATE)[:, None]
    x, _ = calibrated_stft(wave, RATE)
    assert np.square(np.abs(x)).sum(-1).mean() == pytest.approx(0.5, rel=1e-3)


def test_correct_reference_scores_zero(pair):
    ref, dry = pair
    s = score(ref, ref, dry, RATE)
    assert s["missing_lsd_db"] == 0
    assert s["missing_erb_rmse_db"] == 0
    assert s["missing_erb_bias_db"] == 0
    assert s["missing_temporal_corr"] == pytest.approx(1)
    assert s["preserved_complex_rel"] == 0
    assert s["mask_cutoff_hz"] == pytest.approx(16000, abs=50)


def test_untreated_input_is_far_too_quiet(pair):
    ref, dry = pair
    s = score(dry, ref, dry, RATE)
    assert s["missing_erb_bias_db"] < -60
    # Brick-wall leakage into the 2048-point edge bins dominates total energy.
    assert s["missing_energy_ratio_db"] < -30
    assert s["preserved_complex_rel"] < 0.05


def test_gain_only_change_is_pure_bias(pair):
    ref, dry = pair
    high = ref - dry
    s = score(dry + high * 10 ** (-6 / 20), ref, dry, RATE)
    # Edge leakage of the untouched retained band keeps it within 0.02 dB of -6.
    assert s["missing_erb_bias_db"] == pytest.approx(-6, abs=0.02)
    assert s["missing_erb_rmse_db"] == pytest.approx(6, abs=0.02)
    assert s["missing_temporal_corr"] == pytest.approx(1, abs=1e-3)


def test_no_missing_band_is_undefined_not_zero(pair):
    ref, _ = pair
    s = score(ref, ref, ref, RATE)
    assert s["missing_lsd_db"] is None and s["missing_erb_bias_db"] is None
    assert s["preserved_lsd_db"] == 0 and s["mask_missing_bins"] == 0


def test_silence_is_finite_and_undefined():
    silent = np.zeros((RATE, 2))
    s = score(silent, silent, silent, RATE)
    assert s["missing_lsd_db"] is None
    assert all(v is None or np.isfinite(v) for v in s.values())


def test_weak_reference_band_ignored_by_mask():
    x, hz = calibrated_stft(np.zeros((8192, 1)), RATE)
    p = np.square(np.abs(x)) + 1e-40
    assert not missing_mask(p, p, hz).any()


def test_stft_roundtrip_is_exact(pair):
    ref, _ = pair
    assert np.allclose(oracle.istft(oracle.stft(ref), len(ref)), ref, atol=1e-12)


def test_copy_up_keeps_hop_phase_coherent():
    # A 10-bin translation of a bin-centred sine is again a clean sine.
    k = 40
    n = RATE
    wave = np.sin(2 * np.pi * k * RATE / 256 * np.arange(n) / RATE)[:, None]
    hz = np.fft.rfftfreq(256, 1 / RATE)
    x = oracle.stft(wave)[0]
    shifted = oracle.copy_up(x, k + 10, hz)
    # Source band [lo, k+10) maps bin k to k + width; recover only that output bin.
    spec = np.zeros_like(x)
    target = k + 10 + (k - int(np.searchsorted(hz, oracle.PATCH_FLOOR_HZ)))
    spec[:, target] = shifted[:, target]
    out = oracle.istft(spec[None], n)[:, 0][1024:-1024]
    expected = target * RATE / 256
    fit = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / RATE)[np.argmax(fit)]
    assert peak == pytest.approx(expected, abs=3)


def test_oracle_ordering(pair):
    ref, dry = pair
    scores = {k: score(oracle.render(dry, ref, RATE, k), ref, dry, RATE) for k in oracle.ORACLES}
    untreated = score(dry, ref, dry, RATE)
    # The 172 Hz deployment grid leaves one partially attenuated edge bin dry.
    assert scores["oracle_complex"]["missing_lsd_db"] < 2.0
    assert abs(scores["oracle_complex"]["missing_erb_bias_db"]) < 0.5
    for kind in oracle.ORACLES:
        assert scores[kind]["missing_lsd_db"] < untreated["missing_lsd_db"] / 5
        # 256-point edits spill slightly below the 2048-point mask edge.
        assert scores[kind]["preserved_complex_rel"] <= untreated["preserved_complex_rel"] + 1e-2
    for kind in ("oracle_patch_level", "oracle_patch_erb", "oracle_patch_binmag"):
        assert abs(scores[kind]["missing_erb_bias_db"]) < 2
    # Independent random-phase frames are STFT-inconsistent: overlap-add loses ~3 dB.
    assert -4 < scores["oracle_noise_erb"]["missing_erb_bias_db"] < -2
    assert scores["oracle_patch_erb"]["missing_temporal_corr"] > 0.95


def test_missing_region_ignores_floor_bins_above_the_cutoff():
    # P107 v1 counterexample: reference and input both at the noise floor near
    # Nyquist made the old whole-tail rule return no region at all.
    row = np.zeros(129, bool)
    row[90:118] = True
    row[100] = False
    assert missing_region(row) == (90, 118)
    assert missing_region(np.zeros(129, bool)) is None
