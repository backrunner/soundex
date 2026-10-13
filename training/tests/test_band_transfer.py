import torch

from evaluation.band_transfer import (
    analyze,
    band_masks,
    complex_features,
    intervene,
    synthesize,
    transfer,
)
from models.causal_objective import overlap_add
from models.rendered_objective import amplitude_features, reanalyze
from tests.test_fit_diagnostic import paired_batch


def test_complex_operators_match_existing_synthesis_and_analysis():
    amplitude, batch = paired_batch()
    features = amplitude_features(amplitude + 0.002, batch["x"])
    wave = synthesize(complex_features(features))
    torch.testing.assert_close(wave, overlap_add(features)[:, : wave.shape[1]], atol=0, rtol=0)
    torch.testing.assert_close(
        analyze(wave).abs().clamp_min(1e-10), reanalyze(wave), atol=0, rtol=0
    )


def test_band_interventions_partition_both_sample_rates_without_touching_others():
    prediction = torch.randn(2, 2, 100, 129, dtype=torch.float64)
    initial = torch.randn_like(prediction)
    masks = band_masks(
        torch.tensor([[44100], [48000]]), dtype=prediction.dtype, device=prediction.device
    )
    torch.testing.assert_close(masks.sum(1), torch.ones(2, 129, dtype=prediction.dtype))
    residuals = []
    for i in range(5):
        only = intervene(prediction, initial, masks, i)
        removed = intervene(prediction, initial, masks, i, remove=True)
        torch.testing.assert_close(only + removed, prediction + initial)
        residuals.append(only - initial)
    torch.testing.assert_close(sum(residuals), prediction - initial)


def test_zero_residual_and_complex_superposition():
    amplitude, batch = paired_batch()
    initial = amplitude_features(amplitude, batch["x"])
    identity = transfer(initial, initial, batch)
    assert torch.count_nonzero(identity["matrix"]) == 0
    generator = torch.Generator().manual_seed(20261013)
    prediction = amplitude_features(
        amplitude + torch.rand(amplitude.shape, generator=generator, dtype=amplitude.dtype) * 0.002,
        batch["x"],
    )
    result = transfer(prediction, initial, batch)
    assert result["wave_superposition_max_error"].max() < 1e-12
    assert result["spectrum_superposition_max_error"].max() < 1e-12
    assert torch.isfinite(result["matrix"]).all()
    off_diagonal = result["matrix"] * (1 - torch.eye(5))
    assert off_diagonal.max() > 0
