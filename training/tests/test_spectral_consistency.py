import pytest
import torch

from models.rendered_objective import amplitude_features
from models.spectral_consistency import consistency_loss, reference_features
from tests.test_fit_diagnostic import paired_batch


@pytest.mark.parametrize("level", [0.0, 1e-9, 0.1])
def test_reference_has_zero_risk_and_finite_zero_gradient(level):
    _, batch = paired_batch(level)
    reference = reference_features(batch).requires_grad_()
    loss = consistency_loss(reference, batch)
    assert float(loss.detach()) == 0
    gradient = torch.autograd.grad(loss, reference)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("level", [0.0, 1e-9, 0.1])
def test_nonzero_correction_has_finite_gradient_and_finite_difference(level):
    amplitude, batch = paired_batch(level)
    perturbation = torch.linspace(-0.002, 0.003, amplitude.shape[1], dtype=amplitude.dtype)[
        None, :, None
    ]
    prediction = (amplitude + perturbation.abs()).requires_grad_()

    def objective(value):
        return consistency_loss(amplitude_features(value, batch["x"]), batch)

    loss = objective(prediction)
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    direction = torch.ones_like(prediction) / prediction.numel() ** 0.5
    eps = 1e-8 if level == 0.1 else 1e-10
    finite = (
        objective(prediction.detach() + eps * direction)
        - objective(prediction.detach() - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close(finite, (gradient * direction).sum(), rtol=5e-3, atol=1e-4)


def test_target_transient_is_not_penalized_as_smoothing_error():
    _, batch = paired_batch()
    # The paired clean target has a sharp time-varying high-band change.
    batch["y"][0, 0, 95:105, 70:] += 12
    reference = reference_features(batch).requires_grad_()
    assert consistency_loss(reference, batch) == 0
    baseline = amplitude_features(torch.pow(10.0, batch["x"][:, 0] / 20), batch["x"])
    assert consistency_loss(baseline, batch) > 0
