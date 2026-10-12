import pytest
import torch

from models.fit_diagnostic import FixedFitObjective, FreeSpectrum
from models.rendered_objective import RenderedObjective, amplitude_features
from tests.test_rendered_objective import fixture


def paired_batch(level=0.1):
    p, one = fixture(level=level, identity=False)
    batch = {k: torch.cat((v, v), dim=0) for k, v in one.items()}
    batch["identity"][1] = True
    batch["y"] = batch["x"].clone()
    return torch.cat((p, p), dim=0), batch


def test_fixed_fit_matches_p10_per_example_objectives():
    amplitude, batch = paired_batch()
    prediction = amplitude + 0.002
    fit = FixedFitObjective(batch)(amplitude_features(prediction, batch["x"]))
    expected = []
    for i in range(2):
        b = {k: v[i : i + 1] for k, v in batch.items()}
        expected.append(
            RenderedObjective(1)(prediction[i : i + 1], amplitude[i : i + 1], b)["total"]
        )
    torch.testing.assert_close(fit["total"], torch.stack(expected).mean())


@pytest.mark.parametrize("mode", ["amplitude", "complex"])
@pytest.mark.parametrize("level", [0.0, 1e-9, 0.1])
def test_free_spectra_identity_start_and_finite_gradients(mode, level):
    _, batch = paired_batch(level)
    model = FreeSpectrum(batch["x"], mode)
    objective = FixedFitObjective(batch)
    output = model()
    torch.testing.assert_close(output, model.base_features, atol=0, rtol=0)
    torch.testing.assert_close(output[:, :, :80], batch["x"][:, :, :80], atol=0, rtol=0)
    torch.testing.assert_close(objective.render(output), objective.baseline, atol=1e-8, rtol=1e-6)
    with torch.no_grad():
        model.delta.add_(1e-4)
    loss = objective(model())["total"]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.delta.grad).all()


@pytest.mark.parametrize("mode", ["amplitude", "complex"])
def test_free_spectrum_finite_difference(mode):
    _, batch = paired_batch()
    model = FreeSpectrum(batch["x"], mode)
    with torch.no_grad():
        model.delta.fill_(0.002)
    objective = FixedFitObjective(batch)
    gradient = torch.autograd.grad(objective(model())["total"], model.delta)[0]
    direction = torch.randn_like(model.delta)
    direction /= direction.norm()
    epsilon = 1e-7
    with torch.no_grad():
        initial = model.delta.clone()
        model.delta.copy_(initial + epsilon * direction)
        plus = objective(model())["total"]
        model.delta.copy_(initial - epsilon * direction)
        minus = objective(model())["total"]
    torch.testing.assert_close(
        (plus - minus) / (2 * epsilon), (gradient * direction).sum(), atol=2e-4, rtol=2e-3
    )


def test_empty_bins_keep_complex_generation_gradient():
    original = torch.zeros(1, 2, 100, 129, dtype=torch.float64)
    original[:, 0] = -200
    original[:, 1] = 0.3
    model = FreeSpectrum(original, "complex")
    spectrum = model()
    loss = spectrum[:, 0, 90:].sum() + spectrum[:, 1, 90:].sum()
    loss.backward()
    assert torch.isfinite(model.delta.grad).all()
    assert (model.delta.grad[:, 10:].abs().sum(-1) > 0).all()
