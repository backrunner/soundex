import pytest
import torch

from evaluation.phase_swap import VARIANTS, complex_error_parts, high_frequency_swaps


def fixture():
    torch.manual_seed(30)
    prediction = torch.randn(2, 2, 100, 129, dtype=torch.float64)
    prediction[:, 0] -= 20
    x, y = prediction.clone(), prediction.clone()
    y[:, 0] += 6
    y[:, 1] += 0.7
    return prediction, {
        "x": x,
        "y": y,
        "rate": torch.tensor([[44100], [48000]]),
        "identity": torch.tensor([0, 1]),
    }


def test_swaps_only_replace_registered_component_hf_and_post_warmup():
    p, batch = fixture()
    before = p.clone()
    result = high_frequency_swaps(p, batch)
    assert tuple(result) == VARIANTS
    hz = torch.arange(129)[None] * batch["rate"] / 256
    for name, components in (
        ("reference_phase", (1,)),
        ("reference_amplitude", (0,)),
        ("reference_high", (0, 1)),
    ):
        value = result[name]
        torch.testing.assert_close(value[:, :, :80], p[:, :, :80], atol=0, rtol=0)
        for i in range(2):
            torch.testing.assert_close(
                value[i, :, :, hz[i] < 4000], p[i, :, :, hz[i] < 4000], atol=0, rtol=0
            )
            for component in range(2):
                target = batch["y"] if i == 0 and component in components else p
                torch.testing.assert_close(
                    value[i, component, 80:, hz[i] >= 4000],
                    target[i, component, 80:, hz[i] >= 4000],
                    atol=0,
                    rtol=0,
                )
    torch.testing.assert_close(p, before, atol=0, rtol=0)


@pytest.mark.parametrize("phase", [0.0, 0.3, 3.141592653589793, 2 * 3.141592653589793])
def test_complex_error_decomposition_is_exact_and_wrap_safe(phase):
    p, batch = fixture()
    target = p.clone()
    target[:, 0] += 3
    target[:, 1] += phase
    rows = complex_error_parts(p, target, batch["rate"])
    for row in rows:
        for band in row.values():
            assert band["decomposition_abs_error"] < 1e-14
            assert band["phase_power"] >= 0 and band["amplitude_power"] > 0
            if phase in (0.0, 2 * 3.141592653589793):
                assert band["phase_power"] < 1e-20


def test_reference_swaps_remove_only_the_replaced_component_error():
    p, batch = fixture()
    variants = high_frequency_swaps(p, batch)
    phase = complex_error_parts(variants["reference_phase"], batch["y"], batch["rate"])[0][
        "4000_24001"
    ]
    amplitude = complex_error_parts(variants["reference_amplitude"], batch["y"], batch["rate"])[0][
        "4000_24001"
    ]
    both = complex_error_parts(variants["reference_high"], batch["y"], batch["rate"])[0][
        "4000_24001"
    ]
    assert phase["phase_power"] == 0 and phase["amplitude_power"] > 0
    assert amplitude["amplitude_power"] < 1e-30 and amplitude["phase_power"] > 0
    assert both["complex_power"] == 0


def test_silent_components_are_finite_and_short_input_rejected():
    p, batch = fixture()
    p[:, 0] = -200
    rows = complex_error_parts(p, p, batch["rate"])
    assert all(v["complex_power"] == 0 for row in rows for v in row.values())
    with pytest.raises(ValueError, match="insufficient"):
        high_frequency_swaps(
            p[:, :, :90], {**batch, "x": batch["x"][:, :, :90], "y": batch["y"][:, :, :90]}
        )
