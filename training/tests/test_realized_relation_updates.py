"""P83 finite persistent-Adam and exact score/exposure controls."""

import copy

import pytest

from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import (
    canonical_model_state,
    canonical_optimizer_state,
    continued,
    stepper,
)
from evaluation.realized_relation_updates import auxiliary, draws_for_step, fit_scores, scores_v2
from models.restoration_head import predict_budget
from scripts.audit_phase_routing import continued as canonical_continued
from scripts.audit_phase_routing import exact, native, scores
from tests.test_phase_update_coordinates import saved_fixture


@pytest.mark.parametrize("arm", ["direct", "raw", "fused"])
def test_new_auxiliary_preserves_coordinate_equivalent_persistent_adam(arm):
    saved, items = saved_fixture()
    original_state = copy.deepcopy(saved)
    a, oa = canonical_continued(saved)
    b, ob = continued(saved, 1)
    for _ in range(2):
        ra = update(a, items, oa, arm, auxiliary_fn=auxiliary)
        rb = update(b, items, ob, arm, auxiliary_fn=auxiliary, adam_step=stepper(b))
        assert ra["gradient_norm"] == rb["gradient_norm"]
        assert ra["changed_parameters"] > 0
        exact(a.state_dict(), canonical_model_state(b))
        exact(oa.state_dict(), canonical_optimizer_state(b, ob, saved["optimizer"]))
    exact(saved, original_state)
    assert all(int(s["step"]) == 3 for s in oa.state_dict()["state"].values())


def test_every_registered_crop_has_eight_exposures_without_transfer_draws():
    fit, transfer = [2, 3, 4, 5, 6, 7, 9, 11], {0, 1, 8, 10}
    draws = [i for step in range(16) for i in draws_for_step(fit, step)]
    assert not set(draws) & transfer
    assert len(draws) == 64 and all(draws.count(i) == 8 for i in fit)
    assert draws[:8] == fit
    for step in (-1, 16):
        with pytest.raises(ValueError, match="budget"):
            draws_for_step(fit, step)


def test_extended_scores_and_common_fit_risk_are_not_reweighted():
    saved, items = saved_fixture()
    model, _ = continued(saved, 1)
    batch = items[0][0].batch
    prediction, _ = predict_budget(model, batch, items[0][1])
    values = fit_scores(prediction, batch)
    assert len(values) == 2 and all(v["common_risk"] >= 0 for v in values)
    wave = batch["clean_wave"]
    old, new = scores(wave, wave, batch["rate"]), scores_v2(wave, wave, batch["rate"])
    for a, b in zip(old, new, strict=True):
        assert all(b[k] == v for k, v in a.items())
        assert b["hf_spectral_dynamics"] == 0 and b["full_pcm_error_rms"] == 0


def test_native_trace_roles_fail_before_launch_if_out_of_range(tmp_path):
    with pytest.raises(ValueError, match="trace role"):
        native(None, None, {"pcm": [0]}, tmp_path, False, trace_roles=(1,))
