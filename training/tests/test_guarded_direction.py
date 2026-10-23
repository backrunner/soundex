"""Known-answer and degeneracy checks for diagnostic gradient constraints."""

import pytest
import torch

from models.guarded_direction import project_direction
from scripts.audit_guarded_transfer import transfer_index
from scripts.summarize_guarded_transfer import unique_seconds


def test_known_projection_and_idempotence() -> None:
    proposal = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    guards = torch.eye(3, dtype=torch.float64)
    result, report = project_direction(proposal, guards)
    torch.testing.assert_close(result, torch.tensor([0.0, -3.0, 0.0], dtype=torch.float64))
    repeated, _ = project_direction(result, guards)
    torch.testing.assert_close(repeated, result)
    assert report["distance_squared"] == pytest.approx(20)
    assert report["active_guard_indices"] == [0, 2]


def test_opposing_guards_allow_tangent_and_block_infeasible_progress() -> None:
    guards = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    result, _ = project_direction(torch.tensor([1.0, 2.0], dtype=torch.float64), guards)
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0], dtype=torch.float64))
    result, report = project_direction(torch.tensor([1.0, 0.0], dtype=torch.float64), guards)
    assert torch.equal(result, torch.zeros_like(result))
    assert report["degenerate"]


def test_duplicate_zero_and_scaled_guards_do_not_change_solution() -> None:
    proposal = torch.tensor([1.0, 2.0], dtype=torch.float64)
    guards = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    expected, _ = project_direction(proposal, guards)
    redundant = torch.cat((guards.flip(0) * 1e-12, guards * 1e12, torch.zeros(1, 2)))
    result, report = project_direction(proposal, redundant)
    torch.testing.assert_close(result, expected)
    assert report["zero_guards"] == 1


def test_feasible_proposal_and_all_zero_guards() -> None:
    proposal = torch.tensor([-1.0, -2.0])
    for guards in (torch.eye(2), torch.zeros(2, 2)):
        result, report = project_direction(proposal, guards)
        assert torch.equal(result, proposal)
        assert report["distance_squared"] == 0


def test_projection_satisfies_variational_optimality() -> None:
    rng = torch.Generator().manual_seed(20261023)
    for _ in range(10):
        guards = torch.randn(5, 11, generator=rng, dtype=torch.float64)
        proposed = torch.randn(11, generator=rng, dtype=torch.float64)
        projected, report = project_direction(proposed, guards)
        assert (guards @ projected).max() < 1e-8
        # The origin and every positive multiple of a feasible projection belong
        # to this cone. Projection residual is orthogonal to the projected ray.
        assert abs(float((proposed - projected).dot(projected))) < 1e-8
        assert report["complementarity_float64"] <= 1e-10


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        project_direction(torch.tensor([bad]), torch.ones(1, 1))


def test_transfer_selection_rejects_source_aliases_before_scoring() -> None:
    rows = {
        "a": {"track_id": "a", "source_checksum": "one"},
        "b": {"track_id": "b", "source_checksum": "two"},
        "alias": {"track_id": "alias", "source_checksum": "one"},
        "c": {"track_id": "c", "source_checksum": "three"},
    }
    schedule = {"specs": [{"row_id": k} for k in rows], "draws": [[0], [1], [2], [3]]}
    assert transfer_index(schedule, 0, rows) == 3
    rows["c"]["track_id"] = "a"
    assert transfer_index(schedule, 0, rows) == 1
    rows["b"]["source_checksum"] = "one"
    with pytest.raises(ValueError, match="no recording/source-disjoint"):
        transfer_index(schedule, 0, rows)


def test_coverage_uses_physical_support_and_unions_duplicate_views() -> None:
    row = {"source_checksum": "same", "sample_rate": 48000, "start_sample": 1000}
    sample = {"row": row, "spec": {"start": 2000}}
    assert unique_seconds([sample, sample], 88, 254) == pytest.approx(166 * 128 / 48000)
    assert unique_seconds([sample], 88, 256) == pytest.approx(168 * 128 / 48000)
    shifted = {"row": row, "spec": {"start": 2000 + 128}}
    assert unique_seconds([sample, shifted], 88, 254) == pytest.approx(167 * 128 / 48000)
