import pytest
import torch

from evaluation.parameter_factorial import GROUPS, hybrid, shapley


def test_complete_partition_retains_both_endpoints_and_rejects_unknown_state():
    old = {p + "weight": torch.tensor([1.0]) for p in GROUPS}
    routed = {k: v + 2 for k, v in old.items()}
    for mask in range(8):
        result = hybrid(old, routed, mask)
        for i, p in enumerate(GROUPS):
            assert torch.equal(
                result[p + "weight"], (routed if mask & (1 << i) else old)[p + "weight"]
            )
            result[p + "weight"].zero_()
        assert all(v.item() == 1 for v in old.values())
    with pytest.raises(ValueError, match="unpartitioned"):
        hybrid({**old, "other": torch.ones(1)}, {**routed, "other": torch.ones(1)}, 0)


def test_interactions_are_shared_without_selecting_the_best_subset():
    # f=10+2B-3H+5R+8BH: the two-factor interaction splits equally.
    values = [
        10 + 2 * bool(m & 1) - 3 * bool(m & 2) + 5 * bool(m & 4) + 8 * bool(m & 1 and m & 2)
        for m in range(8)
    ]
    result = shapley(values)
    assert [result[p] for p in GROUPS] == pytest.approx([6, 1, 5])
    assert result["nonadditivity"] == 8
    values[3] = None
    assert all(v is None for v in shapley(values).values())
