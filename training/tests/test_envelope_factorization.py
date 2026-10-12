import torch

from models.envelope_objective import BANDS
from scripts.audit_envelope_factorization import factor_oracles


def test_oracle_changes_only_requested_factor_and_preserves_low_band():
    torch.manual_seed(5)
    p, t = torch.rand(1, 7, 129), torch.rand(1, 7, 129)
    result = factor_oracles(p, t, 48000)
    frequency = torch.arange(129) * 48000 / 256
    for name in ("oracle_band_envelope", "oracle_within_band_shape"):
        torch.testing.assert_close(result[name][..., frequency < 4000], p[..., frequency < 4000])
    for low, high in BANDS[1:]:
        mask = (frequency >= low) & (frequency < high)
        a, b = p[..., mask], t[..., mask]
        envelope, shape = (
            result["oracle_band_envelope"][..., mask],
            result["oracle_within_band_shape"][..., mask],
        )
        torch.testing.assert_close(envelope.norm(dim=-1), b.norm(dim=-1))
        torch.testing.assert_close(shape.norm(dim=-1), a.norm(dim=-1))
        torch.testing.assert_close(
            envelope / envelope.norm(dim=-1, keepdim=True), a / a.norm(dim=-1, keepdim=True)
        )
        torch.testing.assert_close(
            shape / shape.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
        )
    silent = factor_oracles(torch.zeros_like(p), torch.zeros_like(t), 44100)
    assert all(torch.isfinite(v).all() and v.count_nonzero() == 0 for v in silent.values())
