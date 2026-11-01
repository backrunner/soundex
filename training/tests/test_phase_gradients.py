"""P68 phase-gradient decomposition stays exact and read-only."""

from types import SimpleNamespace

import torch
from test_fused_spectrum import batch_for

from models.band_phase import BandPhaseCorrection
from models.band_tcn import make_prototype
from scripts.audit_phase_gradients import cosine, probe


def test_phase_and_role_chains_reconstruct_without_updating_weights():
    batch = batch_for(0.01)
    batch["identity"] = torch.tensor([False, True])
    batch["cutoff"] = torch.full((2, 160), 14000.0, dtype=torch.float64)
    model = BandPhaseCorrection(make_prototype("band", seed=68), phase_support="native").double()
    with torch.no_grad():
        model.phase_head[-1].bias.fill_(0.2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    context = torch.zeros(2, 160, 1538, dtype=torch.float64)
    result = probe(model, (SimpleNamespace(batch=batch), context), ["music", "speech"])
    for key in ("gradient_reconstruction_max", "role_reconstruction_max", "phase_chain_max"):
        assert result[key] < 1e-10
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert result["parameter_norms"]["fused"] > 0
    assert cosine(torch.zeros(3), torch.ones(3)) is None
    for value in result["output_phase_gradient_power"].values():
        assert abs(value["all_hf"] - value["available_hf"] - value["discarded_hf"]) < 1e-12
