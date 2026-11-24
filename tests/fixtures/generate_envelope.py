"""Generate the synthetic envelope-probe `.sxenv` and its PyTorch reference outputs.

Seeded random weights with hidden width 8: no learned parameters or audio. The
JSON holds a fixed 12-frame input sequence and the per-frame outputs/state of
`torch.nn.GRU`, for the native Rust parity test (crates/soundex-core/tests).
Run from the repository root: `python tests/fixtures/generate_envelope.py`.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from export_envelope import write  # noqa: E402
from models.envelope_predictor import EnvelopePredictor  # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    torch.manual_seed(20261124)
    model = EnvelopePredictor("band", hidden=8, layers=2).eval()
    with torch.no_grad():
        model.out.bias.fill_(-1.5)
    write(
        model,
        "deficit",
        OUT / "envelope-probe.sxenv",
        warmup_frames=4,
        provenance={"synthetic": "seeded random weights; not a restoration model"},
    )
    x = torch.randn(1, 12, 131) * 0.5
    x[0, :, 129] = -0.4  # loudness reference / 10
    x[0, :, 130] = 1.0  # 48 kHz flag
    with torch.no_grad():
        y, state = model(x)
    (OUT / "envelope-probe.json").write_text(
        json.dumps(
            {
                "inputs": x[0].tolist(),
                "outputs": y[0].tolist(),
                "final_state": state[:, 0].reshape(-1).tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
