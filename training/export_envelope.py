"""Export a P108/P109 EnvelopePredictor endpoint to the experimental `.sxenv` format.

Layout (read by crates/soundex-core/src/envelope.rs): b"SXENV001", u32 LE header
length, JSON header, then f32 LE tensors in PyTorch order. Experimental opt-in
artifact; the production ONNX contract is unchanged.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from models.envelope_predictor import FFT_SIZE, HOP, EnvelopePredictor

MAGIC = b"SXENV001"


def tensors(model):
    sd = model.state_dict()
    names = ["inp.0.weight", "inp.0.bias"]
    for layer in range(model.layers):
        names += [f"gru.{k}_l{layer}" for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    names += ["out.weight", "out.bias"]
    return [(n, sd[n].detach().to(torch.float32).cpu().numpy()) for n in names]


def write(model, target, path, *, warmup_frames=32, provenance=None, fill_gate_db=None):
    """Write `model` to `path`; returns the header dict."""
    if target not in ("deficit", "level"):
        raise ValueError("target must be deficit or level")
    if model.arm != "band":
        raise ValueError("only the half-ERB band arm has a native fill")
    parts = tensors(model)
    payload = b"".join(np.ascontiguousarray(t, "<f4").tobytes() for _, t in parts)
    header = {
        "format": "soundex-envelope",
        "version": 1,
        "experimental": True,
        "input": int(model.inp[0].in_features),
        "hidden": int(model.hidden),
        "layers": int(model.layers),
        "output": int(model.out.out_features),
        "activation": "gelu-erf",
        "band_layout": "half-erb-4k-24k",
        "target": target,
        "warmup_frames": int(warmup_frames),
        "fft_size": FFT_SIZE,
        "hop_size": HOP,
        "tensors": [{"name": n, "shape": list(t.shape)} for n, t in parts],
        "parameters": int(sum(t.size for _, t in parts)),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": provenance or {},
    }
    if fill_gate_db is not None:  # Absent key = no gate, as in P110 artifacts.
        header["fill_gate_db"] = [float(v) for v in fill_gate_db]
    blob = json.dumps(header, sort_keys=True).encode()
    Path(path).write_bytes(MAGIC + struct.pack("<I", len(blob)) + blob + payload)
    return header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("deficit", "level"), required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = EnvelopePredictor("band", config["hidden"], config["layers"])
    model.load_state_dict(state["model"])
    header = write(
        model.eval(),
        args.target,
        args.output,
        warmup_frames=config["warmup_frames"],
        provenance={
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "checkpoint": str(args.checkpoint),
        },
    )
    print(json.dumps({k: header[k] for k in ("parameters", "payload_sha256", "target")}))


if __name__ == "__main__":
    main()
