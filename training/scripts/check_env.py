#!/usr/bin/env python3
"""Validate the training environment for SoundEx.

Checks Python, PyTorch, CUDA availability, and that core modules import.
Exit code 0 = usable for training (CUDA optional but reported).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the SoundEx training environment")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    training_root = Path(__file__).resolve().parents[1]
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    print("=== SoundEx environment check ===")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    py = sys.version_info
    if py < (3, 12):
        print(f"WARN: target is Python 3.12+; found {py.major}.{py.minor}")
    elif py.major == 3 and py.minor == 12:
        print("OK: Python 3.12")
    else:
        print(f"INFO: Python {py.major}.{py.minor} (target documented as 3.12)")

    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: torch not importable: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    expected_prefix = "2.12"
    if not torch.__version__.startswith(expected_prefix):
        print(f"WARN: documented target is torch 2.12.x; found {torch.__version__}")
    else:
        print("OK: torch 2.12.x")

    # Core training imports
    missing = []
    for name in (
        "numpy",
        "yaml",
        "soundfile",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "tensorboard",
    ):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"FAIL: missing packages: {', '.join(missing)}")
        return 1
    print("OK: training and mandatory ONNX export packages import")

    # The supported preprocessing matrix depends on concrete FFmpeg encoders.
    try:
        from data.audio_prep import ffmpeg_encoders_available
        from data.protocol import load_recipe_from_profile

        recipe = load_recipe_from_profile(args.config)
        available_encoders = ffmpeg_encoders_available()
        required_encoders = {codec.encoder for codec in recipe.codecs}
        missing_encoders = sorted(required_encoders - available_encoders)
        if missing_encoders:
            print(f"FAIL: FFmpeg missing audio encoders: {', '.join(missing_encoders)}")
            return 1
        print(f"OK: FFmpeg encoders: {', '.join(sorted(required_encoders))}")
    except Exception as exc:
        print(f"FAIL: data recipe / FFmpeg check: {exc}")
        return 1

    # Optional
    for name in ("torchaudio",):
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"OK: {name} {ver}")
        except ImportError:
            print(f"INFO: {name} not installed (optional for some paths)")

    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available: {cuda_ok}")
    if cuda_ok:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        print(f"device0: {name}  capability=sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(
                "WARN: pre-Turing GPU (e.g. Pascal sm_61). Prefer cu126 (or older) "
                "wheels; CUDA 13 / cu130 builds may omit this architecture."
            )
        # Tiny compute sanity check
        try:
            x = torch.randn(64, 64, device="cuda")
            y = x @ x.T
            torch.cuda.synchronize()
            print(f"OK: CUDA matmul smoke test (result mean={y.mean().item():.4f})")
        except Exception as exc:
            print(f"FAIL: CUDA smoke test: {exc}")
            return 1
    else:
        print(
            "WARN: CUDA not available — training will fall back to CPU/MPS "
            "(very slow for full datasets)."
        )

    # Model construct on CPU
    try:
        from models.discriminator import MultiScaleDiscriminator
        from models.generator import SoundExGenerator

        g = SoundExGenerator()
        d = MultiScaleDiscriminator()
        n = g.count_parameters()
        print(f"OK: SoundExGenerator params={n:,}")
        if n > 2_000_000:
            print("WARN: parameter count exceeds 2M training guard")
        _ = d
    except Exception as exc:
        print(f"FAIL: model import/construct: {exc}")
        return 1

    print("=== check complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
