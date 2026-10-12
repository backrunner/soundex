"""Read original input PCM for past-only features, never the delayed dry tensor."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.causal_study import SAMPLES
from data.weak_band_study import IdentityClips


class EnvelopeClips(IdentityClips):
    def batch(self, specs, device="cpu"):
        batch = super().batch(specs, device)
        waves = []
        for spec in specs:
            row = self.rows[spec["row_id"]]
            wave, rate = sf.read(
                Path(row["_root"]) / row["degraded_path"],
                start=spec["start"],
                frames=SAMPLES,
                dtype="float32",
                always_2d=True,
            )
            if rate != row["sample_rate"] or wave.shape[1] != 1 or not np.isfinite(wave).all():
                raise ValueError("invalid envelope input PCM")
            wave = wave[:, 0] * (10 ** (spec["gain_db"] / 20))
            waves.append(torch.from_numpy(np.pad(wave, (0, SAMPLES - len(wave)))))
        # FFT frontend stays on CPU; MPS FFT is not needed for training.
        batch["pcm"] = torch.stack(waves)
        return batch
