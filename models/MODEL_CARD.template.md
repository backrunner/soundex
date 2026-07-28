# SoundEx Model Card

> Copy this file next to a weight artifact, e.g. `models/soundex-v1.model-card.md`.  
> Fill every field. Delete instructional blockquotes before release.

## 1. Identification

| Field | Value |
|-------|--------|
| **Model name** | SoundEx … |
| **Version** | e.g. `v1.0.0` |
| **Artifact file(s)** | e.g. `soundex-v1.onnx` |
| **SHA-256** | *(hex digest)* |
| **Format** | ONNX / PyTorch checkpoint / other |
| **Checkpoint / artifact schema** | `soundex-checkpoint 1.2` / `soundex artifact 1.2` |
| **Source checkpoint SHA-256** | *(from ONNX metadata)* |
| **Resolved config / manifest-set SHA-256** | *(from ONNX metadata)* |
| **Software commit** | `git` SHA |
| **Training config** | e.g. `training/configs/default.yaml` |
| **Release date** | YYYY-MM-DD |
| **Authors / maintainers** | |

## 2. License (weights — not software)

| Field | Value |
|-------|--------|
| **Weight license** | SoundEx Model Weights License 1.0 |
| **License file** | [`MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md) |
| **Weight Tier (required)** | `SoundEx-Weights-Tier-A-BY-1.0` **or** `…-Tier-B-NC-SA-1.0` **or** `…-Tier-C-Research-1.0` |
| **SPDX-style id** | `LicenseRef-SoundEx-Model-Weights-1.0` + tier id above |
| **Software license** | Apache-2.0 (does **not** apply to this weight file) |

**Human-readable grant summary (pick the tier you declared):**

- **Tier A:** Use and redistribution allowed, including commercial, with attribution.  
- **Tier B:** Non-commercial use and redistribution only; share-alike for derivatives; attribution required.  
- **Tier C:** Non-commercial research / evaluation only; redistribution limited as in the weight license.

## 3. Training data bill-of-materials

| Dataset | Snapshot / version | Approx. duration | Split role | Upstream license | Mix-only? |
|---------|-------------------|------------------|------------|------------------|-----------|
| e.g. MUSDB18-HQ | 2019 HQ | ~10 h | train/validation/test | Research-only | yes (`mixture.wav`) |
| e.g. Slakh2100 | flac_redux | ~145 h | train/validation/test | CC BY 4.0 | yes (`mix.*`) |
| e.g. MedleyDB | V1+V2 | ~… h | train/validation/test | CC BY-NC-SA | yes (`*_MIX.wav`) |

| Data protocol field | Value |
|---------------------|-------|
| **Recipe version / SHA-256** | |
| **Manifest file(s) / SHA-256** | |
| **Manifest row and track counts by split** | |
| **Track-ID split intersections** | required: all zero |
| **Sample rates** | |
| **Codec/encoder/mode/settings** | |
| **Held-out codec setting** | |
| **Channel roles and weights** | mono and/or left/right/mid/side |
| **Alignment tolerance / rejection count** | |

**Degradation recipe:** Copy the resolved recipe and manifest summary, not CLI recollection.  
**Stems used?** No (required: No for standard SoundEx releases).

## 4. Intended use

- **Primary:** Bandwidth extension / restoration of high-frequency content lost to lossy coding or band-limiting in **music** signals.  
- **Out of scope:** Guaranteed recovery of original studio masters; forensic authentication; speech-only telephony standards compliance unless separately evaluated.

## 5. Limitations and risks

- Hallucinated high band may be musically plausible but incorrect.  
- The deployed model processes channels independently even though training includes
  post-codec left/right/mid/side examples; joint-stereo coherence still requires evaluation.  
- Dataset bias (genre, synthetic vs live, Western popular music skew).  
- Weight Tier may prohibit commercial deployment — check Section 2.

## 6. Evaluation (required for release)

| Evaluation field | Value |
|------------------|-------|
| **Evaluation report / SHA-256** | |
| **Test manifest version / SHA-256** | |
| **High-band LSD baseline / enhanced / delta 95% CI** | |
| **Low-band preservation LSD** | |
| **ViSQOLAudio baseline / enhanced** | |
| **Stereo correlation / phase / width** | |
| **Gate call / false-bypass rate** | |
| **Streaming chunk equivalence** | |
| **PyTorch / ORT / Rust parity evidence** | |
| **Listening protocol / anonymized aggregate** | randomized, level-matched, blinded degraded/enhanced/reference clips across every required stratum; include the signed record and aggregate |
| **Supported sample rates / codecs** | |
| **Algorithmic latency** | |
| **Real-model runtime evidence** | Apple Silicon and x86_64 benchmark filenames plus each report SHA-256; reports contain the model hash, ORT settings, raw samples, p50/p95/p99/max, RSS, RTF, stress duration, and deadline misses |
| **Known evaluation limitations** | |

Synthetic CI fixtures are tests only and must never be presented as model-quality
claims. A release report must come from the held-out manifest through the Rust
streaming path, include degraded baselines and uncertainty, and pass the versioned
machine gates. ViSQOLAudio and the listening record may not be silently omitted.

## 7. Attribution (copy into redistributions)

```text
SoundEx Model Weights ([TIER_ID])
Copyright [YEAR] SoundEx Contributors
Licensed under the SoundEx Model Weights License 1.0
(MODEL_WEIGHTS_LICENSE.md). This file is NOT Apache-2.0.
Training data attributions: see Section 3 of this Model Card and NOTICE.
```

## 8. Contact

- Project / issue tracker:  
- Security / licensing contact:  

## 9. Maintainer sign-off

I confirm that the declared Weight Tier is consistent with the training data BOM and `legal/TRAINING_DATA_AND_WEIGHT_TIERS.md`.

```text
Name: _______________  Date: _______________
```
