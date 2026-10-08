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
| **Checkpoint / artifact schema** | `soundex-checkpoint 1.2` / `soundex artifact 1.3` (default 256/128 profile) |
| **Source checkpoint SHA-256** | *(from ONNX metadata)* |
| **Resolved config / manifest-set SHA-256** | *(from ONNX metadata)* |
| **Software commit** | `git` SHA |
| **Training config** | e.g. `training/configs/default.yaml` |
| **Release date** | YYYY-MM-DD |
| **Authors / maintainers** | |

## 2. License and source approval

| Field | Value |
|-------|-------|
| **Weight license** | Apache-2.0 |
| **License file** | LICENSE |
| **Data rights review / SHA-256** | path/to/completed-rights-review.json + SHA-256 |
| **Software license** | Apache-2.0 |

Official weights permit application use, redistribution, adaptation and commercial
use under Apache-2.0. Complete the rights review for the exact artifact and source
checkpoint before publishing; see [training data policy](../legal/TRAINING_DATA.md).

## 3. Training data bill-of-materials

| Dataset | Snapshot / version | Approx. duration | Split role | Upstream license | Mix-only? |
|---------|-------------------|------------------|------------|------------------|-----------|
| e.g. Slakh2100 | official redux | measured | train/validation/test | CC BY 4.0 | yes (`mix.*`) |
| e.g. music library | catalog SHA-256 | measured | train/validation/test | actual per-source grants | yes |

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
- Genre and source bandwidth coverage require measured evaluation.

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
SoundEx Model Weights
Copyright [YEAR] SoundEx Contributors
Licensed under Apache License 2.0 (LICENSE).
Training-source credits and modifications: see this model card and NOTICE.
```

## 8. Contact

- Project / issue tracker:  
- Security / licensing contact:  

## 9. Maintainer sign-off

I confirm the data and initialization provenance support the Apache-2.0 release,
and the signed rights review and release evidence describe this exact artifact.

```text
Name: _______________  Date: _______________
```
