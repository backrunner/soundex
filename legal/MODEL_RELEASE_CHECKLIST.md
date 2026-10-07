# Model Weights Release Checklist

Use this checklist before publishing any learned SoundEx checkpoint or ONNX file (official or community release that claims a SoundEx Weight Tier).

Synthetic test graphs without learned parameters are Apache-2.0 software fixtures;
this learned-weight release checklist does not apply to them.

## A. Training provenance

- [ ] Record git commit hash of the Software used to train and export
- [ ] Record training config path(s) and any CLI overrides
- [ ] List **every** dataset that contributed clean or degraded pairs
- [ ] Confirm only **full-track mixes** were used (no stems / RAW / MIDI audio)
- [ ] Estimate training hours / steps / epochs for the Model Card
- [ ] Note random seed(s) if reproducibility is claimed

## B. Tier selection

- [ ] Apply `legal/TRAINING_DATA_AND_WEIGHT_TIERS.md` matrix
- [ ] Choose tier **≤ maximum permitted** (stricter is OK)
- [ ] If MUSDB18-HQ appears in the BOM → **Tier C only**
- [ ] If MedleyDB appears without MUSDB → at best **Tier B**
- [ ] If only Slakh2100 (or other Tier-A-approved data) → **Tier A** allowed
- [ ] Set SPDX-style identifier string in the Model Card

## C. Artifacts

- [ ] Weight file(s) (e.g. `soundex-v1.onnx`, checkpoint)
- [ ] Model Card completed from `models/MODEL_CARD.template.md`
- [ ] SHA-256 checksum of each weight file recorded in the Model Card
- [ ] ONNX checker and mandatory PyTorch / ORT / Rust parity passed (`<1e-5`)
- [ ] Tensor contract is dynamic batch only and fixed `[B, 2, 1, fft_size / 2 + 1]` FP32 (`129` bins for default 256/128)
- [ ] Required `soundex.*` metadata matches the schema-1.2 source checkpoint contract; default 256/128 ONNX artifact schema is 1.3
- [ ] `best-validation.pth` was selected only from immutable validation rows and its validation-manifest hash is recorded
- [ ] Held-out test evaluation ran the ONNX model through Rust offline and both required chunk patterns
- [ ] High/low-band, perceptual, stereo, gate, clipping, and chunk-equivalence metrics include degraded baselines and 95% CIs
- [ ] Versioned release gates pass, including artifact-bound PyTorch / ORT / Rust evidence and ViSQOLAudio
- [ ] Randomized, level-matched, blinded listening protocol is completed and its anonymized aggregate recorded
- [ ] The same ONNX artifact passes 30-minute mono/stereo performance evidence on named Apple Silicon and x86_64 reference CPUs
- [ ] `python -m evaluation.model_card ARTIFACT CARD REPORT APPLE_PERF_JSON X86_PERF_JSON` passes
- [ ] Copy of or link to `MODEL_WEIGHTS_LICENSE.md` (version 1.0)
- [ ] Dataset attributions matching `NOTICE`

## D. Legal packaging

- [ ] Do **not** mark weight archives as `Apache-2.0` only
- [ ] Archive or release notes state: “Software: Apache-2.0; Weights: SoundEx Model Weights License 1.0, Tier X”
- [ ] `NOTICE` included or linked for third-party data credits
- [ ] No dataset audio redistributed in the weight archive
- [ ] Trademark use limited to attribution (no false “official” claims if unofficial)

## E. Distribution channels

- [ ] Git LFS / release asset / Hugging Face / other host updated
- [ ] README download section points to Model Card + weight license
- [ ] (If HF) `license` metadata set to a custom/other tag + link to Model Card, not bare Apache-2.0

## F. Sign-off

- [ ] Maintainer name / date
- [ ] Statement: “I confirm the Weight Tier matches the training data BOM.”

```text
Signed: __________________  Date: __________
```
