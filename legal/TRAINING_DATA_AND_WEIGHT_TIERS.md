# Training Data and Model Weight Tiers

**Status:** Normative project policy  
**Related:** [`../MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md)

This document maps **which training corpora may be used** for each official Weight Tier. It is binding for any release that claims an official SoundEx tier identifier.

---

## 1. Corpora recognized by the training pipeline

| Corpus | Clean audio used by SoundEx | Upstream license (summary) | Project preprocessor |
|--------|----------------------------|----------------------------|----------------------|
| **MUSDB18-HQ** | Full-track `mixture.wav` only | Research / non-redistribution of audio; commercial use restricted by upstream terms | `training/data/preprocess_musdb.py` |
| **Slakh2100** (+ redux) | Full-track `mix.flac` / `mix.wav` only | **CC BY 4.0** | `training/data/preprocess_slakh.py` |
| **MedleyDB** / **2.0** | Full-track `*_MIX.wav` only | **CC BY-NC-SA** (non-commercial, share-alike) | `training/data/preprocess_medleydb.py` |

Stems, RAW multitracks, and MIDI are **out of scope** for SoundEx training data and must not be listed as training audio in a Model Card unless a future policy explicitly allows them.

**Upstream links (verify before release):**

- MUSDB: https://sigsep.github.io/datasets/musdb.html  
- Slakh: http://www.slakh.com/  
- MedleyDB: https://medleydb.weebly.com/  

---

## 2. Tier assignment matrix

Let **S** = Slakh2100 used, **M** = MedleyDB used, **U** = MUSDB18-HQ used.

| S | M | U | Maximum permitted Weight Tier | Typical official label |
|---|---|---|-------------------------------|-------------------------|
| ✓ | — | — | **Tier A** (Attribution / commercial-friendly) | `SoundEx-Weights-Tier-A-BY-1.0` |
| ✓ | ✓ | — | **Tier B** (Non-Commercial SA) | `SoundEx-Weights-Tier-B-NC-SA-1.0` |
| — | ✓ | — | **Tier B** | `SoundEx-Weights-Tier-B-NC-SA-1.0` |
| ✓ | — | ✓ | **Tier C** (Research only) | `SoundEx-Weights-Tier-C-Research-1.0` |
| — | — | ✓ | **Tier C** | `SoundEx-Weights-Tier-C-Research-1.0` |
| ✓ | ✓ | ✓ | **Tier C** | `SoundEx-Weights-Tier-C-Research-1.0` |
| — | ✓ | ✓ | **Tier C** | `SoundEx-Weights-Tier-C-Research-1.0` |
| — | — | — | **Tier C** (no BOM / synthetic-only without review) | default if undeclared |

**Rule:** The declared tier must be **equal to or stricter than** the maximum permitted tier for the data used. You may always choose a stricter tier (e.g., train on Slakh only but release as Tier C).

---

## 3. Default configuration of this repository

`training/configs/default.yaml` may list paths for MUSDB18-HQ, Slakh2100, and MedleyDB. If a full multi-corpus training run uses all three:

→ Official weights **must** be labeled **Tier C**.

If maintainers later ship a **commercial-friendlier** checkpoint, they must:

1. Train (or fine-tune from random init) **without** MUSDB and MedleyDB;  
2. Document the data bill-of-materials;  
3. Declare **Tier A** (or Tier B if only MedleyDB+Slakh, etc.) in the Model Card.

---

## 4. Data bill-of-materials (BOM)

Every Model Card must include a BOM table, for example:

```markdown
| Dataset    | Version / snapshot | Hours (approx.) | Role   | Upstream license   |
|------------|--------------------|-----------------|--------|--------------------|
| Slakh2100  | flac_redux         | 145             | train  | CC BY 4.0          |
| MUSDB18-HQ | 2019               | 10              | train  | Research-only      |
| MedleyDB   | V1+V2              | ~10             | train  | CC BY-NC-SA        |
```

Incomplete BOM ⇒ treat as **Tier C**.

---

## 5. What the weight license is not

- It is **not** a license to redistribute MUSDB, MedleyDB, or Slakh **audio**.  
- It is **not** Apache-2.0.  
- It does **not** exhaust all obligations under upstream dataset terms; You remain responsible for compliance with upstream licenses (attribution URLs, research-only clauses, NC clauses, etc.).

---

## 6. Future corpora

New datasets may be added only with an update to this matrix and `NOTICE`. Until listed here, any new corpus forces **Tier C** for weights trained with it.
