# SoundEx Model Weights License

**Version:** 1.0  
**SPDX-License-Identifier (umbrella):** `LicenseRef-SoundEx-Model-Weights-1.0`  
**Copyright:** 2026 SoundEx Contributors  

This license governs **SoundEx Model Weights** only. It does **not** govern SoundEx source code, which remains under the Apache License, Version 2.0 (see the `LICENSE` file).

---

## 0. Definitions

**“Software”** means the SoundEx source code, build scripts, documentation that is not Model Weights, and related materials licensed under Apache-2.0.

**“Model Weights”** means any learned parameters, checkpoints, exported graphs, quantized or distilled derivatives, LoRA/adapters, embeddings produced as part of a SoundEx neural model, and any file primarily consisting of such parameters (including but not limited to `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.onnx`, and packed runtime blobs), whether official or third-party fine-tunes **derived from** SoundEx Model Weights.

**“Work”** means a specific set of Model Weights distributed under this license together with its **Model Card**.

**“Model Card”** means the human-readable disclosure file required by Section 4, identifying the Weight Tier, training data, and attribution.

**“Weight Tier”** means one of the tiers defined in Section 2, as declared in the Model Card for that Work.

**“You”** means an individual or legal entity exercising rights under this license.

**“Non-Commercial”** means not primarily intended for or directed toward commercial advantage or monetary compensation. Research use by commercial entities for internal evaluation may still be Non-Commercial if no product, paid service, or customer-facing system incorporates the Model Weights. **When in doubt, treat the use as commercial.**

**“Output”** means audio or other content produced by running Model Weights (inference). Output is **not** Model Weights; this license does not claim copyright in Output solely by virtue of inference. Separate laws (e.g., rights in the input audio) may still apply to Output.

---

## 1. Separation from Software License

1.1. Apache-2.0 rights in the Software **do not** include a license to Model Weights.  
1.2. Possession of the Software does not imply any right to obtain, use, or redistribute Model Weights beyond this license and the declared Weight Tier.  
1.3. Linking the Software with Model Weights at runtime does not merge the two licenses: Software stays Apache-2.0; Weights stay under this document and their Tier.

---

## 2. Weight Tiers

Every distribution of Model Weights **must** declare exactly one Weight Tier in its Model Card. If no tier is declared, **Tier C (Most Restrictive Research)** applies by default.

### Tier A — Attribution (CC BY 4.0–aligned)

**Identifier:** `SoundEx-Weights-Tier-A-BY-1.0`  
**Intended data profile:** Training/fine-tuning data limited to corpora that permit commercial use and redistribution of derived models with attribution only (e.g., Slakh2100 under CC BY 4.0, and similarly permissive sources). **Must not** include MUSDB18-HQ or MedleyDB (or other Non-Commercial / research-only corpora).

**Grant.** Subject to the terms of this license, each contributor grants You a worldwide, royalty-free, non-exclusive license to:

- use, reproduce, and create derivative Model Weights;  
- redistribute Model Weights and derivatives;  
- use Model Weights for commercial and Non-Commercial purposes;

provided that You:

1. give appropriate **credit** (Section 5);  
2. indicate if changes were made;  
3. do not imply endorsement by the licensor; and  
4. include a copy of this license and the Model Card (or a URL to a stable copy).

This Tier is intended to be **compatible in spirit with Creative Commons Attribution 4.0 International** for the Model Weights as a whole. It is **not** a CC legal code dual-license unless the Model Card explicitly dual-licenses under CC BY 4.0.

### Tier B — Non-Commercial Attribution Share-Alike (MedleyDB-aligned)

**Identifier:** `SoundEx-Weights-Tier-B-NC-SA-1.0`  
**Intended data profile:** Training includes MedleyDB / MedleyDB 2.0 or other CC BY-NC-SA (or equivalent Non-Commercial Share-Alike) material, and does **not** include stricter research-only corpora such as MUSDB18-HQ **unless** those corpora separately allow the same Non-Commercial redistribution (they generally do not — use Tier C if MUSDB is included).

**Grant.** Subject to the terms of this license, each contributor grants You a worldwide, royalty-free, non-exclusive license to:

- use, reproduce, and create derivative Model Weights **for Non-Commercial purposes only**;  
- redistribute Model Weights and derivatives **for Non-Commercial purposes only**;

provided that You:

1. give appropriate **credit** (Section 5);  
2. indicate if changes were made;  
3. distribute derivatives under **Tier B or a license no less restrictive** regarding Non-Commercial use and share-alike (Section 6);  
4. include this license and the Model Card with redistributions.

**Commercial use of Tier B Model Weights is not granted.** Contact the rights holders for a separate commercial agreement if available.

### Tier C — Research / Evaluation Only (Default mixed-training / MUSDB-aligned)

**Identifier:** `SoundEx-Weights-Tier-C-Research-1.0`  
**Intended data profile:** Training includes MUSDB18-HQ or any corpus limited to research/evaluation, or a **mixture** of datasets that includes any such corpus, or any training run whose data bill-of-materials is incomplete.

**Grant.** Subject to the terms of this license, each contributor grants You a worldwide, royalty-free, non-exclusive license **solely** to:

- use and reproduce the Model Weights for **non-commercial research, academic teaching, and personal evaluation**;  
- publish scientific results and non-commercial demos that **do not** distribute the Model Weights as a product component;

provided that You:

1. give appropriate **credit** (Section 5);  
2. **do not** use the Model Weights in any commercial product, paid API, or revenue-generating service;  
3. **do not** redistribute Model Weights except (a) within a research collaboration under the same Tier C terms, or (b) as required for paper reproducibility with this license and Model Card attached;  
4. do not remove or obscure the Model Card.

**No commercial license is granted under Tier C.** No patent license is granted beyond what is necessary for the limited research use above, to the extent such a grant is recognized.

### Tier selection rule (mandatory)

When training data spans multiple corpora, the Work’s Weight Tier **must** be the **most restrictive** tier required by any included corpus:

```
MUSDB18-HQ (research-only)     → at least Tier C
MedleyDB (CC BY-NC-SA)         → at least Tier B  (or Tier C if combined with MUSDB)
Slakh2100 (CC BY 4.0) alone    → may be Tier A
Unknown / undeclared data      → Tier C
```

See `legal/TRAINING_DATA_AND_WEIGHT_TIERS.md` for the full matrix.

---

## 3. Restrictions (all tiers)

You may **not**:

1. misrepresent the Weight Tier or training data of a Work;  
2. remove, alter, or obscure copyright, license, Model Card, or NOTICE information;  
3. use Model Weights to violate applicable law;  
4. claim that Outputs or Weights are official SoundEx releases unless distributed by the project under a matching Model Card;  
5. sublicense Model Weights under terms that grant third parties broader rights than this license and the declared Tier allow.

---

## 4. Model Card requirement

4.1. Any public or private redistribution of Model Weights **must** be accompanied by a Model Card substantially containing the fields in `models/MODEL_CARD.template.md`.  
4.2. Minimum fields: weight tier identifier; list of training datasets; software commit or version; author; date; license pointer to this file.  
4.3. Official SoundEx releases place the Model Card next to the weight file (e.g., `models/soundex-v1.onnx` + `models/soundex-v1.model-card.md`).

---

## 5. Attribution

If You redistribute Model Weights or publish substantial use of them, You must retain and reasonably display:

1. “SoundEx Model Weights” and the Weight Tier identifier;  
2. a copyright notice (e.g., “Copyright 2026 SoundEx Contributors”);  
3. a link or path to this license;  
4. dataset attributions required by upstream corpora (see `NOTICE` and the Model Card), including at minimum any of: MUSDB18-HQ, Slakh2100, MedleyDB that were used.

Example short notice:

```text
SoundEx Model Weights (Tier C — Research Only)
Copyright 2026 SoundEx Contributors
Licensed under SoundEx Model Weights License 1.0
(see MODEL_WEIGHTS_LICENSE.md). Training data: see Model Card.
```

---

## 6. Derivatives and fine-tunes

6.1. Fine-tunes, quantizations, distillations, and format conversions of Model Weights are **derivative Model Weights** under the **same Weight Tier** as the parent Work, unless a new Model Card documents a **stricter** tier.  
6.2. You **may not** re-tier a Work to a **more permissive** tier than the parent without a full retrain (or documented training) that **excludes** all data requiring the stricter tier, and a truthful Model Card.  
6.3. Combining Tier A weights with Tier B/C data or weights yields at least the stricter tier.

---

## 7. Outputs

7.1. This license does not restrict Output solely because it was produced by inference, subject to Section 3 and applicable law.  
7.2. You are solely responsible for rights in input audio and for lawful use of Output.

---

## 8. Trademarks

This license does not grant permission to use the name “SoundEx”, logos, or project marks except as required for reasonable attribution under Section 5.

---

## 9. Disclaimer of warranty

MODEL WEIGHTS ARE PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. YOU BEAR THE RISK OF USE.

---

## 10. Limitation of liability

TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE MODEL WEIGHTS OR THEIR USE — INCLUDING ANY CLAIM RELATED TO TRAINING DATA RIGHTS, OUTPUT AUDIO, OR DOWNSTREAM PRODUCTS.

---

## 11. Termination

11.1. Rights under this license terminate automatically if You fail to comply with its terms.  
11.2. Upon termination, You must cease distribution of the Model Weights. Sections 9–10 and 12 survive termination.

---

## 12. Miscellaneous

12.1. If any provision is unenforceable, the remainder remains in effect.  
12.2. No waiver is implied by failure to enforce.  
12.3. This license is the entire agreement regarding Model Weights between You and the licensors, except for separate written commercial agreements.  
12.4. License text may be updated in future versions for new releases; **existing** distributed Works remain under the version stated in their Model Card unless You accept a newer version for a new distribution.  
12.5. Governing language of this legal instrument is **English**.

---

## 13. Acceptance

By downloading, copying, using, or distributing Model Weights, You agree to this SoundEx Model Weights License 1.0 and to the Weight Tier declared in the accompanying Model Card (or Tier C if undeclared).
