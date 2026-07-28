# SoundEx Licensing Overview

**语言说明 / Language:** 下面「简明中文」便于快速理解；**具有约束力的条款以英文法律文本为准**（`LICENSE`、`MODEL_WEIGHTS_LICENSE.md` 及本目录下英文正文）。

---

## 简明中文

SoundEx 仓库里有 **两套彼此独立的许可**：

| 客体 | 许可文件 | 默认含义 |
|------|----------|----------|
| **源代码**（Rust、CLI、训练脚本、文档中的代码） | 根目录 [`LICENSE`](../LICENSE) | **Apache License 2.0** — 可商用、可修改、可再分发（须保留声明） |
| **模型权重**（`.pth`、`.onnx`、导出的网络参数及等价物） | [`MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md) | **不**自动适用 Apache-2.0；按训练数据落入对应 **权重等级** |
| **第三方数据集音频** | 各数据集自身条款 | **不随本仓库分发**；使用者自行获取并遵守 |
| **第三方依赖**（如 ONNX Runtime） | 各依赖自有许可 | 见构建与链接说明 |

**关键规则：**

1. 能跑 SoundEx **代码**，不等于能随便商用某个 **预训练权重**。  
2. 官方若发布权重，必须附带 **模型卡（Model Card）**，写明训练数据与权重等级。  
3. 使用 **MUSDB18-HQ** 和/或 **MedleyDB** 训练出的权重，默认只能 **非商业 / 研究**，且须署名。  
4. 仅在 **Slakh2100**（及同等宽松、可再分发的数据）上训练的权重，可按更宽松的 **署名（CC BY 4.0 对齐）** 等级发布。  
5. 权重许可 **不能** 放宽到严于所用训练数据所允许的范围。

详细矩阵见 [TRAINING_DATA_AND_WEIGHT_TIERS.md](TRAINING_DATA_AND_WEIGHT_TIERS.md)。  
发布清单见 [MODEL_RELEASE_CHECKLIST.md](MODEL_RELEASE_CHECKLIST.md)。  
第三方数据集摘要见根目录 [NOTICE](../NOTICE)。

---

## English (binding structure)

### 1. Dual-licensing architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SoundEx Software (source code)                             │
│  → Apache License 2.0  (./LICENSE)                          │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  SoundEx Model Weights (parameters / ONNX / checkpoints)     │
│  → SoundEx Model Weights License 1.0                        │
│    (./MODEL_WEIGHTS_LICENSE.md)                             │
│  → Specific Tier declared per release (Model Card required) │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Third-party training corpora (audio)                       │
│  → Not redistributed here; each corpus keeps its own terms  │
└─────────────────────────────────────────────────────────────┘
```

### 2. Document map

| Document | Role |
|----------|------|
| [`../LICENSE`](../LICENSE) | Software: Apache-2.0 full text |
| [`../MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md) | Operative license for model weights |
| [`TRAINING_DATA_AND_WEIGHT_TIERS.md`](TRAINING_DATA_AND_WEIGHT_TIERS.md) | Dataset → weight-tier matrix |
| [`MODEL_RELEASE_CHECKLIST.md`](MODEL_RELEASE_CHECKLIST.md) | Mandatory steps before publishing weights |
| [`../models/MODEL_CARD.template.md`](../models/MODEL_CARD.template.md) | Per-artifact disclosure template |
| [`../NOTICE`](../NOTICE) | Attribution & third-party training-data notices |
| [`DISCLAIMER.md`](DISCLAIMER.md) | Warranty, liability, non-advice notice |

### 3. Precedence

1. For **software**: Apache-2.0 controls.  
2. For **weights**: `MODEL_WEIGHTS_LICENSE.md` + the **Tier and Model Card** shipped with that artifact control.  
3. If software and weights conflict regarding a weight file, **weight terms control for that file**.  
4. Nothing in this repository grants rights in third-party audio datasets.

### 4. No legal advice

These documents are project policies and license grants from the copyright holders of SoundEx contributions. They are **not** a substitute for legal advice. Dataset terms can change; verify upstream licenses before commercial use or redistribution.
