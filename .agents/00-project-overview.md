# SoundEx 项目总览

## 愿景

SoundEx 是一个用 Rust 编写的实时音频音质增强库，通过轻量化 AI 模型恢复有损压缩音频（MP3、AAC 等）在编码过程中丢失的高频信息，提升实际回放体验。其目标类似于 Sony DSEE（Digital Sound Enhancement Engine），但完全开源、跨平台、可嵌入任意音频管线。

## 核心目标

1. **实时性**：实时回放增强，默认128-frame回调；新增软件延迟目标<8ms、红线<10ms，回调不等模型并保持连续输出（完整标准见 `docs/realtime-standard.md`）
2. **轻量化**：模型参数 1-2M，ONNX 文件 < 8MB，单 forward pass 无迭代
3. **通用性**：同时支持音乐和语音场景
4. **流式处理**：支持实时音频流的逐帧低延迟输入输出
5. **文件处理**：支持对完整音频文件进行离线增强（不集成解码，由调用方提供 PCM）
6. **智能门禁**：自动检测输入音频是否需要增强（高频是否被裁剪），避免无效处理
7. **DSP 修复**：确保补偿部分的响度、相位与原始频段一致

## 技术选型结论

### 模型方案

融合 2023-2026 年音频超分辨率领域 SOTA 成果：

| 来源 | 借鉴内容 |
|------|----------|
| AERO (ICASSP 2023) | 频谱域 U-Net 编码器-解码器 + GAN 训练策略 |
| AP-BWE (TASLP 2024) | 双流（幅度 + 相位）并行预测 |
| UL-UNAS (2025) | 深度可分离卷积(DWS) + 倒置残差块(MB) 轻量化 |
| Bridge-SR (ICASSP 2025) | 1.7M 参数验证 + 频域辅助监督 |
| SFNet (TASLP 2026) | 源滤波器启发的频率分解 |
| MS-BWE (Interspeech 2024) | STFT 域操作 + iSTFT 重建 |

### 推理引擎

- **主选**：ort 2.0.0-rc.9+（ONNX Runtime Rust 绑定）
- **备选**：tract（纯 Rust，feature-gated，面向嵌入式）

### 训练框架

- PyTorch 2.x + torchaudio
- 数据：MUSDB18-HQ + Slakh2100 + MedleyDB（仅整轨缩混 mixture，不用分轨）
- 代码许可：Apache-2.0（LICENSE）
- 权重许可：SoundEx Model Weights License 1.0 + Tier A/B/C（MODEL_WEIGHTS_LICENSE.md、legal/）
- 第三方数据：NOTICE；默认多数据集合训 → Tier C
- 导出：PyTorch → ONNX

### Rust 核心依赖

| 功能 | Crate |
|------|-------|
| 推理 | ort |
| 解码(CLI) | symphonia |
| WAV 输出 | hound |
| FFT | rustfft + realfft |
| 重采样 | rubato |
| DSP 辅助 | dasp |
| CLI | clap |

## 工程结构

```
soundex/
├── Cargo.toml              # workspace root
├── crates/
│   ├── soundex-core/       # 核心 lib：模型推理 + 流式处理
│   ├── soundex-dsp/        # DSP：STFT、带宽检测、后处理
│   └── soundex-cli/        # CLI：文件解码 → 增强 → WAV 输出
├── training/               # Python 训练工程
├── models/                 # 预训练 ONNX 模型（git-lfs）
├── tests/                  # 集成测试 + 测试音频
├── benches/                # Criterion benchmarks
└── .agents/                # 项目规划文档（本目录）
```

## 开源协议

Apache License 2.0

## 交付物

1. `soundex-core`：Rust lib crate，暴露简洁 API 供调用方集成
2. `soundex-cli`：命令行工具，`soundex input.mp3 -o output.wav`
3. 训练工程：可本地/AutoDL 运行的完整训练 pipeline
4. 预训练模型：ONNX 格式，通过 git-lfs 管理
