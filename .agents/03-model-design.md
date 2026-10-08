# SoundEx 模型设计

## SOTA 调研（2023-2026）

### 完整对比表

| 模型 | 年份/会议 | 方法 | 参数量 | 实时性 | 通用性 | ONNX友好 |
|------|-----------|------|--------|--------|--------|----------|
| Bridge-SR | ICASSP 2025 | 薛定谔桥/波形域 | 1.7M | 需ODE求解(1阶) | 语音 | 中 |
| AudioLBM | NeurIPS 2025 | 隐空间桥模型 | 较大 | 需迭代 | 语音+音乐+音效 | 低 |
| SFNet | TASLP 2026.1 | 源滤波器+DNN | 轻量 | 实时 | 语音 | 高 |
| AERO | ICASSP 2023 | 频谱域U-Net+GAN | ~2M | 单pass | 语音+音乐 | 高 |
| AP-BWE | TASLP 2024 | 双流幅相(ConvNeXt) | ~3.5M | GPU 1000x RT | 语音 | 高 |
| MS-BWE | Interspeech 2024 | 多级级联 | ~1.5M/级 | CPU 60x RT | 语音 | 高 |
| FLowHigh | ICASSP 2025 | 单步Flow Matching | 较大 | 单步但模型大 | 语音 | 中 |
| VM-ASR | TASLP 2025 | 轻量双流U-Net | ~1.5M | 可实时 | 通用 | 高 |
| UL-UNAS | 2025 | 超轻量U-Net(DWS) | <1M | 边缘实时 | 语音 | 高 |
| TRAMBA | IMWUT 2025 | Transformer+Mamba | ~2M | 移动端实时 | 语音 | 低(SSM) |
| HP-Codec BWE | arXiv 2025.11 | 谐波-打击乐分离 | - | - | 音乐 | 中 |
| Opus BbWENet | 2026(产品) | DNN BWE | 9MB | 实时 | 语音 | - |
| FSPEN | ICASSP 2024 | 全带+子带 | 79K | 实时 | 语音增强 | 高 |
| AudioSR | ICASSP 2024 | Latent Diffusion | >100M | 不可实时 | 通用 | 低 |

### 关键洞察

1. **Bridge-SR (ICASSP 2025)**：仅 1.7M 参数即达 SOTA 质量，验证了小模型上限
2. **SFNet (TASLP 2026)**：物理先验（源滤波器）大幅降低计算需求
3. **HP-Codec BWE (2025.11)**：谐波-打击乐分离对音乐场景至关重要
4. **Opus BbWENet (2026)**：DNN BWE 已在量产编解码器中部署，证明工程可行性
5. **AERO (ICASSP 2023)**：频谱域单 pass 推理最适合 ONNX 部署和流式处理

### 排除方案及理由

| 方案 | 排除理由 |
|------|----------|
| Mamba/SSM 系 (TRAMBA, AEROMamba) | ONNX Runtime 对 SSM 算子支持不完善 |
| Diffusion 系 (AudioSR, NU-Wave) | 模型 >100M，需多步迭代，不满足实时约束 |
| Flow Matching (FLowHigh) | 模型体积大，Transformer 骨干不利于轻量部署 |
| 纯波形域 (Bridge-SR) | 需 ODE 求解，非单 pass，延迟不可控 |

## 融合架构设计

### 总体架构

```
Input: 低质量音频帧 (时域)
         │
         ▼
    ┌─────────┐
    │  STFT   │  1024-pt FFT, hop=512, Hann
    └────┬────┘
         │ log-magnitude [513 bins] + phase [513 bins]
         ▼
    ┌─────────────────────────────────────┐
    │     Frequency Splitter              │
    │  (基于检测到的截止频率 fc 分割)      │
    │  低频: [0, fc] → passthrough        │
    │  高频: [fc, nyquist] → 待增强       │
    └────┬────────────────────────────────┘
         │ 高频区域的 log-mag + phase
         ▼
    ┌─────────────────────────────────────┐
    │     Dual-Stream U-Net Generator     │
    │                                     │
    │  ┌───────────┐  ┌───────────┐      │
    │  │ Amplitude │  │   Phase   │      │
    │  │  Stream   │  │  Stream   │      │
    │  │(DWS+MB)   │  │(DWS+MB)   │      │
    │  └─────┬─────┘  └─────┬─────┘      │
    │        │               │            │
    │        └───────┬───────┘            │
    │                │                    │
    │        ┌───────▼───────┐            │
    │        │  Fusion Layer │            │
    │        └───────────────┘            │
    └────┬────────────────────────────────┘
         │ 候选完整频带 log-mag + phase（部署端仅融合缺失高频）
         ▼
    ┌─────────────────────────────────────┐
    │     DSP Post-Processing             │
    │  - 响度匹配 (RMS normalization)     │
    │  - 相位平滑 (相邻频段连续)          │
    │  - 交叉渐变 (crossover blend)       │
    └────┬────────────────────────────────┘
         │
         ▼
    ┌─────────┐
    │  iSTFT  │  重建时域信号
    └────┬────┘
         │
         ▼
Output: 增强后音频帧 (时域)
```

### 生成器网络细节

**编码器-解码器结构（U-Net with skip connections）**：

```
Encoder:
  Conv2d(in_ch, 24, k=(1,3), s=(1,2))             → [B, 24, 1, F/2]
  DWSBlock(24→48, stride=(1,2))                    → [B, 48, 1, F/4]
  DWSBlock(48→96, stride=(1,2))                    → [B, 96, 1, F/8]
  DWSBlock(96→96, stride=(1,2))                    → [B, 96, 1, F/16]

Bottleneck:
  InvertedResidual(96, expand=4, k=(1,3))          → [B, 96, 1, F/16]
  InvertedResidual(96, expand=4, k=(1,3))          → [B, 96, 1, F/16]

Decoder (对称，带 skip connections):
  DWSBlock(96→96, frequency upsample) + skip       → [B, 96, 1, F/8]
  DWSBlock(96→48, frequency upsample) + skip       → [B, 48, 1, F/4]
  DWSBlock(48→24, frequency upsample) + skip       → [B, 24, 1, F/2]
  ConvTranspose2d(24→out_ch, k=(1,4), s=(1,2))    → [B, out_ch, 1, F]
```

**DWSBlock（深度可分离卷积块，来自 UL-UNAS）**：

```
DWSBlock(channels):
  DepthwiseConv2d(ch, ch, k=(1,3), padding=(0,1))
  BatchNorm2d(ch)
  GELU()
  PointwiseConv2d(ch, ch, k=1)
  BatchNorm2d(ch)
  GELU()
  + Residual Connection
```

**InvertedResidual（倒置残差块）**：

```
InvertedResidual(ch, expand_ratio=4):
  PointwiseConv2d(ch, ch*expand) + BN + GELU
  DepthwiseConv2d(ch*expand, ch*expand, k=(1,3)) + BN + GELU
  PointwiseConv2d(ch*expand, ch) + BN
  + Residual Connection
```

### 双流设计（来自 AP-BWE）

- **幅度流 (Amplitude Stream)**：输入完整 degraded log-magnitude，预测候选 clean log-magnitude
- **相位流 (Phase Stream)**：输入完整 degraded phase，预测候选 clean phase
- 两流结构相同但参数独立，各自完成编码器、bottleneck 和解码器
- 最终通过 Fusion Layer 合并

### 源滤波器启发（未纳入当前合同）

当前 artifact 固定只有 log-magnitude 和 phase 两个输入通道，不包含 HPSS 或额外
源滤波特征。以下仅是未来实验方向，若采用必须升级模型/metadata 合同并重新训练：
- 对输入频谱进行简单的谐波-打击乐分离（HPSS 简化版）
- 将谐波成分和噪声成分作为额外输入通道
- 使模型对周期性成分（人声、乐器基频）和非周期成分（打击乐、齿音）分别建模

### 输入/输出规格

| 项目 | 规格 |
|------|------|
| FFT 大小 | 1024 |
| Hop 大小 | 512 |
| 频率 bins | 513 (0 ~ Nyquist) |
| 输入通道 | 2 (完整频带 log-magnitude + phase) |
| 输出通道 | 2 (候选完整频带 log-magnitude + phase) |
| 时间维度 | 固定 1 帧；batch 维度与时间维度不可互换 |
| 模型输入形状 | [batch, 2, 1, 513] |
| 模型输出形状 | [batch, 2, 1, 513] |

### 流式推理契约

- 生成器是无状态、单帧模型。所有卷积核和步幅在时间轴上的尺寸均为 1，模型只沿频率轴运算。
- 时间历史只存在于部署端的因果 STFT 环形缓冲区中；生成器既不读取未来帧，也不携带隐藏状态。
- 训练和验证都必须使用 `context_frames: 1`。训练可随机选择一个对齐帧，验证必须确定性选择中间帧。
- 输入输出频率轴通过确定性的右侧补零或裁剪对齐；幅度和原始相位角均不得通过插值修复形状。
- 旧的多帧、时间卷积 checkpoint 属于不兼容的 schema 0，不能直接导出或部署，必须用当前契约重新训练。

### 参数量估算

| 组件 | 参数量 |
|------|--------|
| Encoder (4 级 DWS) | ~200K |
| Bottleneck (2× InvertedResidual) | ~400K |
| Decoder (4 级 DWS + skip) | ~250K |
| Phase Stream 额外参数 | ~150K |
| Fusion + Output | ~50K |
| **总计** | **~1.0-1.5M** |

## 训练方案

### 数据集

| 数据集 | 用途 | 采样率 | 规模 | 许可 | 使用内容 |
|--------|------|--------|------|------|----------|
| 通用音频目录 | 真实音乐 | 优先原始无损；兼容低采样率素材 | 按录入结果统计 | 可选来源元数据；发布前核对 | directory / optional catalog |
| Slakh2100 | 合成音乐扩量 | 44.1kHz FLAC | 2100曲/~145h | CC BY 4.0 | 仅 `mix.flac` |
| 来源限制 | MUSDB 入口移除；训练不设许可白名单，发布时核对用途授权 | 详见 legal/TRAINING_DATA.md | — | — | — |

训练不依赖分轨/stems 标签；优先全带无损缩混作 clean 目标，其他素材按实际带宽记录，避免误作全带恢复证据。

### 数据预处理 Pipeline

唯一权威配置是 `data.recipe`。预处理在 44.1/48 kHz 下保留立体声完成 MP3
CBR/VBR、AAC-LC、Vorbis 编解码，再校正 encoder delay，并从编码后的声道生成
left/right/mid/side（单声道来源保留 mono）。每个 corpus 使用 canonical track ID
形成互斥的 train/validation/test 三分割；AAC 160 kbps setting 只进入
validation/test。生成结果先写入 staging，通过样本率、有限值、非静音、残余对齐、
pair checksum 和 split 泄漏检查后，才原子发布为 `<corpus>-<recipe-hash>`。

训练 loader 只读取 `manifest.jsonl` 中列出的 pair，目录中的游离文件不会进入训练。
manifest 固定 validation/test 的样本位置，并返回 codec、sample rate、channel role、
track ID、测得 cutoff 等部署评估元数据。

### 训练配置

```yaml
# training/configs/default.yaml
model:
  generator:
    channels: [24, 48, 96, 96]
    bottleneck_blocks: 2
    expand_ratio: 4
    use_phase_stream: true
  discriminator:
    type: multi_scale
    scales: [1, 2, 4]

training:
  batch_size: 16
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_steps: 1000
  max_epochs: 200
  gradient_clip: 1.0

audio:
  sample_rate: 44100
  fft_size: 1024
  hop_size: 512

data:
  music_library_path: /data/music-library
  slakh2100_path: /data/slakh2100
  recipe:
    schema_version: 1
    sample_rates: [44100, 48000]
    # 完整 codec/channel/split/alignment matrix 见实际 profile
```

### 损失函数

```
L_total = λ1 * L_recon + λ2 * L_adv + λ3 * L_feat + λ4 * L_phase

其中:
- L_recon: 多分辨率 STFT 损失 (幅度 L1 + 相位敏感)
- L_adv: 对抗损失 (Multi-Scale Discriminator)
- L_feat: 特征匹配损失 (判别器中间层)
- L_phase: 相位一致性损失 (来自 Bridge-SR 的频域辅助监督)

权重: λ1=1.0, λ2=0.1, λ3=0.5, λ4=0.2
```

### 训练策略

1. **Phase 1 (Warm-up)**：仅 L_recon，50 epochs，学习基本映射
2. **Phase 2 (GAN)**：加入 L_adv + L_feat，100 epochs，提升感知质量
3. **Phase 3 (Fine-tune)**：降低 lr，加入 L_phase，50 epochs，精修相位

### ONNX 导出

```python
python export_onnx.py \
  --checkpoint checkpoints/epoch_0200.pth \
  --output ../models/soundex-v1.onnx
```

- schema 1.2 checkpoint 是模型、音频、特征和数据语义的唯一来源；可选 `--config` 只做一致性比较。
- 导出 artifact schema 1.2、opset 17、FP32、动态 batch，其他维度严格为 `[2, 1, 513]`。
- ONNX 内嵌 `soundex.*` schema、特征、sample-rate、checkpoint/config/data 哈希元数据。
- `onnx.checker`、五类确定性 PyTorch/ORT 对比和 `<8 MiB` 门全部通过后才原子发布。
- Rust 在首帧前验证完整张量与元数据契约；量化模型需要独立 schema 和质量/延迟门。

### 评估指标

| 指标 | 说明 |
|------|------|
| LSD (Log-Spectral Distance) | 频谱失真，越低越好 |
| ViSQOL | 感知质量，越高越好 |
| PESQ (语音) | ITU-T P.862，越高越好 |
| SI-SDR | 信号失真比 |
| FAD (Fréchet Audio Distance) | 生成质量分布距离 |
| 推理延迟 | 单帧 ms |
| RTF (Real-Time Factor) | 处理速度/实时 |
