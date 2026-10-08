# SoundEx 需求方案

## 功能需求

### F1: 实时流式音频增强

- 对实时音频流进行逐帧增强处理
- 输入：短瞬态 PCM 信号帧（f32 交织/平面格式）
- 输出：增强后的 PCM 信号帧
- 支持单声道和立体声
- 支持采样率：44.1kHz、48kHz（主要），可选 16kHz、22.05kHz、32kHz
- 每次调用接收 128 个新 samples/channel（约 2.90ms @ 44.1kHz），因果 STFT
  分析窗为 256 samples

### F2: 文件级音频增强

- 对完整 PCM buffer 进行离线增强
- 调用方负责音频文件解码和码流解码，SoundEx 仅接收/输出 PCM
- 支持任意长度的音频数据
- 内部自动处理帧边界和 overlap-add

### F3: 智能带宽检测门禁

- 自动分析输入音频的频谱特征
- 检测高频是否存在截断（有损压缩的典型特征）
- 若高频完整（无需增强），则 bypass 模型推理，直接 passthrough
- 检测应基于帧级别，支持动态切换

### F4: DSP 后处理修复

- **响度匹配**：确保生成的高频成分与原始低频段响度一致
- **相位一致性**：补偿部分的相位应与相邻频段平滑过渡
- **平滑过渡**：在截止频率附近使用渐变窗，避免频谱拼接伪影
- **限幅保护**：防止输出信号削波

### F5: CLI 工具

- 命令格式：`soundex <input> [-o output.wav] [options]`
- 支持输入格式：MP3、AAC、FLAC、WAV、OGG（通过 symphonia 解码）
- 输出格式：WAV（PCM 16-bit / 24-bit / 32-bit float）
- 可选参数：
  - `--model <path>`：指定模型文件
  - `--bypass-threshold <db>`：门禁阈值
  - `--dry-run`：仅分析是否需要增强
  - `--verbose`：输出处理信息
- 不支持流式输入/输出

### F6: 模型管理

- 支持从文件加载 ONNX 模型
- 支持嵌入默认模型（可选 feature）
- 模型热切换（运行时更换模型）

## 非功能需求

### NF1: 性能

| 指标 | 目标值 | 测试条件 |
|------|--------|----------|
| 算法库延迟 | <= 128 samples/channel | 由 `fft_size - hop_size` 计算；不含计算时间和设备缓冲 |
| 音频回调处理时间 | p99 < 0.1ms，max < 0.5ms 且不超过回调周期 | 非阻塞 `RealtimeProcessor`，128 frames 与 hop 对齐；Apple Silicon / x86_64 分别验收 |
| 实时稳定性 | 连续 30 分钟 0 丢样、0 设备 xrun | 44.1/48kHz、mono/stereo、128-frame 回调；保存原始回调采样，真实设备验收 |
| 增强可用性 | worker 迟到比例 <=0.1%，最长连续迟到 <=20ms | 排除启动 padding，分别验收每组；迟到必须平滑旁路，不延迟播放 |
| 持续实时系数 | RTF < 1.0 | 处理时间 / 对应音频时长；离线倍速仅报告，不设 `>100x` 门槛 |
| 热路径分配 | SoundEx Rust 代码 0 次 heap allocation/hop | 构造和预热后；ORT C/C++ 内部分配通过 RSS/压力测试另行观测 |
| 冷启动 | 单独报告模型构造与首帧时间 | 模型加载必须在音频回调之外，不与稳态回调 p99 合并 |
| 内存占用 | < 50MiB peak process RSS | 含模型、ORT 和工作缓冲，按平台注明测量方法 |

新增软件输出延迟目标 **<8ms**，红线 **<10ms**。5ms 仅保留为历史优化参考。
主机默认 128 frames 与 hop 对齐，保留原生 44.1/48kHz，不为延迟指标额外重采样。
默认 `256/128` STFT 延迟128 samples，独立 worker 再交接128 samples，实时适配层
总固定延迟256 samples/channel（5.805ms @44.1kHz / 5.333ms @48kHz）。
预算包含回调执行和接入额外缓冲。worker 迟到不增加延迟，而是按时间戳丢弃并平滑旁路。
同步模型 benchmark 的计算认证与实时音频输出连续性分开验收，不用纯推理 p99 代替
音频回调或总延迟。实际设备不支持128帧时，协商设备周期并对转换路径重新验收。
产品必须实测相对适配层外未处理通路的输出时间差；内部 dry fallback 已保留相同
256样本延迟，不能用内部 wet/dry 切换测总新增延迟。设备原有延迟单独列出。
完整标准见 `docs/realtime-standard.md`；不把软件预算表述为设备端到端实测。
模型按新协议训练并通过质量、30分钟压力、内存及跨硬件门后才可发布。

### NF2: 模型约束

| 指标 | 目标值 |
|------|--------|
| 参数量 | 1-2M |
| ONNX 文件大小 | < 8MB (FP32)，< 4MB (FP16/INT8) |
| 推理模式 | 单 forward pass，无自回归/迭代 |
| 算子兼容性 | ONNX Runtime 标准算子集 |

### NF3: 平台支持

- 桌面：macOS (x86_64, aarch64)、Linux (x86_64, aarch64)、Windows (x86_64)
- 移动：iOS (aarch64)、Android (aarch64)（交叉编译）
- 嵌入式：可选 tract 后端（feature-gated）

### NF4: 可维护性

- 模块化设计，核心 lib 无 I/O 依赖
- 完善的文档注释（rustdoc）
- 单元测试覆盖率 > 80%（核心 DSP 和推理逻辑）
- 集成测试 + benchmark 体系

### NF5: 训练约束

- 单卡 GPU（RTX 3090/4090）可完成训练
- 训练时间 < 24h
- 支持本地 macOS/Linux 训练
- 支持 AutoDL 等云平台迁移（提供 Dockerfile）

## 约束条件

1. **不集成音频解码**：lib 层面不处理 MP3/AAC 等格式的解码，仅接收 PCM
2. **不集成音频播放**：不包含任何音频 I/O 设备交互
3. **模型格式**：仅支持 ONNX（通过 ort 推理）
4. **开源协议**：Apache 2.0，训练数据集需兼容此协议
5. **数据与许可**：训练使用 Slakh mix 与通用音频目录，通过压缩合成训练对，不依赖分轨标签；MUSDB 入口移除，不设训练许可白名单，扩充曲风、乐器、瞬态和声场覆盖，正式发布时核对授权。正式权重采用 Apache-2.0，支持其他开源软件集成和分发；见 legal/TRAINING_DATA.md。

## API 设计草案

```rust
// soundex-core 对外暴露的核心 API

/// 增强器配置
pub struct SoundExConfig {
    pub model_path: PathBuf,
    pub sample_rate: u32,
    pub channels: u16,
    pub bypass_threshold_db: f32,
    // ...
}

/// 流式增强器
pub struct SoundExProcessor { /* ... */ }

impl SoundExProcessor {
    /// 创建处理器实例
    pub fn new(config: &SoundExConfig) -> Result<Self, SoundExError>;

    /// 处理一帧音频（流式）
    /// 输入和输出长度相同
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<ProcessInfo, SoundExError>;

    /// 处理完整 buffer（文件级）
    pub fn process_buffer(&mut self, input: &[f32]) -> Result<Vec<f32>, SoundExError>;

    /// 重置内部状态
    pub fn reset(&mut self);

    /// 获取当前帧是否被 bypass
    pub fn is_bypassed(&self) -> bool;
}

/// 处理信息
pub struct ProcessInfo {
    pub bypassed: bool,
    pub detected_bandwidth_hz: f32,
    pub enhancement_gain_db: f32,
}
```
