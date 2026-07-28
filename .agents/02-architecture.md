# SoundEx 架构设计

## Workspace 结构

```
soundex/                          # Cargo workspace root
├── Cargo.toml                    # [workspace] members = ["crates/*"]
├── crates/
│   ├── soundex-core/             # 核心推理 lib
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # 公共 API 入口
│   │       ├── config.rs         # 配置结构
│   │       ├── processor.rs      # SoundExProcessor 主逻辑
│   │       ├── inference.rs      # ONNX 推理封装
│   │       ├── stream.rs         # 流式处理状态机
│   │       └── error.rs          # 错误类型
│   ├── soundex-dsp/              # DSP 算法库
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── stft.rs           # STFT / iSTFT
│   │       ├── bandwidth.rs      # 带宽检测（门禁）
│   │       ├── loudness.rs       # 响度分析与匹配
│   │       ├── phase.rs          # 相位处理与平滑
│   │       ├── crossover.rs      # 频段平滑过渡
│   │       ├── limiter.rs        # 限幅保护
│   │       └── window.rs         # 窗函数
│   └── soundex-cli/              # CLI 工具
│       ├── Cargo.toml
│       └── src/
│           └── main.rs           # CLI 入口
├── training/                     # Python 训练工程（独立于 Cargo）
│   ├── configs/
│   │   └── default.yaml
│   ├── data/
│   │   ├── audio_prep.py         # 共享 MP3 降质 / 切片
│   │   ├── preprocess_musdb.py   # MUSDB18-HQ 整轨 mixture
│   │   ├── preprocess_slakh.py   # Slakh2100 整轨 mix（忽略 stems）
│   │   ├── preprocess_medleydb.py# MedleyDB 整轨 *_MIX（忽略分轨）
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/
│   │   ├── generator.py          # 生成器网络
│   │   ├── discriminator.py      # 判别器网络
│   │   └── losses.py             # 损失函数
│   ├── train.py                  # 训练入口
│   ├── evaluate.py               # 评估脚本
│   ├── export_onnx.py            # ONNX 导出
│   ├── requirements.txt
│   └── Dockerfile                # AutoDL 迁移
├── models/                       # 预训练模型 (git-lfs)
│   └── soundex-v1.onnx
├── tests/                        # 集成测试
│   ├── fixtures/                 # 测试音频片段
│   └── integration_test.rs
├── benches/                      # Benchmark
│   └── bench.rs
├── .gitignore
├── .gitattributes                # git-lfs 配置
├── LICENSE                       # Apache 2.0
└── .agents/                      # 项目规划文档
```

## 模块职责与依赖关系

```
┌─────────────────────────────────────────────────────┐
│                   soundex-cli                        │
│  (symphonia 解码 → PCM → soundex-core → WAV 输出)   │
└──────────────────────┬──────────────────────────────┘
                       │ 依赖
                       ▼
┌─────────────────────────────────────────────────────┐
│                   soundex-core                       │
│  (模型加载、推理调度、流式状态管理、API 暴露)         │
└──────────────────────┬──────────────────────────────┘
                       │ 依赖
                       ▼
┌─────────────────────────────────────────────────────┐
│                   soundex-dsp                        │
│  (STFT/iSTFT、带宽检测、响度匹配、相位修复、限幅)    │
└─────────────────────────────────────────────────────┘
```

- `soundex-core` 依赖 `soundex-dsp` + `ort`
- `soundex-cli` 依赖 `soundex-core` + `symphonia` + `hound` + `clap`
- `soundex-dsp` 依赖 `rustfft` + `realfft` + `dasp`（纯计算，无 I/O）

## 数据流

### 流式处理管线

```
Input PCM Frame (512 samples, f32)
        │
        ▼
┌─────────────────┐
│  Ring Buffer    │  ← 累积至完整分析帧 (1024 samples)
│  (overlap-add)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  STFT           │  → 1024-pt FFT, Hann window
│  Analysis       │  → 输出: log-magnitude + phase
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bandwidth      │  → 检测高频是否截断
│  Detector       │  → 若未截断: bypass
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ONNX Model     │  → 输入: 完整 log-mag + phase
│  Inference      │  → 输出: 候选完整 log-mag + phase
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DSP Post-      │  → 响度匹配
│  Processing     │  → 相位平滑
│                 │  → 交叉渐变
│                 │  → 限幅
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  iSTFT          │  → 重建时域信号
│  Synthesis      │  → overlap-add
└────────┬────────┘
         │
         ▼
Output PCM Frame (512 samples, f32)
```

### 文件处理管线

```
Input File (MP3/AAC/FLAC/WAV/OGG)
        │
        ▼  [soundex-cli: symphonia 解码]
PCM Buffer (full length, f32)
        │
        ▼  [soundex-core: process_buffer]
Enhanced PCM Buffer
        │
        ▼  [soundex-cli: hound 编码]
Output WAV File
```

## 核心接口设计

### soundex-dsp

```rust
/// STFT 分析器
pub struct StftAnalyzer {
    fft_size: usize,      // 1024
    hop_size: usize,      // 512
    window: Vec<f32>,     // Hann
}

impl StftAnalyzer {
    pub fn new(fft_size: usize, hop_size: usize) -> Self;
    /// 分析一帧，返回 (log_magnitude, phase)
    pub fn analyze(&self, frame: &[f32]) -> (Vec<f32>, Vec<f32>);
    /// 从幅度+相位重建时域信号
    pub fn synthesize(&self, magnitude: &[f32], phase: &[f32]) -> Vec<f32>;
}

/// 带宽检测器
pub struct BandwidthDetector {
    threshold_db: f32,
    smoothing_frames: usize,
}

impl BandwidthDetector {
    pub fn new(threshold_db: f32) -> Self;
    /// 检测当前帧的有效带宽 (Hz)
    pub fn detect(&mut self, magnitude_spectrum: &[f32], sample_rate: u32) -> f32;
    /// 判断是否需要增强
    pub fn needs_enhancement(&self, detected_bandwidth: f32, nyquist: f32) -> bool;
}

/// 响度匹配器
pub struct LoudnessMatcher { /* ... */ }

/// 相位平滑器
pub struct PhaseSmoother { /* ... */ }

/// 频段交叉渐变
pub struct CrossoverBlend { /* ... */ }

/// 限幅器
pub struct Limiter { /* ... */ }
```

### soundex-core

```rust
/// 推理引擎封装
pub struct InferenceEngine {
    session: ort::Session,
}

impl InferenceEngine {
    pub fn load(model_path: &Path) -> Result<Self, SoundExError>;
    /// 执行推理：输入低频频谱特征，输出高频频谱特征
    pub fn infer(&self, input_features: &[f32]) -> Result<Vec<f32>, SoundExError>;
}

/// 流式处理器（组合 DSP + 推理）
pub struct SoundExProcessor {
    engine: InferenceEngine,
    stft: StftAnalyzer,
    detector: BandwidthDetector,
    loudness: LoudnessMatcher,
    phase_smoother: PhaseSmoother,
    crossover: CrossoverBlend,
    limiter: Limiter,
    ring_buffer: RingBuffer,
    state: ProcessorState,
}
```

## Feature Flags

```toml
# soundex-core/Cargo.toml
[features]
default = ["ort-backend"]
ort-backend = ["ort"]           # ONNX Runtime 后端
tract-backend = ["tract-onnx"]  # 纯 Rust 后端（嵌入式）
embedded-model = []             # 嵌入默认模型到二进制
```

## 线程模型

- `SoundExProcessor` 本身 **非 Send/Sync**（单线程音频回调使用）
- 若需多线程，调用方通过 `Mutex<SoundExProcessor>` 或 channel 包装
- 推理引擎内部不持有锁，由调用方保证串行调用
- CLI 使用单线程处理（文件级），无需并发

## 错误处理

```rust
#[derive(Debug, thiserror::Error)]
pub enum SoundExError {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Sample rate mismatch: expected {expected}, got {got}")]
    SampleRateMismatch { expected: u32, got: u32 },
}
```
