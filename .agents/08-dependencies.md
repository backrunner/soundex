# SoundEx 技术依赖

## Rust 依赖

### 推理引擎

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **ort** | 2.0.0-rc.9+ | ONNX Runtime 推理 | 多 EP 支持(CPU/CUDA/CoreML)、活跃维护、生产验证(Twitter/Google)、自动下载 ORT 二进制 |
| tract-onnx | 0.21+ | 备选纯 Rust 推理 | 无外部依赖、极轻量、适合嵌入式；feature-gated |

**对比决策**：

| 维度 | ort | tract |
|------|-----|-------|
| 性能 | 高（C++ ORT 后端） | 中高（纯 Rust SIMD） |
| 算子覆盖 | 完整 ONNX opset | 主流算子 |
| GPU 加速 | CUDA/CoreML/DirectML | 无 |
| 二进制大小 | 较大（链接 ORT） | 极小 |
| 维护 | 活跃（pykeio） | 活跃（Sonos） |
| 结论 | **主选** | 嵌入式备选 |

### 音频解码（CLI 专用）

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **symphonia** | 0.5.x | MP3/AAC/FLAC/WAV/OGG 解码 | 纯 Rust、SIMD 优化(SSE/AVX/Neon)、性能接近 FFmpeg(±15%)、100% safe |

**对比决策**：

| 维度 | symphonia | rodio | minimp3 |
|------|-----------|-------|---------|
| 格式支持 | MP3/AAC/FLAC/WAV/OGG/ALAC | 同(封装 symphonia) | 仅 MP3 |
| 定位 | 解码库 | 播放库 | 极简单 MP3 |
| 依赖量 | 中 | 大(含 cpal) | 极小 |
| 适用 | **文件解码** | 播放场景 | 不够用 |

### WAV 输出

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **hound** | 3.5 | WAV 文件写入 | 零依赖、轻量、支持 16/24/32-bit PCM + Float、API 简洁 |

### FFT / 频谱分析

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **rustfft** | 6.2 | 通用 FFT | 任意长度、AVX/SSE/Neon 自动选择、纯 Rust、成熟稳定 |
| **realfft** | 3.5 | 实数 FFT 优化 | 基于 rustfft，real→complex 性能翻倍、API 友好 |

**对比决策**：

| 维度 | rustfft+realfft | microfft | 自研 |
|------|-----------------|----------|------|
| 任意长度 | 是 | 仅 2^n | 可选 |
| SIMD | AVX/SSE/Neon | 无 | 需手动 |
| no_std | 否 | 是 | 可选 |
| 性能 | 高 | 中 | - |
| 维护 | 活跃 | 活跃 | - |
| 结论 | **选用** | 嵌入式备选 | 不必要 |

### 重采样

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **rubato** | 0.16 | 采样率转换 | 实时安全(无分配)、多种算法(Sinc/Fastest)、高质量 |

### DSP 辅助

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **dasp** | 0.11 | Ring buffer、Signal 抽象、帧操作 | 无分配、等待自由 ring buffer、音频专用原语 |

### CLI 框架

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **clap** | 4.x (derive) | 命令行参数解析 | 标准选择、derive 宏、自动帮助生成 |

### 错误处理

| Crate | 版本 | 用途 | 选型理由 |
|-------|------|------|----------|
| **thiserror** | 1.x/2.x | Lib 错误类型定义 | 派生宏、零开销、类型安全 |
| **anyhow** | 1.x | CLI 错误传播 | 灵活、支持上下文附加 |

### 其他工具

| Crate | 版本 | 用途 |
|-------|------|------|
| ndarray | 0.16 | 多维数组（推理 tensor） |
| log + env_logger | latest | 日志 |
| indicatif | 0.17 | CLI 进度条 |
| criterion | 0.5 | Benchmark 框架 |

### 完整 Cargo.toml 示例

```toml
# soundex-core/Cargo.toml
[package]
name = "soundex-core"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[dependencies]
soundex-dsp = { path = "../soundex-dsp" }
ort = { version = "2.0.0-rc.9", default-features = false, features = ["download-binaries"] }
ndarray = "0.16"
thiserror = "2"
log = "0.4"

[features]
default = ["ort-backend"]
ort-backend = ["ort"]
tract-backend = ["tract-onnx"]
embedded-model = []

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

```toml
# soundex-dsp/Cargo.toml
[package]
name = "soundex-dsp"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[dependencies]
rustfft = "6.2"
realfft = "3.5"
dasp = { version = "0.11", features = ["ring-buffer", "signal"] }

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
rand = "0.8"
```

```toml
# soundex-cli/Cargo.toml
[package]
name = "soundex-cli"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[[bin]]
name = "soundex"
path = "src/main.rs"

[dependencies]
soundex-core = { path = "../soundex-core" }
symphonia = { version = "0.5", features = ["mp3", "aac", "flac", "wav", "ogg", "vorbis"] }
hound = "3.5"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
indicatif = "0.17"
env_logger = "0.11"
log = "0.4"
```

## Python 依赖（训练工程）

### requirements.txt

```
# Core
torch>=2.2.0
torchaudio>=2.2.0
numpy>=1.24.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
audioread>=3.0.0

# Data
musdb>=0.5.0

# Training
pyyaml>=6.0
tensorboard>=2.14.0
tqdm>=4.65.0

# Evaluation
pesq>=0.0.4
pystoi>=0.4.0
visqol>=3.0.0  # 可选

# ONNX export
onnx>=1.15.0
onnxruntime>=1.17.0

# Tools
ruff>=0.4.0
```

### 训练平台对比

| 平台 | GPU | 适用场景 | 备注 |
|------|-----|----------|------|
| 本地 (macOS) | Apple Silicon MPS | 开发调试、小规模实验 | 使用 MPS 后端 |
| 本地 (Linux) | NVIDIA RTX 3090/4090 | 全量训练 | CUDA |
| AutoDL | A100/4090 | 全量训练、超参搜索 | 提供 Dockerfile |

### Dockerfile（AutoDL 迁移）

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training/

# 数据挂载点
VOLUME /data

CMD ["python", "training/train.py", "--config", "training/configs/default.yaml"]
```

## 系统依赖

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| ONNX Runtime 二进制 | ort 推理 | ort 自动下载（download-binaries feature） |
| libasound2-dev (Linux) | cpal（仅测试用） | apt install |
| ffmpeg/lame (训练) | MP3 压缩模拟 | apt install / brew install |
| git-lfs | 模型文件管理 | apt install git-lfs / brew install git-lfs |

## 最低系统要求

| 平台 | 最低版本 |
|------|----------|
| Rust | 1.88.0+ (edition 2021; ort rc.12) |
| Python | 3.10+ |
| macOS | 12.0+ (Apple Silicon) / 11.0+ (Intel) |
| Linux | glibc 2.31+ (Ubuntu 20.04+) |
| Windows | 10+ (MSVC 2019+) |
