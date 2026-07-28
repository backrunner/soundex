# SoundEx 实现路线图

## 里程碑总览

| 里程碑 | 名称 | 预计周期 | 核心交付 |
|--------|------|----------|----------|
| M0 | 工程骨架 | 1 周 | Workspace 搭建、CI、基础结构 |
| M1 | DSP 核心 | 2 周 | STFT、带宽检测、后处理算法 |
| M2 | 推理集成 | 1 周 | ort 集成、模型加载、推理管线 |
| M3 | 模型训练 | 3-4 周 | 数据预处理、训练、评估、ONNX 导出 |
| M4 | CLI + 集成 | 1 周 | CLI 工具、端到端验证 |
| M5 | 发布准备 | 1 周 | 文档、benchmark、发布 |

---

## M0: 工程骨架（第 1 周）

### 目标
建立完整的工程基础设施，确保团队可以立即开始开发。

### 任务清单

- [ ] 初始化 Cargo workspace（root Cargo.toml + 3 个 crate）
- [ ] 配置 `.gitignore`（Rust + Python + 模型文件）
- [ ] 配置 `.gitattributes`（git-lfs for *.onnx）
- [ ] 添加 Apache 2.0 LICENSE 文件
- [ ] 创建 `soundex-dsp` crate 骨架（lib.rs + 模块声明）
- [ ] 创建 `soundex-core` crate 骨架（lib.rs + error.rs + config.rs）
- [ ] 创建 `soundex-cli` crate 骨架（main.rs + clap 配置）
- [ ] 创建 `training/` 目录结构（configs, data, models）
- [ ] 配置 CI（GitHub Actions）：
  - cargo fmt --check
  - cargo clippy
  - cargo test
  - cargo build --release
- [ ] 配置 Criterion benchmark 框架
- [ ] 编写初始 rustdoc 文档框架

### 验收标准
- `cargo build` 通过
- `cargo test` 通过（空测试）
- `cargo clippy` 无警告
- CI pipeline 绿色

---

## M1: DSP 核心（第 2-3 周）

### 目标
实现所有 DSP 算法模块，通过单元测试验证正确性。

### 任务清单

- [ ] 实现 `window.rs`：Hann 窗生成
- [ ] 实现 `stft.rs`：
  - STFT 分析（加窗 + Real FFT）
  - iSTFT 合成（Inverse FFT + 加窗 + COLA）
  - 单元测试：分析→合成 roundtrip 误差 < 1e-6
- [ ] 实现 `bandwidth.rs`：
  - 带宽检测算法
  - 指数平滑
  - Hangover 逻辑
  - 单元测试：合成截断信号验证检测准确性
- [ ] 实现 `loudness.rs`：
  - RMS 计算
  - 响度匹配增益
  - 单元测试
- [ ] 实现 `phase.rs`：
  - 相位平滑（复数域插值）
  - 相位传播（谐波外推）
  - 单元测试
- [ ] 实现 `crossover.rs`：
  - Raised cosine 过渡带
  - 频谱混合
  - 单元测试：验证过渡带平滑性
- [ ] 实现 `limiter.rs`：
  - 软限幅
  - 单元测试：验证无削波
- [ ] Benchmark：STFT roundtrip 性能、带宽检测性能

### 验收标准
- 所有 DSP 模块单元测试通过
- STFT roundtrip SNR > 120dB
- 带宽检测对合成截断信号准确率 > 95%
- Benchmark 报告生成

---

## M2: 推理集成（第 4 周）

### 目标
集成 ort 推理引擎，建立完整的处理管线（使用 dummy 模型）。

### 任务清单

- [ ] 实现 `inference.rs`：
  - ort Session 创建与配置
  - 模型加载（从文件）
  - 推理执行（tensor 构造 + run）
  - 错误处理
- [ ] 实现 `stream.rs`：
  - Ring buffer 管理
  - Overlap-add 状态机
  - 帧对齐逻辑
- [ ] 实现 `processor.rs`：
  - 组合 DSP + 推理的完整管线
  - `process_frame()` 流式接口
  - `process_buffer()` 文件级接口
  - Bypass 逻辑
- [ ] 创建 dummy ONNX 模型（用于测试，如恒等映射）
- [ ] 集成测试：dummy 模型下 process_buffer 输出 == 输入
- [ ] Feature flag 配置：ort-backend / tract-backend

### 验收标准
- 使用 dummy 模型的端到端处理通过
- 流式处理与 buffer 处理结果一致
- Bypass 逻辑正确工作
- 推理延迟 benchmark 建立基线

---

## M3: 模型训练（第 5-8 周）

### 目标
完成数据预处理、模型训练、评估、ONNX 导出的完整 pipeline。

### 任务清单

**数据准备（第 5 周）**：
- [ ] 编写 `preprocess_musdb.py`：
  - 加载 MUSDB18-HQ mixture.wav
  - 多码率 MP3 压缩 + 解码
  - 切片 + 数据增强
  - 输出：配对数据集 (degraded, clean)
- [ ] 编写 `dataset.py`：PyTorch Dataset + DataLoader
- [ ] 本地验证数据 pipeline 正确性

**模型实现（第 5-6 周）**：
- [ ] 实现 `generator.py`：
  - DWSBlock、InvertedResidual
  - 双流 U-Net（幅度 + 相位）
  - 参数量验证 < 2M
- [ ] 实现 `discriminator.py`：Multi-Scale Discriminator
- [ ] 实现 `losses.py`：
  - Multi-resolution STFT loss
  - Adversarial loss
  - Feature matching loss
  - Phase consistency loss

**训练（第 6-7 周）**：
- [ ] 编写 `train.py`：
  - 三阶段训练策略
  - 日志（TensorBoard/WandB）
  - Checkpoint 保存/恢复
- [ ] 本地小规模验证（1-2 epoch）
- [ ] 编写 `Dockerfile`（AutoDL 迁移）
- [ ] 全量训练（MUSDB18-HQ）
- [ ] 超参数调优

**评估与导出（第 8 周）**：
- [ ] 编写 `evaluate.py`：LSD、ViSQOL、PESQ 等指标
- [ ] 在测试集上评估
- [ ] 编写 `export_onnx.py`：导出 + 验证 + 可选量化
- [ ] 在 Rust 端验证 ONNX 模型推理结果与 PyTorch 一致

### 验收标准
- 模型参数量 < 2M
- ONNX 文件 < 8MB
- 测试集 LSD 显著优于 baseline（无增强）
- 主观听感有明显改善
- Rust 端推理结果与 PyTorch 误差 < 1e-5

---

## M4: CLI + 集成（第 9 周）

### 目标
完成 CLI 工具，实现端到端的文件处理验证。

### 任务清单

- [ ] 实现 CLI 完整功能：
  - symphonia 解码（MP3/AAC/FLAC/WAV/OGG）
  - 调用 soundex-core 处理
  - hound WAV 输出（16/24/32-bit）
  - 进度条显示
  - 错误处理与用户提示
- [ ] 端到端测试：
  - MP3 → 增强 → WAV
  - 验证输出质量
  - 对比原始/压缩/增强三者频谱
- [ ] 编写集成测试（使用短测试音频）
- [ ] 性能测试：处理一首完整歌曲的耗时

### 验收标准
- `soundex test.mp3 -o output.wav` 正常工作
- 输出 WAV 可正常播放
- 频谱对比显示高频有明显恢复
- 处理速度 > 10x 实时

---

## M5: 发布准备（第 10 周）

### 目标
完善文档、benchmark、发布到 crates.io。

### 任务清单

- [ ] 完善 rustdoc 文档（所有公共 API）
- [ ] 编写 README.md（项目介绍、快速开始、API 示例）
- [ ] 完善 benchmark 报告：
  - 单帧延迟（不同平台）
  - 吞吐量
  - 内存占用
  - 与 baseline 对比
- [ ] 发布前检查：
  - `cargo publish --dry-run`
  - 所有测试通过
  - clippy 无警告
- [ ] Tag v0.1.0 发布
- [ ] 可选：提交到 crates.io

### 验收标准
- 文档完整，新用户可在 5 分钟内上手
- Benchmark 数据达标
- crates.io 发布成功（或至少 dry-run 通过）

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型质量不达预期 | 增强效果不明显 | 多轮超参调优；增加训练数据；调整损失权重 |
| ONNX 算子不兼容 | 无法导出/推理 | 避免使用自定义算子；提前验证 opset 兼容性 |
| 实时性不达标 | 无法用于流式场景 | INT8 量化；减少模型层数；优化 STFT 实现 |
| MUSDB18-HQ 数据不足 | 泛化能力差 | 混合 Slakh2100 / MedleyDB 整轨缩混；数据增强 |
| ort 版本不稳定 (rc) | API 变更 | 锁定版本；封装适配层 |
