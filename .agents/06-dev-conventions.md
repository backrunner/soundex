# SoundEx 开发规范

## Rust 代码规范

### 文件与模块规模控制

- **单文件上限 300 行**（含注释和测试）；超过时必须拆分为子模块
- **单模块（含子模块）上限 ~1000 行**；超过时考虑提升为独立 crate 或拆分为平级模块
- 拆分原则：
  - 按职责拆分（如 `stft.rs` 拆为 `stft/analyzer.rs` + `stft/synthesizer.rs`）
  - 按抽象层级拆分（接口定义 vs 实现细节）
  - 测试代码计入行数，若测试过长可移至 `tests/` 或 `#[cfg(test)] mod tests` 独立文件
- 每个文件应有单一明确职责，文件头 `//!` 注释说明本文件做什么
- 避免「上帝模块」：如果一个 `mod.rs` 需要 re-export 超过 10 个公共类型，说明模块边界需要重新划分

### 格式化与 Lint

- 使用 `rustfmt` 默认配置（`cargo fmt`）
- 使用 `clippy` 并开启 `pedantic` 组（允许部分合理豁免）
- CI 中强制执行：`cargo fmt --check` + `cargo clippy -- -D warnings`

### 命名约定

| 类型 | 风格 | 示例 |
|------|------|------|
| Crate | kebab-case | `soundex-core` |
| Module | snake_case | `bandwidth_detector` |
| Struct/Enum/Trait | PascalCase | `SoundExProcessor` |
| Function/Method | snake_case | `process_frame` |
| Constant | SCREAMING_SNAKE | `MAX_FFT_SIZE` |
| Type Parameter | 单字母 PascalCase | `T`, `E` |
| Lifetime | 短小写 | `'a` |

### 错误处理

- **Lib crate**（soundex-core, soundex-dsp）：使用 `thiserror` 定义错误类型
- **Bin crate**（soundex-cli）：使用 `anyhow` 简化错误传播
- 禁止在公共 API 中 `unwrap()`/`expect()`（测试代码除外）
- 内部实现可使用 `?` 传播

### 文档注释

- 所有 `pub` 项必须有 `///` 文档注释
- 文档注释包含：功能说明、参数含义、返回值、错误条件、使用示例
- 模块级文档使用 `//!`
- 复杂算法需注明参考文献

```rust
/// 检测输入频谱的有效带宽。
///
/// 从高频向低频扫描功率谱，找到能量骤降至噪底以下的拐点频率。
/// 使用指数平滑避免逐帧抖动。
///
/// # Arguments
/// * `magnitude_db` - 对数幅度谱 (dB)，长度为 fft_size/2 + 1
/// * `sample_rate` - 采样率 (Hz)
///
/// # Returns
/// 检测到的有效带宽 (Hz)
///
/// # References
/// 基于频谱能量衰减拐点的带宽估计方法
pub fn detect(&mut self, magnitude_db: &[f32], sample_rate: u32) -> f32 { ... }
```

###  unsafe 使用

- 原则上禁止 `unsafe`
- 若性能关键路径确需（如 SIMD），须：
  - 封装在独立模块中
  - 添加 `// SAFETY:` 注释说明不变量
  - 有对应的 Miri 测试

### 依赖管理

- 新增依赖需在 PR 中说明理由
- 优先选择：纯 Rust > C 绑定；活跃维护 > 停滞项目
- 锁定 `Cargo.lock`（bin crate），lib crate 使用宽松版本约束

## Python 代码规范（训练工程）

### 文件与模块规模控制

- **单文件上限 400 行**；超过时拆分为子模块（如 `models/generator.py` 拆为 `models/generator/encoder.py` + `decoder.py`）
- 每个文件单一职责：一个文件 = 一个类/一组紧密相关的函数
- 配置、数据、模型、训练逻辑必须分离在不同文件中

### 格式化与 Lint

- 使用 `ruff` 进行格式化和 lint（替代 black + flake8 + isort）
- 行宽 100 字符
- 类型注解：所有函数签名必须有 type hints

### 命名约定

| 类型 | 风格 | 示例 |
|------|------|------|
| 文件/模块 | snake_case | `train.py`, `dataset.py` |
| 类 | PascalCase | `SoundExGenerator` |
| 函数/变量 | snake_case | `compute_loss` |
| 常量 | SCREAMING_SNAKE | `SAMPLE_RATE` |
| 配置键 | snake_case | `batch_size` |

### 项目结构

- 使用 Hydra 或 YAML 管理配置
- 训练脚本支持命令行覆盖配置
- 所有实验可复现（固定 seed、记录 config）

## 测试规范

### Rust 测试

| 层级 | 位置 | 工具 | 覆盖目标 |
|------|------|------|----------|
| 单元测试 | 各模块 `#[cfg(test)]` | `cargo test` | DSP 算法、边界条件 |
| 集成测试 | `tests/` 目录 | `cargo test --test` | 端到端管线 |
| Benchmark | `benches/` 目录 | Criterion | 性能回归检测 |
| 文档测试 | rustdoc 示例 | `cargo test --doc` | API 示例正确性 |

### 测试原则

1. **DSP 算法**：必须有已知答案测试（如正弦波的 STFT 峰值位置）
2. **Roundtrip 测试**：STFT 分析→合成应恢复原始信号（SNR > 120dB）
3. **边界测试**：空输入、极短输入、全零、全满幅
4. **回归测试**：模型推理结果与 golden file 对比
5. **性能测试**：设定阈值，CI 中检测性能回归

### Benchmark 体系

```rust
// benches/bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_stft_roundtrip(c: &mut Criterion) {
    c.bench_function("stft_roundtrip_1024", |b| {
        b.iter(|| { /* STFT analyze + synthesize */ })
    });
}

fn bench_bandwidth_detection(c: &mut Criterion) {
    c.bench_function("bandwidth_detect", |b| {
        b.iter(|| { /* 带宽检测 */ })
    });
}

fn bench_inference(c: &mut Criterion) {
    c.bench_function("model_inference", |b| {
        b.iter(|| { /* 单帧推理 */ })
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    c.bench_function("full_pipeline_per_frame", |b| {
        b.iter(|| { /* 完整处理管线 */ })
    });
}
```

### 性能基线

| Benchmark | 目标 | 平台 |
|-----------|------|------|
| STFT roundtrip (1024-pt) | < 50μs | Apple M1 |
| Bandwidth detection | < 10μs | Apple M1 |
| Model inference (1 frame) | < 5ms | Apple M1 CPU |
| Full pipeline (1 frame) | < 6ms | Apple M1 CPU |
| Full song (4min) | < 30s | Apple M1 |

### Python 测试

- 数据预处理：验证输出 shape、数值范围
- 模型：验证参数量、forward pass 输出 shape
- 训练：1-2 step smoke test 确保无报错
- 导出：ONNX checker 验证 + 与 PyTorch 输出对比

## Git 工作流

### 分支策略

- `main`：稳定分支，始终可编译通过
- `dev`：开发分支
- `feature/xxx`：功能分支
- `fix/xxx`：修复分支
- `training/xxx`：训练实验分支

### PR 规范

- 每个 PR 对应一个明确的功能/修复
- PR 标题遵循提交规范格式
- 必须通过 CI（fmt + clippy + test）
- 涉及 API 变更需更新文档

## 版本管理

- 遵循 Semantic Versioning (SemVer)
- 初始版本：0.1.0
- API 不稳定期间（0.x.y）：minor 版本可包含 breaking changes
- 模型文件使用 git-lfs 管理
- 训练 checkpoint 不入库（.gitignore）
