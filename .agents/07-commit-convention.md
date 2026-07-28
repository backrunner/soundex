# SoundEx 提交规范

## 提交格式

```
<type>(<component>): <description>

[optional body]

[optional footer]
```

### 格式说明

- **type**：提交类型（必填）
- **component**：影响的组件（必填）
- **description**：简短描述（必填），祈使句，首字母小写，不加句号
- **body**：详细说明（可选），解释 why 而非 what
- **footer**：关联 issue 或 BREAKING CHANGE（可选）

### Type 列表

| Type | 说明 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| perf | 性能优化 |
| refactor | 重构（不改变功能） |
| test | 添加/修改测试 |
| bench | 添加/修改 benchmark |
| docs | 文档变更 |
| style | 代码格式（不影响逻辑） |
| build | 构建系统/依赖变更 |
| ci | CI 配置变更 |
| chore | 其他杂务 |
| train | 训练相关（模型、数据、配置） |

### Component 列表

| Component | 对应范围 |
|-----------|----------|
| core | soundex-core crate |
| dsp | soundex-dsp crate |
| cli | soundex-cli crate |
| train | training/ 训练工程 |
| model | 模型文件、架构定义 |
| data | 数据预处理 |
| infra | CI/CD、构建配置 |
| agents | .agents/ 规划文档 |
| deps | 依赖更新 |

### 示例

```
feat(core): add streaming process_frame API

Implement the frame-by-frame processing interface for real-time
audio enhancement. Includes ring buffer management and overlap-add
state machine.

Closes #12
```

```
fix(dsp): correct phase wrap-around in crossover blend

The previous implementation used linear interpolation on raw phase
values, which caused discontinuities at ±π boundaries. Now uses
complex-domain interpolation.
```

```
perf(dsp): optimize STFT with pre-allocated buffers

Reduce per-frame allocation by reusing FFT scratch buffers.
Benchmark shows 30% improvement in STFT roundtrip time.
```

```
train(model): add multi-bitrate MP3 augmentation

Support 64k/128k/192k/320k MP3 compression simulation during
training to improve model robustness across quality levels.
```

```
feat(cli)!: change output format option syntax

BREAKING CHANGE: --format is replaced by --bits-per-sample
to be more explicit about WAV output configuration.
```

## 提交原则

1. **原子性**：每个提交只做一件事
2. **可编译**：每个提交后代码必须能编译通过
3. **描述性**：description 应清楚表达"做了什么"
4. **一致性**：使用祈使句（"add" 而非 "added"）
5. **长度**：description 不超过 72 字符；body 每行不超过 100 字符

## Apache 2.0 开源适配

### 文件头

所有源代码文件应包含 Apache 2.0 许可头（可选，SPDX 标识符方式）：

```rust
// SPDX-License-Identifier: Apache-2.0
```

或完整头（推荐用于主要源文件）：

```rust
// Copyright 2026 SoundEx Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

### .gitignore 配置

```gitignore
# Rust
/target/
**/*.rs.bk
Cargo.lock  # lib crate 不锁定（bin crate 需要锁定）

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
*.egg

# Training artifacts
training/checkpoints/
training/logs/
training/outputs/
*.pth
*.pt
!training/export_onnx.py

# Data (不入库)
data/
*.wav
*.mp3
*.flac
!tests/fixtures/*.wav  # 测试用短音频除外

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Models (git-lfs 管理)
# 见 .gitattributes

# OS
Thumbs.db
```

### .gitattributes 配置

```gitattributes
# Git LFS for model files
*.onnx filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text

# Ensure consistent line endings
*.rs text eol=lf
*.py text eol=lf
*.toml text eol=lf
*.yaml text eol=lf
*.md text eol=lf

# Binary files
*.wav binary
*.mp3 binary
*.png binary
```

### 不应包含在仓库中的内容

| 内容 | 理由 |
|------|------|
| 训练数据集原始文件 | 体积过大，需单独下载 |
| 训练 checkpoint (.pth/.pt) | 体积大，通过 release 分发 |
| Cargo.lock (lib crate) | Rust 社区惯例 |
| IDE 配置 | 个人偏好 |
| 操作系统文件 | 无关 |
| 编译产物 | 可重新生成 |
| 密钥/Token | 安全 |

### 贡献者协议

- 项目使用 Apache 2.0，贡献即表示同意以该许可发布
- 无需额外 CLA（Contributor License Agreement）
- 在 README 中注明贡献即授权
