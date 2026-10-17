# SoundEx

[English](README.md) · [CI](https://github.com/backrunner/soundex/actions/workflows/ci.yml) · [Apache-2.0](LICENSE)

SoundEx 是实验性的压缩音频恢复库，目标是对被裁减或损伤的高频频谱进行 inpainting，
结合 DSP 融合和伪影抑制，让最终音频尽量接近原始无损参考。项目使用 Rust 实现
DSP、ONNX 推理和实时流适配，同时提供文件处理 CLI 和 PyTorch 训练工程。

**当前状态：**源码与接入 demo 可用，尚未发布合格的音质恢复权重。
仓库附带的合成恒等测试模型只用于验证接入，不代表音质提升。
详细进展见[模型说明](models/README.md)。

[P25 部署与覆盖审计](docs/context-deployment-audit.md)已接通实验模型的 Rust 前端 →
ORT → 生产 DSP。审计发现 P24 实际音乐独立监督区间仅约 8 分钟，且原生成预算
没有约束局部残差路径；新增完整残差预算候选已通过结构测试，尚未训练或提升默认权重。

[P24 生成能量对照](docs/generation-budget-study.md)固定 640 宽度大模型，分别验证
loss 归一化与输入驱动能量分支；训练和完整回放验证已于 10 月 17 日结束。
能量分支改善了旧弱频带的相对误差，但相对恒等 DSP，音乐波形误差及部分原音
扰动仍增加，长测也出现超时。前端差异与合成一致性分别保留，尚无合格新权重。

[P23 完整预算对照](docs/full-capacity-study.md)已完成 129,536 次更新和 4,450 次
实际 Rust 角色回放。长训练降低了训练误差，但所有规模均未通过完整质量检查。
多数开发音乐片段的 paired 代理指标改善，少数极弱高频片段的严重退化主导了均值；
[绝对误差审计](docs/capacity-outlier-audit.md)确认有小幅高频误生成及相对评分放大，
可闻严重程度尚未验证，目前没有合格新权重。

[P22 延迟优先扩容](docs/latency-capacity-study.md)已完成两个种子、每配置 512 步的校准，
最大 1,192 万参数。计算成本通过初筛，但恢复收益较小，原音损伤抵消了收益，
没有候选通过训练优化检查。[P21](docs/capacity-calibration-study.md)也在校准后停止，
未进入长训练。这些短期训练结果不能证明容量上限，也不代表留出音质验证。

经核查的曲库共 300 首原生无损音乐，另有 VCTK 语音。原始音频、配对数据和实验检查点
保留在本地。使用入口见[训练指南](training/README.md)，来源与授权记录见
[数据文档](docs/datasets.md)，历次方法和结果集中在[实验索引](docs/research.md)。

## 直接运行

需要 **Rust 1.88+** 和 C/C++ 编译工具链。默认构建会下载 ONNX Runtime，首次构建需联网。
在仓库根目录执行：

```bash
git clone https://github.com/backrunner/soundex.git
cd soundex
cargo run --locked -p soundex-core --example offline_demo
cargo run --locked -p soundex-core --example realtime_demo
```

离线 demo 生成两秒 48 kHz 双声道测试音频，通过内置 ONNX 模型处理，保存至
`target/demo/input.wav` 与 `target/demo/output.wav`。实时 demo 模拟每块 128 帧的回调，
输出截止时间、回退和 worker 统计，不打开声卡。

使用自己的合格 **256/128** 模型：

```bash
cargo run --locked --release -p soundex-core --example offline_demo -- /path/to/model.onnx target/demo
cargo run --locked --release -p soundex-core --example realtime_demo -- /path/to/model.onnx
```

## 文件处理

```bash
# 无需模型的分析
cargo run --locked -p soundex-cli -- target/demo/input.wav --dry-run
# 用测试模型验证完整处理流程
cargo run --locked -p soundex-cli -- target/demo/input.wav \
  --model tests/fixtures/low-latency-identity.onnx \
  --output target/demo/cli-output.wav --bits 24
```

实际恢复时应换成通过验证的训练模型。输入支持 MP3、AAC、FLAC、WAV、OGG Vorbis，
输出支持 16/24 位 PCM 和 32 位浮点 WAV。

## 实时流接入

在开始播放前于控制线程创建 `RealtimeProcessor`。默认先完成 worker 推理预热，
仅构造线程等待就绪（最长 10 秒）；音频回调仅调用 `process(input, output)`。
回调不执行推理、不分配堆内存、不加互斥锁；独立 worker 通过有界队列推理。
结果超时或异常时使用时间对齐的原音回退，并通过双声道联动渐变、输入清理与最终限幅
保持输出连续。创建、替换、销毁及 `shutdown` 必须放在控制线程。

实时配置：单/双声道 **44.1/48 kHz，FFT 256 / hop 128**，推荐与 hop 对齐的 **128 帧回调**。
固定新增延迟为 **256 帧**：44.1 kHz 下 **5.80 ms**，48 kHz 下 **5.33 ms**。
软件新增延迟目标 **低于 8 ms**、红线 **低于 10 ms**，包含回调耗时与额外接入缓冲。
声卡与驱动缓冲应另行测量；固定帧延迟不等于真实设备或完整链路的实测延迟。

[库接入示例与契约](docs/library.md) · [实时验收标准](docs/realtime-standard.md) ·
[时延与资源 benchmark](docs/performance.md)

文件处理使用同步 `SoundExProcessor`；它直接调用 ORT，不应放进音频回调。

## 训练与验证

[训练说明](training/README.md)包含环境安装、混音数据预处理、固定数据划分、CPU/MPS/CUDA
与导出命令。默认模型契约为 FP32 `[batch, 2, 1, 129]`，仅 batch 动态，
checkpoint schema 为 1.2，artifact schema 为 1.3。导出必须通过 ONNX 与 ORT 一致性检查；
Rust 在处理前验证模型元数据与张量契约。
可选[因果上下文](docs/causal-context-20261011.md)使用 artifact 1.4：九帧历史/当前输入、
一帧输出，不读取未来；host 保存每声道历史，八个 hop 的增强预热期间保持对齐原音。

[一致性误差预算](docs/parity.md) · [评估流程](training/evaluation/README.md) · [模型卡](models/MODEL_CARD.template.md) ·
[权重发布清单](legal/MODEL_RELEASE_CHECKLIST.md)

构建与检查命令见英文 README。[贡献说明](CONTRIBUTING.md)约定测试与提交格式。

## 开源许可

源码、文档、demo、合成测试图与**正式发布权重**统一采用 **[Apache-2.0](LICENSE)**。
正式权重允许其他开源项目集成、随软件分发、修改、微调和商用；发布时附模型卡、
校验值及适用的署名声明。

训练音频和依赖保留各自许可。现有 MUSDB 候选保留为未发布的研究记录，退出正式权重
路线。训练不设数据集或许可白名单，可直接扩充音频目录；正式发布时核对授权和模型来源。
依赖说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

[权重许可范围](MODEL_WEIGHTS_LICENSE.md) · [NOTICE](NOTICE) · [许可说明](legal/LICENSING.md) ·
[训练数据政策](legal/TRAINING_DATA.md) · [数据来源调查](docs/datasets.md)

Copyright 2026 BackRunner and SoundEx Contributors.
