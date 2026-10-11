# SoundEx

[English](README.md) · [CI](https://github.com/backrunner/soundex/actions/workflows/ci.yml) · [Apache-2.0](LICENSE)

SoundEx 是实验性的压缩音频恢复库，目标是对被裁减或损伤的高频频谱进行 inpainting，
结合 DSP 融合和伪影抑制，让最终音频尽量接近原始无损参考。项目使用 Rust 实现
DSP、ONNX 推理和实时流适配，同时提供文件处理 CLI 和 PyTorch 训练工程。

**当前状态：**源码与接入 demo 可用，尚未发布合格的音质恢复权重。
仓库附带的合成恒等测试模型只用于验证接入，不代表音质提升。
详细进展见[模型说明](models/README.md)。

[本次无损训练](docs/native-training-20261009.md)使用 300 首经核查的音乐及 VCTK 语音，
从随机权重开始，不使用 MUSDB、Slakh 或旧模型。原始音频、配对数据和训练检查点
保留在本地，仓库提供来源文档和可复现流程。
原训练已完成 200 轮；当前开展[损失审计与短程续训](docs/loss-audit-20261010.md)，
模型质量与发布资格仍以实际流处理验证为准。
[高频 inpainting 目标与模型/DSP 协作](docs/inpainting-objective.md)明确了新版 loss 和
三种对照路径；DSP 独立延展及组合模式目前仍为实验方案。
[P2 统一训练对照](docs/causal-training-20261012.md)六组已全部完成：旧主干、GRU、TCN
各训练两个种子，每组 4096 步，共用重建目标与实际 DSP 验证。
实验权重未替换默认模型。
[首种子复核](docs/causal-p2-review-20261012.md)：新主干优于同预算重训控制，
尚未全面超过原权重；原生弱高频材料仍存在多余生成能量的反例。
[P3 受控续训](docs/weak-band-study-20261012.md)已加入经过数值验证的过量能量约束、
恒等控制及泛化检查。[基线复核](docs/baseline-generalization-20261012.md)确认了增益策略的取舍
及恒等 DSP 路径的限幅失真，目前不足以把旧基线的问题直接归因为过拟合。

本轮[训练与结构对照](docs/optimization-20261011.md)验证复数重建、幅度/相位交互与实时启动预热。
后续 [V5 目标复核](docs/goal-refinement-20261011.md)验证重建能量校准和圆周相位输入，
包含录音留出复核与数据覆盖限制；实验权重继续保留在本地。

[V6 部署对齐续训](docs/deployment-refinement-20261011.md)进一步把边缘增益和重建后的瞬态约束纳入训练代理，继续通过实际 Rust 输出验证。
[架构调研与容量实验](docs/architecture-research-20261011.md)按实时预算筛选近期方法，
通过全频带残差分支分别比较输入表示和模型容量，并验证实际输出与部署成本。
三组新模型均完成 8 轮；[实测结果](docs/architecture-study-20261011.md)显示验证 loss
略降，但尚未取得足以替换基线的综合音质收益。
[因果上下文续轮](docs/causal-context-20261011.md)已实现过去八帧输入，保持输出延迟，
并完成两组各八轮的匹配训练；权重继续处于实验状态。
[V7 连续重建实验](docs/multires-reconstruction-20261011.md)加入真实 DSP 预热和多分辨率
波形监督，两组各四轮已完成；实际输出仍有改善与退步，不替换现有基线。
[输入条件化增益校准](docs/gain-calibration-20261011.md)进一步用实际分析器训练10个DSP系数；
回归误差下降，但实际高频恢复退步，默认增益策略保持不变。
[相位重建对照](docs/phase-reconstruction-20261011.md)保持模型幅度，按输入残留功率融合相位；
波形保真和部分高频指标改善，但音乐仍未全面达标。新DSP策略仅显式启用，不替换正式权重。
[结构重建调研](docs/architecture-reset-20261011.md)进一步比较复数频谱生成、因果时间主干、
可微 DSP 与生成式训练，提出从头训练新主干的对照顺序及质量/实时门槛。
[第一阶段实现](docs/causal-prototypes-20261011.md)已完成原创 GRU/TCN、显式状态部署契约、
生产 DSP oracle 诊断和六次小样本训练；全库同预算对照与发布资格仍待验证。

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
