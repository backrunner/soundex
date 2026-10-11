# 有限因果上下文实验

本轮在小型频谱 refiner 上加入过去八帧，验证时间线索是否能改善缺失高频，
而非继续扩大逐帧静态映射。结果尚待完成，候选不替换默认权重。

## 结构与协议

- 父模型为上一轮 `shape-small` epoch4，905,734 参数；因果分支增加 17,792 参数，
  共 923,526 参数。保留原有 gain/圆周谱形表示与有界幅度、相位修正。
- 九帧共享频谱编码，depthwise 时间卷积与零初始化 mixing 合并当前 latent。
  原 U-Net 只处理当前帧；不用未来帧、循环隐状态或迭代生成。
- artifact/architecture 1.4：输入 `[B,2,9,129]`，输出 `[B,2,1,129]`，
  float32、opset17、动态 batch 1/2。输入按最旧到当前排序。
  graph 的 `stateless` 指无隐式递归状态，host 仍负责保存每个物理声道的频谱历史。
- Rust 每个观察到的 hop 都推进历史，包括静音与 bypass；按物理声道缓存后再组 batch，
  防止活跃 batch 位置变化造成串音。reset、worker 序号缺口、模型重载清空历史。
  冷历史为 -200 dB / 0 rad；前八 hop 输出对齐 dry，第九 hop 才允许增强并走已有淡入。
- 历史填充约 21.33/23.22 ms 是增强启动时间，不是额外播放延迟；输出仍遵守固定
  256 samples 的实时适配器时间轴。ORT 控制线程预热不推进音频历史。
- 缓存只用预分配空间；回调零分配、不阻塞，推理留在 worker。历史会增加内存和
  推理工作量，仍须测量 CPU、RSS、逐 hop 时延、miss 与恢复，不能仅凭参数估计。

## 公平训练与选择

`prepare_temporal_study.py` 从同一审核过的父检查点生成 frame-control 与 causal。
保持原生无损数据、划分、253/254 音乐及 1/254 语音抽样、V6 loss 不变；
每组八轮 / 1,024 updates，LR 2e-5、128-step warmup/cosine、clip1、冻结 BN、无 GAN。
两组都取八帧真实 warmup 加八帧监督，固定相同裁剪和验证 896 行。
控制组同样跳过前八帧监督；上一轮无 warmup 的总分不能直接相比较。

显式 `--initialize-temporal-refiner` 只引入四个新张量，mix 权重和 bias 为零，
不沿用旧 optimizer。共享参数和 RNG 保持一致；九帧 GEMM 与单帧计算可能存在
FP32 舍入差异，迁移检查用严格绝对容差，而非要求逐位相同。
Loss 仍使用现有短片段 DSP 代理；本轮只改变时间观察，尚未实现更长多分辨率
监督或可学习 DSP 增益，以免混淆结构收益。

各组按验证 total 选 epoch，高频幅度打破平局；用同一 CPU FP32、8+8 裁剪的
V6 total 在两组之间选候选，评分历史留出前冻结模型哈希。此前用过的测试录音
明确标作留出复核。总分、实际 Rust 音质、实时表现与听感是独立证据。

## 验证与证据

实现测试覆盖：非零学习后的过去帧影响、未来隔离、逐帧与窗口一致性、动态
batch、PyTorch/ORT/Rust 数值 parity、历史轴与声道切换、冷启动/reset/重载、
不规则分块、真实 worker 饱和恢复以及回调/固定 hop 的 Rust 零分配。
`temporal-probe.onnx` 是合成协议测试器，不是修复模型。

实验配置/检查点写入 `data/runs/causal-context-20261011/`；
质量、资源与选择报告写入 `data/reports/causal-context-20261011/`。
训练原始音频和实验产物留在忽略目录。结果和剩余验收将在本页更新。
