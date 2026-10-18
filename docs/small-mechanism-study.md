# P28：先用小模型验证失活机制

2026-10-19，承接 [P27 定位与外部方法调研](head-collapse-audit.md)。
这轮固定宽度 96，不恢复 640 宽度大模型训练，也不以扩大参数量作为改善方案。
先检验解码器失活、预算监督梯度和相位输入三个机制，再决定是否实现新的 DSP 激励分支。

## 固定对照

| 组别 | 相对控制组的唯一干预 | 参数量 | 预检 ONNX |
|---|---|---:|---:|
| `control` | 小尺寸 complex residual head，原 GELU、共享预算梯度、幅度输入 | 431,015 | 1.688 MiB |
| `decoder_norm` | decoder Linear 与 GELU 之间增加逐帧 LayerNorm | 431,207 | 1.689 MiB |
| `budget_detach` | 只阻断预算辅助 loss 对共享 latent 的梯度 | 431,015 | 1.688 MiB |
| `phase_input` | 编码器增加观测复数载波的零初始化投影 | 455,783 | 1.784 MiB |

四组均使用完整残差预算 `budget_scope="residual"`、`residual_slope=1/64`，
相同 shared 参数初始化、零输出初始化和相同随机数据日程。phase 投影额外增加
24,768 个参数，因此其结果不能完全分离相位信息与新增参数化的影响；本轮没有规模排名。
这些是单因素对照，不检验三种修改的交互作用，也不把全部修改叠在一起后归因。

`budget_detach` 使用同一个预算头和数值相同的预算预测，只为辅助监督重新计算
`budget_head(latent.detach())`。部署输出、主恢复 loss 到预算头／共享特征的梯度
均保持原路径；不是关掉预算、绕过主损失或减小增强。

`phase_input` 将当前输入相位的 sin/cos 与归一化 log 幅度载波结合，经过无偏置
258→96 投影，加到输入编码器的 LayerNorm 之前。数字静音的特征底噪先扣除，
不给零信号注入任意相位。仅访问输入与过去状态，参考相位只用于既有训练监督。
原有 TCN 累积历史；这里没有新增前瞻、声道共享状态或额外帧缓冲。
LayerNorm 只沿特征维度计算，无跨时间统计或训练／推理 running statistics 差异。

## 训练与数据协议

- 两个固定种子 `20261012 / 20261013`，每组 **2,048 次更新**，共八条轨迹、16,384 次更新。
  这是小尺寸机制诊断预算，不是充分收敛或容量上限对照。不得与 P26 的 16,192 步结果
  直接排名架构，也不以中途结果缩减某组预算。
- 每个种子复用 P26 broad 日程的前 2,048 个 draw；每步 4 个不同音乐录音＋1 个语音，
  加对应 clean identity，paired/identity 等权。保持 codec、采样率、声道视图及源时间规则。
  数据仍来自已审核原生无损音乐与既定语音材料，不引入第三方权重或数据。
- 整个输入窗口与 development/shifted 源区间隔离，跨 codec、采样率和声道视图核查；
  保留录音、来源、work-group 划分审计。实际独立监督时长写入 `coverage.json`，
  不能用曲库数替代覆盖，也不把单声道 L/R/M/S 视图当成成对立体声评价。
  两个种子均涉及 253 个音乐、193 个语音录音及 9 种 codec 配方；按源时间去重后，
  音乐监督为 3565.78 / 3565.40 秒，语音为 531.73 / 527.10 秒。
- 统一 Adam `1e-4`、无 weight decay、梯度范数上限 5；既有部署重建主项＋
  `0.1365237430457469 × clean-complex consistency`＋`0.1 × budget risk`。
  本轮不改变 loss 数值定义、权重、频带归一化或 DSP，不引入 GAN。
- 相同 80-hop 历史预热，监督从第 89 帧开始，预热不反传。训练缓存为 128 裁剪 LRU，
  新随机裁剪不写磁盘；原生审计的少量缓存放在本轮目录，避免重建已清理的大缓存。
- 各组完整跑完，再冻结最终 checkpoint 哈希；没有 best checkpoint 选择或早停。
  数值非有限、来源漂移、运行错误必须中止并留下记录。

## 验证与判断

固定训练监测集只报告优化过程。最终机制诊断使用首个种子日程第
0、682、1364、2047 个 draw，共 16 个音乐＋4 个语音裁剪：
记录 GELU 前后激活、预算、分频带原始／应用后残差、恒等 DSP 差异和参考 oracle。
在第一批固定样本上分解五项损失的共享／decoder／budget 梯度，预算隔离组按真实
训练梯度路由检查。归一化组的激活前统计取实际 GELU 输入，不能误取 LayerNorm 前值。

八个最终端点统一复核：

1. 相同已复用的 development/shifted 裁剪，分别报告音乐、语音、原始预测与模型＋DSP。
   development 不是新的独立测试，不依据其结果宣称无过拟合。
2. 保留未处理压缩输入、恒等模型经过真实 DSP 的控制，以及原始无损参考。
   检查 HF 谱收敛、能量、瞬态、低频保真、identity 扰动、绝对 PCM RMS/dBFS；
   不靠弱频带 LSD 或更小的增强量判断恢复成功。
3. 原生预测回放 1,602 个角色，联合 PCM 复核含基线共 1,780 个角色；再进行
   **1,424 个完整 Rust 前端→ORT→DSP 角色回放**，保留前端差异与同观测合成差异。
4. 八个导出均执行 320-hop ONNX 状态／静音／reset 一致性验证；完整链路性能安排
   44.1/48 kHz、单／双声道、重复测量及 paced 长流，共 56 个进程、560,000 个测量 hop。
   单独报告 <5 ms 计算上限、48 kHz/hop128 的 2.667 ms deadline、CPU/RSS 和调度迟到。
   固定 256 帧缓冲继续计入完整输出延迟；合成性能刺激不是设备／听感认证。

恢复有效性与机制有效性分开判断。激活恢复、梯度非零或训练 loss 下降，只能说明
机制干预生效。只有跨种子实际高频恢复、保留频段和原音保真均有支持时，才继续同规模
完整训练或 DSP 结构实验。现有相对指标规则仅作筛查，仍保留全部绝对误差与反例；
不放宽既定规则、不挑种子、不自动扩大规模、不自动替换正式权重。

## 工程预检与运行

预检目录：`data/runs/small-mechanisms-p28-preflight-20261019/`。
四组各跑 4 步，仅用训练裁剪完成训练、机制诊断、冻结端点、共同评分、原生回放和导出，
再完成 16 个完整前端→ORT→DSP 角色回放。预检权重丢弃，不用于选择候选或证明音质。
它覆盖完整流水线，但不运行正式长时性能测量。
四个导出均通过 320-hop 状态／reset／静音一致性检查；16 个完整 Rust 角色回放的
同观测合成检查通过。完整本地测试 **569 项通过**，Ruff lint/format 通过。
修正了旧 resume 测试替身对新增关键字参数的兼容，并加入隔离组的精确续跑验证。

新增结构验证包括：共享初始化与旧 checkpoint 兼容、预算辅助梯度隔离且主梯度不变、
负公共偏移下的 decoder 梯度、相位输入的有效性／因果性、有限差分、静音／弱信号梯度、
分块／reset／声道隔离、非零模型 ONNX 和真实 Rust 一致性，以及注册端点的计时选择。

正式训练由 `training/scripts/run_small_mechanism_study.py` 驱动，从已提交的冻结源码运行。
运行路径、登记哈希和启动状态记录在本页后续状态及本地 `provenance.json` 中。
训练完成后的机制诊断、原生复核和时延测试由同一个流水线顺序执行。

```bash
rtk proxy env PYTHONPATH=training training/.venv/bin/python \
  training/scripts/run_small_mechanism_study.py \
  --parent data/runs/broad-budget-p26-20261017-v2/study \
  --output data/runs/small-mechanisms-new/study \
  --native-binary target/release/soundex-context-replay --prepare-only
```

执行完整流水线时，应先按登记保存 `source/training` 快照，再从该快照启动；
不要从日后修改的 main 续跑已有登记。`--preflight` 固定为独立 4 步训练材料预检，
必须使用另一个目录。原生图仍为实验接口，正式 artifact 和默认权重保持原状。

## 启动记录

正式流水线已于 **2026-10-19 03:29 +08** 启动，目录
`data/runs/small-mechanisms-p28-20261019/`。
冻结训练源码来自提交 `435024a86a4c80e53a19c2f26d6494ee3d0c274c`，登记 SHA-256 为
`d99d33421be0cb94ddcc861b22e95856f97feeb025dae467114b32c080caa8a0`。
启动时 supervisor PID 4967、流水线 PID 4969、caffeinate PID 4970；PID 仅供本次运行定位。
最新阶段看 `study/progress.json`，日志看 `pipeline.log`，退出状态看 `supervisor.json`。
训练与正式复核尚未完成，不从预检的 4 步分数推断哪一组更好。
