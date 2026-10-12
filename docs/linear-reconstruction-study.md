# P8：线性重建目标与独立分组检查

承接 [P7 loss 审计](energy-objective-audit.md)。本轮固定 P7 加法能量双分支模型，
仅比较训练目标；不把模型扩容、loss 修改和新增推理特征混在一个实验里。
以下规则在正式训练前固定，默认权重不变。

## 重建目标

对每个样本、每个频带，先排除前 80 hop。设 `f` 为参考全谱 RMS 的 1e-4，
绝对下限 1e-7 DFT 幅度，与 P6/P7 相同；`D² = mean(target²) + f²`。
均值覆盖该频带的频点和有效时间帧。定义平滑范数：

`N(error) = sqrt(mean(error²) / D² + 1e-6) - 1e-3`。

- 幅度重建项：`N(prediction - target)`，共同约束能量和带内细节。
- 参考动态项：`N(diff(prediction) - diff(target))`，保留差分符号，分母仍为参考幅度能量。
- 总分：幅度重建 + 0.25×参考动态。0–4 kHz 权重 0.25，其余四带各 1，除以 4.25。

这是以直接重建目标替代旧分项总和，**不把旧对数总分再加回去**。
旧能量、归一化形状、动态、log1p 幅度仍记录，旧总分另存 `factor_total`，便于解释变化。
参考正确时新目标为零；平滑项避免零残差时 sqrt 梯度不定；地板避免极小参考无限放大。
所有参考统计只用于训练与评分，不进入推理。

已核验审计中的固定反例：正确谱形但仅 1% 幅度，新 loss **0.23271**；正确 RMS
但平坦谱形为 **0.30610**，不再奖励这次细节损伤。幅度 0.1 倍与 10 倍分别为
**0.21153 / 2.11741**，不再近似等价。此结果是数学排序验证，不是音乐质量结论。

## 对照与数据

| 组别 | 模型 | 训练目标 | 共同选模目标 |
| --- | --- | --- | --- |
| energy_control | P7 rms_residual | 原 P6/P7 分项 loss | P8 线性重建总分 |
| linear_reconstruction | 完全相同模型 | P8 线性重建总分 | P8 线性重建总分 |

两种子 `20261012/20261013`，四次从头训练，每次 2048 更新、batch 4。
同种子所有初始化张量完全相同，均为 807,398 参数，未加载 P7 权重。
AdamW lr=1e-4，余弦衰减至 0.1 倍、weight decay=1e-4、clip=5。
保持 past-only 1024/2048 点输入、80 hop 预热、现有音频和作品隔离。

每种子用 `seed+60000` 新抽样 8192 个，1024 个为恒等控制；音乐/语音采样及恒等替换
方式保持不变。每 512 步验证，共同选模仍使用音乐/语音 253:1 和 paired/identity 87.5:12.5。
第 0 步允许被选中，但不等于恢复成功。全部最佳若位于末步，也不据此宣称已收敛。
新旧总 loss 数值不能跨目标直接比较；正式对照用相同新目标重算两组。

## 分组检查不能被总体均值覆盖

共同总分只负责选出研究检查点。选模冻结后，另外执行全部分组检查；不会根据检查结果
反复改选模型，也不会用其他组的改善抵消某一组的退步。

弱频带定义：参考 clip-band RMS 不大于上述既有 `f`，只统计 4 kHz 以上频带。
对照与候选必须具有相同参考分组；没有弱频带的片段不以零值稀释弱组均值。
先对每个录音的片段取均值，再进行 2000 次配对 bootstrap。

- 开发音乐高频 SC、能量误差：相对同预算对照和原压缩输入均要求改善。
- paired 和 identity 的音乐弱频带重建误差：各自单列，不允许退步。
- 音乐及语音的恒等重建误差：各自单列，不允许退步。
- 音乐及语音的保留低带 SC、高频 flux，以及 paired 语音重建误差：分别检查。

“支持改善”要求差值 95% 区间上界 <= -1e-6；“支持未退步”要求上界 <= +1e-6。
1e-6 是数值容差，不是可闻容差。少于两个录音标记 unsupported，区间不足以支持结论时
标记 inconclusive；所有检查均 supported 才标记 `all_groups_supported=true`。
因此原样输出、样本不足或总体均值改善，都不能自动通过。

这是一项保守的研究诊断，并非统计等效证明或生产晋级。复用开发录音不能证明独立泛化，
低功率频带的相对误差也不能代替听感。未通过时完整报告各项，不删除失败素材或改变阈值。

## 流程与验收边界

loss 测试覆盖参考最优、弱/静音有限梯度、有限差分、batch 独立性、warmup 排除、
真实参考动态、MPS/CPU 对应及前述排序反例。分组回归验证平均收益不能掩盖弱组损伤，
且配对、参考分组和缺失样本必须正确处理。模型没有结构改变，仍验证单/双声道 ONNX 状态对应。

正式训练完成后自动运行录音配对比较、全部分组检查、第一种子候选 ONNX 导出、逐 hop
CPU 成本与独立 RSS；benchmark 不与本轮训练并行。随后执行 890 次真实 Rust DSP 回放，
包含对应种子的 P3、恒等 DSP、两组模型配 P3 相位、候选配输入相位。
保留动态 HF、固定 HF、同窗五频带原始/重分析输出指标，代理误差要求 <=2e-5。

上述分组检查只覆盖原始幅度；最终仍要求实际 DSP 输出收益、恒等成品回放、独立评测、
盲听及完整实时稳定性，不能因原始幅度检查通过而自动替换默认模型。
模型继续受 <=2M 参数、<8 MiB 限制，前端+推理耗时与完整输出延迟分开报告。

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_linear_study.py \
  --parent data/runs/energy-p7-20261012/study/config.json \
  --output data/runs/linear-p8-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache

PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_envelope_study.py \
  --study data/runs/linear-p8-20261012/study --benchmark

PYTHONPATH=training training/.venv/bin/python training/scripts/replay_factorized_study.py \
  --study data/runs/linear-p8-20261012/study \
  --phase-study data/runs/causal-p3-20261012/study \
  --replay-binary target/debug/soundex-prediction-replay
```

从提交后的冻结源码执行；配置、抽样、检查点和流水线阶段保留本地。
源音频和实验权重不推送。此文登记规则，正式结果尚待完成后核验。
