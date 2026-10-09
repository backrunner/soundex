# 权重授权核查（更新于 2026-10-09）

**目前没有可认定为已完成授权核查的正式恢复模型权重。** 源码与 demo 采用
Apache-2.0；这不自动授权本地实验权重。正式权重仍以 Apache-2.0 发布为目标，
支持其他开源项目集成、分发、修改和商用，不设置学术用途限制。

## 已有实验的实际来源

本次核查读取了三个实验的实际 checkpoint、配置和五份绑定 manifest，验证它们的
SHA-256 一致性，并记录 42 个 checkpoint/诊断 ONNX 文件的哈希。训练集、验证集和
测试集分别记录；配对样本数量不等于独立作品数量。

| 实验 | 核查结果 | 正式发布状态 |
| --- | --- | --- |
| `local-mps-20261007-125023` | manifest 为 MUSDB18-HQ | 未发布，排除出可商用正式权重路线 |
| `local-mps-256-128-20261007-220130` | manifest 为 MUSDB18-HQ；`parity-qualified.onnx` 仅表示数值核查 | 未发布，数值合格不代表授权合格 |
| `local-mps-diverse-lossless-smoke-20261008` | 音乐为 The Crypts! / Revolution Void；另含 Slakh 和 VCTK；并非已经完成的新大规模训练 | 未发布，Slakh 底层权利及音乐来源完整授权仍待确认 |
| 新的 300 首区域无损训练 | 已冻结核查通过的真实音乐与 VCTK；入口随机初始化，不传入 `--resume`，未配置旧模型作为教师；实时阶段见[当前训练](../docs/native-training-20261009.md) | MUSDB、Slakh 均不参与；最终权重仍待实际 artifact 核查 |

两个 MUSDB 实验的最佳 checkpoint SHA-256 分别为
`167174450fc6591f616a3a5d645623417d06c76d3651623ddf802d573ff52c20`、
`5134cd9b80c5e36cb674dcad1686b77dc1fc39dbac1595d58d230e33407d9d21`。
非 MUSDB smoke 的最佳 checkpoint 为
`df2704704f7c68b1da3218abd25b5fc8153a71d77d94cd6e7838cc9901b83639`。
本地完整文件清单位于 `data/reports/weight-license-audit-20261008.json`；数据和权重未入库。

[MUSDB 发布方](https://sigsep.github.io/datasets/musdb.html)限定数据集用于学术用途。
删除训练目录中的 MUSDB 或修改许可文档，都不能改变已经训练的 checkpoint 来源。

## Slakh：数据集许可与底层权利分开核查

[Slakh 作者的原始生成仓库](https://github.com/ethman/slakh-generation#license-and-attribution)
明确将数据集标为 CC BY 4.0，
[官方存档](https://zenodo.org/records/4599666)说明音频由 Lakh MIDI 和采样乐器合成。
但 [Lakh 的维护者](https://colinraffel.com/projects/lmd/)同时说明：MIDI 抓取自公开网络，
并非由他转录，内部版权信息不一致，无法逐首归属作者。这是一个具体的作曲及转录
权利缺口；不能仅凭数据集的 CC 标记确认所有第三方作品都已授权。

作者的[渲染说明](https://github.com/ethman/slakh-generation#step-2-installsetup-kontakt-andor-other-synthesis-vsts)
确认使用 Kontakt Komplete 12（2018）。还需保留该实际渲染版本的乐器/采样许可依据。
Native Instruments 的
[现行 EULA](https://www.native-instruments.com/pages/end-user-license-agreement)
日期为 2026-07-01，包含对产品及其音视频内容的 AI 训练限制。该条款本身不能证明
它追溯适用于历史 Slakh 渲染，也不能直接认定完整合成音乐等同于乐器原始采样。
需要的是 Slakh 所用版本与已发布混音的实际许可范围，而非推测适用关系。

按 2026-10-09 的用户决定，新区域训练配置及队列已移除 Slakh，不再要求其配对文件。
历史 manifest、hash 和初始化来源记录保留为审计证据；旧实验已与当前任务分开归档。
本次工作区清理删除了不再使用的 Slakh 原始/配对音频、旧生成音频及重复周期检查点，
保留独立的历史核查检查点和删除/迁移清单。移除数据不会追溯净化旧权重。
通用历史导入工具仍保留，没有新增导入白名单或训练许可硬拦截。
含 Slakh 的具体权重在上述缺口解决前，不能声明已获核准的 Apache 商用发布。
可补充发布方对底层作品/渲染权利的说明，或逐首建立有授权依据的 Slakh 子集。
标题识别、MIDI 哈希和公共领域作曲者名字本身，都不足以证明该转录版本已授权。

## 已确认的许可路线

- **原创、自有或第一方明确授权的录音**：同时核对作曲、演奏/录音和可识别采样来源，
  保存实际版本、来源页与音频哈希。
- **CC0 / CC BY 的录音**：可用于取得其授权范围内的商业训练；CC BY 保留作者、
  来源、许可及必要的修改说明。[CC BY 4.0 原文](https://creativecommons.org/licenses/by/4.0/legalcode.en)
  的授权限于许可方拥有的权利，不能补上他人的版权。
- **VCTK 0.92**：[爱丁堡大学原始记录](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)
  明确给出 CC BY 4.0，作者为 Junichi Yamagishi、Christophe Veaux、Kirsten MacDonald。
  数据由 96 kHz / 24-bit 录音降采样至 48 kHz / 16-bit，适用于机器学习；报纸文本有
  Herald & Times Group 的许可。它不是“公共领域语音”。保留语音录音的署名及来源；
  本项目处理音频，不发布阅读文本或声纹身份模型。

[Creative Commons 的 AI 指引](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
支持按实际 CC 条件处理训练及后续公开使用，并建议保留可追溯署名。指引不是关于
所有模型权重版权属性的统一裁决。CC 数据不会自动把新权重指定为 CC 或 Apache；
应核对具体权重是否保留了上游受保护内容及适用条件。

## 正式权重发布时的结论

Apache-2.0 发布路线可行，但需要完成实际数据组合、初始化/父模型来源及最终 artifact
的核查。本次事实清单标为 `rights_confirmed: false`，不会代替维护者填写已经批准的
权利声明。发布时附完整 LICENSE、适用 NOTICE、源数据署名、模型卡和精确哈希，
同时检查模型是否在无关输入或静音输入时重现训练作品。

这些是维护者的发布工作，不是给下游 Apache-2.0 用户追加许可限制。
参见[权重许可](../MODEL_WEIGHTS_LICENSE.md)和[发布清单](MODEL_RELEASE_CHECKLIST.md)。
