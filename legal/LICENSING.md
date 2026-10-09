# SoundEx 许可说明 / Licensing

## 适用范围

| 内容 | 许可与范围 |
|------|------------|
| 源码、文档、demo、合成测试模型 | [Apache-2.0](../LICENSE) |
| 附完整模型卡的正式发布权重 | [Apache-2.0](../LICENSE)，适用范围见[权重许可说明](../MODEL_WEIGHTS_LICENSE.md) |
| 本地实验、外部 checkpoint | 按其来源和实际授权处理，不能默认视为正式 Apache 权重 |
| 第三方音频、依赖 | 各自上游条款；音频不随仓库分发，依赖见 [THIRD_PARTY.md](../THIRD_PARTY.md) |

## 在其他项目中使用

正式权重允许其他开源项目实际集成、随应用分发、修改和微调，也允许商用。
遵守 Apache-2.0 的许可副本、声明保留和修改说明要求即可；没有额外的学术用途限制。
不同许可证的项目组合分发时，仍需满足对应许可证的兼容性要求。

## 如何保证发布路线可用

MUSDB18-HQ 已退出下载、预处理和训练入口。训练不设数据集或许可白名单，也不强制逐条
填写授权字段；可直接导入音频目录，尽量覆盖不同曲风、乐器、瞬态、声场和录音条件。
来源、署名和许可可作为元数据保留，不影响训练启动。

正式发布权重时，维护者核对实际数据授权、初始化与恢复来源，确认能够支持应用集成、
分发和商用。已有 MUSDB 实验结果保留为未发布的研究记录，不能靠修改文档变成正式权重。
训练入口的宽松接入不改变上游权利，未确认或不兼容的授权需在发布前解决。

[训练数据说明](TRAINING_DATA.md)介绍导入与发布的区别；[数据来源调查](../docs/datasets.md)
列出候选曲库和质量特点；[发布清单](MODEL_RELEASE_CHECKLIST.md)说明维护者发布步骤。
自动校验只核对记录与文件的一致性，实际授权范围由维护者核验。

[本次权重授权核查](WEIGHT_LICENSE_REVIEW.md)记录已有实验的实际来源、哈希和结论。
目前尚无已完成授权核查的正式恢复权重。新的区域训练已经移除 Slakh，使用核查通过的
真实音乐和 VCTK，并重新初始化。Slakh 的底层 MIDI／渲染权利缺口仅适用于包含它的
历史实验，不是新训练的检查项；新权重仍需核对实际 artifact 和完整来源。

## English scope

Apache-2.0 covers SoundEx source materials and official weights explicitly released
with a completed Apache-2.0 model card. It permits application integration,
redistribution, adaptation and commercial use under its standard conditions.
No custom research-only tier applies to new official weight releases.

Unreleased experiments and third-party materials are outside this weight grant.
Training-data approval and release checks are maintainer publishing procedures,
not additional downstream license terms. A project notice cannot extend upstream
rights. This policy does not determine that every trained model is a copyright
derivative of its training audio.

The full English [Apache License](../LICENSE) controls the license grant, warranty
disclaimer and limitation of liability. This overview is explanatory.
