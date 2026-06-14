# 基础设置、上下文与 Caption 消融

生成时间：2026-06-14

本文档整理早期基础实验：baseline、no-CoT、rounds、ensemble、n-shot、caption 类型、context mode、answer extraction、Qwen caption 等。

## 对应代码阶段说明

本文件混合了两个相近但不完全一致的代码阶段：

| 阶段 | 包含实验 | 当时代码主要特点 |
|---|---|---|
| 早期 forward baseline | `forward_baseline`、`forward_rounds*`、`forward_ocr`、`forward_qwen_caption` 等 | 原始 forward 多轮 CoT 流程，prompt 中容易包含 round state、previous thoughts、caption、objects 等较长上下文 |
| forward2 clean/context | `clean_rounds1`、`no_cot_rounds*`、`context_*`、`answer_*`、`qwen_caption_*` | 后续修正 shard/merge 和 answer extraction，并加入 `context_mode`、`answer_extraction_strategy`、Qwen caption 注入控制等 args |

因此，本文件里的结果最适合说明趋势：旧式 CoT 和复杂上下文容易掉点，direct/no-CoT 和简化上下文更强。若要做最严格公平对比，应在当前最新代码上重跑关键配置。

## 核心对照

| 实验 | 准确率 | 正确数 | 主要设置/操作 |
|---|---:|---:|---|
| `forward_baseline` | 52.72% | 603/1145 | 早期 5-round CoT baseline |
| `clean_rounds1` | 55.33% | 633/1145 | 干净 1-round CoT |
| `no_cot_rounds1` | 59.24% | 678/1145 | 去掉 CoT，1 round，当前最高 |
| `no_cot_rounds3` | 58.18% | 666/1145 | no-CoT，多 round |
| `direct_image_question_only` | 58.54% | 670/1145 | 只给图像+问题，无基础 caption/context |

结论：direct no-CoT 是最强；单纯“只给图像+问题”的 direct 略低，说明基础 VinVL caption/context 对 direct 有帮助。

## Context Mode 消融

| 实验 | 准确率 | 正确数 | context 设置 | 解释 |
|---|---:|---:|---|---|
| `context_empty` | 55.90% | 640/1145 | `--context_mode empty` | 不给 brief context，CoT 反而提升 |
| `context_no_round_state` | 55.48% | 635/1145 | 去掉 round state | 去除多轮状态噪声有帮助 |
| `context_caption_only` | 55.17% | 631/1145 | 只保留 caption | 稍低于 empty |
| `context_objects_only` | 55.10% | 630/1145 | 只保留 selected objects | 不如 empty/no_round_state |
| `clean_rounds1` | 55.33% | 633/1145 | 常规 1-round CoT | 对照 |

操作差异：这批实验主要通过 `--context_mode` 控制最终 answer prompt 中 `Brief Context` 的内容。

主要结论：

- CoT 阶段里，更多上下文不一定更好。
- round state 容易累积错误答案假设、错误选择对象或错误增强判断。
- 对 CoT 来说，`empty` 反而最高，说明 CoT 本身很容易被文本上下文牵引。

## No-CoT / Rounds / Ensemble

| 实验 | 准确率 | 正确数 | 设置 |
|---|---:|---:|---|
| `no_cot_rounds1` | 59.24% | 678/1145 | no-CoT, rounds=1 |
| `no_cot_rounds3` | 58.18% | 666/1145 | no-CoT, rounds=3 |
| `no_cot_ensemble1` | 58.18% | 666/1145 | no-CoT, ensemble=1 |
| `no_cot_ensemble3` | 57.36% | 656/1145 | no-CoT, ensemble=3 |
| `forward_no_cot` | 58.04% | 664/1145 | 早期 no-CoT |

主要结论：

- 对当前模型，增加 rounds/ensemble 并没有稳定提升。
- 最强设置是简单的 no-CoT rounds=1。
- 这说明模型第一反应通常已经很强，重复询问或投票可能引入不一致答案。

## N-shot 与检索设置

| 实验 | 准确率 | 正确数 | 设置 |
|---|---:|---:|---|
| `forward_nshot0` | 52.52% | 601/1145 | 0-shot |
| `forward_nshot4` | 52.48% | 600/1145 | 4-shot |
| `forward_sim_question` | 52.45% | 600/1145 | question-only retrieval |
| `forward_baseline` | 52.72% | 603/1145 | 原始设置 |

主要结论：

- 在旧 CoT baseline 下，n-shot 与检索方式变化都没有带来明显收益。
- 这些差异只有 1-3 题，不能作为主贡献。

## Caption 类型与 Caption 移除

| 实验 | 准确率 | 正确数 | 设置 |
|---|---:|---:|---|
| `forward_remove_caption` | 52.85% | 605/1145 | 移除 caption |
| `forward_caption_vinvl_sg` | 52.69% | 603/1145 | scene-graph caption |
| `forward_caption_vinvl_tag` | 51.71% | 592/1145 | tag caption |
| `forward_baseline` | 52.72% | 603/1145 | VinVL caption |

主要结论：

- 在旧 CoT baseline 下，caption 类型影响有限。
- `vinvl_tag` 明显差一些，可能是标签列表缺少句子语义，也更容易让模型输出 object-list 风格答案。

## Qwen Caption 消融

| 实验 | 准确率 | 正确数 | 操作 |
|---|---:|---:|---|
| `qwen_caption_no_final` | 52.66% | 602/1145 | 调用 Qwen caption，但不注入最终 prompt |
| `qwen_caption_local` | 45.23% | 517/1145 | 注入 local Qwen caption |
| `qwen_caption_global` | 44.06% | 504/1145 | 注入 global Qwen caption |
| `qwen_caption_short` | 43.87% | 502/1145 | 注入短 Qwen caption |
| `forward_qwen_caption` | 44.46% | 509/1145 | 早期 Qwen caption |

主要结论：

- 问题不在“调用 Qwen caption”，而在“把 Qwen 生成 caption 注入最终回答 prompt”。
- 注入后大幅掉点，说明生成 caption 的错误细节或过度描述会强烈误导答案。

## Answer Extraction 消融

| 实验 | 准确率 | 正确数 | 策略 |
|---|---:|---:|---|
| `answer_strict_final` | 52.43% | 600/1145 | 严格抽取 `Final Answer:` |
| `answer_raw` | 52.37% | 599/1145 | 使用原始回答 |
| `answer_last_line` | 52.25% | 598/1145 | 取最后一行 |

主要结论：

- answer extraction 不是 CoT 掉点的主因。
- 真正的问题在 prompt/context/reasoning 阶段已经发生。

## 小模块

| 实验 | 准确率 | 正确数 | 模块 |
|---|---:|---:|---|
| `forward_qwen_thought` | 53.09% | 607/1145 | Qwen thought verifier |
| `forward_ocr` | 52.90% | 605/1145 | OCR context |
| `forward_clip_thought` | 52.86% | 605/1145 | CLIP thought verify |
| `forward_all_regions` | 52.82% | 604/1145 | all regional captions |
| `forward_all_added` | 43.90% | 502/1145 | 所有附加模块同时开启 |

主要结论：

- 单个小模块收益很弱。
- 所有模块一起开严重伤害结果，说明“更多证据”不等于更好，证据噪声会堆叠。
