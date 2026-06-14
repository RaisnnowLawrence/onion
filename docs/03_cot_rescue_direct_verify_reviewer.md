# Direct-Verify / Reviewer Evidence 消融

生成时间：2026-06-14

本文档整理把 CoT 从“生成答案链”改成“答案审查链”的实验。

## 对应代码阶段说明

本文件覆盖三个逐步演化的代码阶段：

| 阶段 | 包含实验 | 当时代码主要特点 |
|---|---|---|
| compact / conflict CoT rescue | `compact_*`、`conflict_*` | 将长 step-by-step CoT 改成短 visual cues 或 conflict-only 复核，减少自由推理 |
| direct_verify | `direct_verify_*` | 新增 `cot_style=direct_verify`，先 direct answer，再 verifier 复核；新增 `direct_verify_policy` 与 fallback |
| reviewer evidence | `reviewer_evidence_*` | 新增 `cot_style=reviewer_evidence`，把 caption/image/knowledge/MCTS 等模块整理为 evidence 给 reviewer |

这三类实验都属于“救 CoT”，但代码结构不一样：compact 仍是答案生成式 CoT，direct_verify/reviewer 则是多次模型调用的答案审查链。

## 方法思想

传统 CoT：

```text
图像 + 问题 + 上下文 -> 先推理 -> 再生成答案
```

当前更有效的审查式 CoT：

```text
图像 + 问题 + 上下文 -> 初始短答案 -> 证据检查 -> 仅在明确矛盾时修改答案
```

这和三增强模块的关系更自然：

| 模块 | 在 reviewer 框架中的角色 |
|---|---|
| image / MCTS | 提供局部视觉证据 |
| caption enhance | 提供语义描述证据 |
| knowledge enhance | 提供常识证据 |
| reviewer CoT | 判断证据是否足以推翻初始答案 |

## Direct-Verify 结果

| 实验 | 准确率 | 正确数 | 设置 |
|---|---:|---:|---|
| `direct_verify_conflict_only_rounds1_4shards` | 58.96% | 675/1145 | 只有 Evidence Check=contradicted 才修改 |
| `direct_verify_no_fallback_rounds1_4shards` | 57.57% | 659/1145 | 禁用 fallback |
| `direct_verify_rounds1_4shards` | 57.34% | 656/1145 | 默认 direct_verify |
| `direct_verify_marker_mcts_n10_4shards` | 57.28% | 655/1145 | direct_verify + marker MCTS |
| `direct_verify_outline_mcts_n10_4shards` | 56.97% | 652/1145 | direct_verify + outline MCTS |
| `direct_verify_keep_stronger_rounds1_4shards` | 56.51% | 646/1145 | 更强 keep 初始答案 |
| `direct_verify_keep_stronger_marker_mcts_n10` | 56.38% | 645/1145 | keep_stronger + marker MCTS |
| `direct_verify_qwen_caption_rounds1_4shards` | 49.48% | 566/1145 | Qwen caption evidence |

结论：

- `conflict_only` 最好，说明 reviewer 最大问题是“过度修改”。
- Qwen caption evidence 明显伤害，再次证明生成 caption 不能直接当强证据。
- MCTS evidence 在 direct_verify 中没有稳定收益，可能是证据表达和 reviewer 判断不匹配。

## Conflict / Compact CoT Rescue

| 实验 | 准确率 | 正确数 | 说明 |
|---|---:|---:|---|
| `conflict_no_round_state_rounds1_2shards` | 59.10% | 676/1145 | conflict-only + 去掉 round state |
| `conflict_caption_only_rounds1_2shards` | 58.93% | 674/1145 | 只用 caption context |
| `conflict_marker_mcts_rounds1_n5_3shards` | 58.90% | 674/1145 | conflict + marker MCTS n=5 |
| `conflict_marker_mcts_rounds1_n20_3shards` | 58.79% | 673/1145 | conflict + marker MCTS n=20 |
| `conflict_marker_mcts_rounds1_n10_3shards` | 58.72% | 672/1145 | conflict + marker MCTS n=10 |
| `conflict_ocr_context_rounds1_3shards` | 58.72% | 672/1145 | conflict + OCR |
| `conflict_outline_mcts_rounds1_n10_3shards` | 58.66% | 671/1145 | conflict + outline MCTS |
| `compact_rounds3_4shards` | 58.10% | 665/1145 | compact CoT, rounds=3 |
| `compact_rounds1_2shards` | 57.70% | 660/1145 | compact CoT, rounds=1 |
| `compact_rounds5_4shards` | 57.38% | 657/1145 | compact CoT, rounds=5 |

结论：

- conflict-only 系列明显好于 compact 系列。
- round state 是噪声源，`no_round_state` 是 conflict 系列里最强。
- 增加 rounds 不一定更好；compact rounds=5 比 rounds=3 低。

## Reviewer Evidence 结果

| 实验 | 准确率 | 正确数 | evidence scope |
|---|---:|---:|---|
| `reviewer_evidence_caption_only_rounds1_3shards_gpu4` | 59.13% | 676/1145 | 只给 caption evidence |
| `reviewer_evidence_caption_only_strict_rounds1_2shards_gpu7` | 58.51% | 669/1145 | caption-only + 更严格 |
| `reviewer_evidence_caption_object_rounds1_3shards_gpu6` | 58.59% | 670/1145 | caption + objects |
| `reviewer_evidence_no_objects_rounds1_3shards_gpu5` | 58.45% | 669/1145 | 去掉 objects |
| `reviewer_evidence_selective_rounds1_3shards_gpu456` | 58.49% | 669/1145 | 问题类型选择证据 |
| `reviewer_evidence_all_rounds1_2shards` | 58.54% | 670/1145 | 所有 evidence |

结论：

- 最强是 caption-only reviewer，说明基础 caption 是相对干净、稳定的证据。
- 加 object、selective、all evidence 没有提升，可能因为 evidence 越多噪声越多。
- reviewer evidence 的正确方向是“少而准”，不是“证据越多越好”。

## 操作与参数

核心参数：

| 参数 | 含义 |
|---|---|
| `--cot_style direct_verify` | 初始短答后做 verifier |
| `--cot_style reviewer_evidence` | 三增强模块/上下文作为 evidence，reviewer 复核 |
| `--direct_verify_policy conflict_only` | 只在明确矛盾时改答案 |
| `--direct_verify_policy keep_stronger` | 更强保持初始答案 |
| `--disable_direct_verify_fallback` | 禁用对象列表 fallback |
| `--reviewer_evidence_scope` | 控制 reviewer 看哪些证据 |
| `--reviewer_disable_enhanced_image` | reviewer 不看增强图，只看原图 |

## 总结

这一组实验给出的最重要方法论是：

```text
三增强模块负责找证据，reviewer CoT 负责判断证据是否足以推翻初始答案。
```

它比传统 CoT 更符合当前结果，也更容易和 image/caption/knowledge 三模块结合。实验上，最值得保留的是：

- `conflict_only`
- `caption_only reviewer`
- `no_round_state`
- 少证据、强保守、不自由改答案
