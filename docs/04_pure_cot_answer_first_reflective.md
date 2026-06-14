# 纯 CoT / Answer-First / Reflective Answer-First 消融

生成时间：2026-06-14

本文档整理“不使用三增强模块、不把 CoT 只当 reviewer”的纯 CoT 方向，重点是 answer-first 系列。

## 对应代码阶段说明

本文件覆盖两个最新代码阶段：

| 阶段 | 包含实验 | 当时代码主要特点 |
|---|---|---|
| constrained pure CoT | `answer_first_locked`、`visual_facts`、`direct_image_question_only` | 新增受约束的 `cot_style`，尤其 `answer_first_locked` 会先输出答案，并且评测只抽第一行 |
| reflective answer-first | 当前 running 的 `reflective_*` | 新增 `cot_style=reflective_answer_first` 和 `reflect_rounds`，流程为先答、补证据、保守复核 |

这说明 `answer_first_locked` 的高分不是普通 step-by-step CoT，而是特定代码逻辑下的 answer-first 结构。它应主要和 direct/no-CoT、reflective answer-first 比较。

## 背景问题

传统 CoT 在当前任务上明显弱于 direct：

| 实验 | 准确率 | 正确数 |
|---|---:|---:|
| `no_cot_rounds1` | 59.24% | 678/1145 |
| `clean_rounds1` | 55.33% | 633/1145 |
| `forward_baseline` | 52.72% | 603/1145 |

原因判断：

- step-by-step 先产生中间文字，容易引入错误视觉事实。
- A-OKVQA 需要常识，长 CoT 会放大语言先验。
- 后续推理链可能污染最终答案抽取。

## 已完成纯 CoT 实验

| 实验 | 准确率 | 正确数 | 设置 |
|---|---:|---:|---|
| `pure_cot_answer_first_locked_rounds1_3shards_gpu4` | 59.17% | 677/1145 | 先输出 Answer，再给短 reasons；最终只取第一行 |
| `pure_cot_visual_facts_rounds1_3shards_gpu5` | 53.20% | 609/1145 | 先列 visible facts，再回答 |
| `pure_cot_visual_facts_no_caption_rounds1_3shards_gpu6` | 53.31% | 610/1145 | 无基础 caption 的 visual facts |
| `direct_image_question_only_rounds1_3shards_gpu7` | 58.54% | 670/1145 | 只看图像+问题，直接短答 |

## Answer-First Locked 做了什么

Prompt 结构：

```text
Answer: <short answer>
Reasons:
1. <visible reason>
2. <visible reason>
```

关键实现：

- 让模型第一行先给最终答案。
- 后续 reasons 不允许修改答案。
- 评测时只抽取第一行 `Answer:`，不让 reasons 污染最终答案。

为什么它强：

- 保留了 direct answer 的第一反应能力。
- reasons 只作为解释，不参与答案改写。
- 避免 object-list、visual cue list、长 reasoning 被当成答案。

结果上它达到 `59.17%`，只比 direct best 少 1 题，是目前最强纯 CoT。

## Visual Facts 为什么失败

Visual facts 结构：

```text
Visible Facts:
1. ...
2. ...
Answer:
```

结果只有约 `53.2%`。原因判断：

- 模型先列事实时容易列错，后续答案被错误事实牵引。
- 事实列表常常不是回答问题所需的最小证据。
- 对 A-OKVQA 来说，显式视觉事实不一定覆盖常识推断。
- 这个结构更像传统 CoT，仍然是“先推理后答案”，所以掉点。

## 当前正在跑的 Reflective Answer-First

最新代码版本：

```text
Git commit: a9a7a18 add reflective answer first ablations
```

新增参数：

```bash
--cot_style reflective_answer_first
--reflect_rounds 3 或 5
--direct_verify_policy conflict_only
```

结构：

```text
Round 1：先直接短答
Round 2：补最多 2 条最小视觉证据，不允许改答案
Round 3：保守复核，只有 contradicted 才允许修改
Round 4/5：可重复一轮证据和复核
```

已完成结果：

| 实验 | 准确率 | 正确数 | 目的 |
|---|---:|---:|---|
| `reflective_answer_first_caption_r3_rounds1_3shards_gpu4` | 59.60% | 682/1145 | 有基础 caption，3 阶段 |
| `reflective_answer_first_no_caption_r3_rounds1_3shards_gpu5` | 59.22% | 678/1145 | 无基础 caption，3 阶段 |
| `answer_first_locked_no_caption_rounds1_3shards_gpu6` | 59.51% | 681/1145 | 最强纯 CoT 的无 caption 对照 |
| `reflective_answer_first_caption_r5_rounds1_3shards_gpu7` | 58.68% | 671/1145 | 有基础 caption，5 阶段 |

这批实验已经完成并 merge。最重要的结果是：`reflective_answer_first_caption_r3` 首次超过 direct no-CoT best；但 `r5` 明显掉点，说明继续增加反思轮数会引入噪声。

## 这批 reflective 实验要回答的问题

### 1. CoT 能不能只作为“解释和保守复核”，而不是答案生成？

实际结果中，caption r3 超过 `answer_first_locked`，说明“补证据 + 保守复核”有实际收益；但 r5 掉点，说明这个收益只存在于短链条里。

### 2. 基础 caption 是否关键？

比较：

```text
reflective_answer_first_caption_r3
vs
reflective_answer_first_no_caption_r3
```

以及：

```text
answer_first_locked
vs
answer_first_locked_no_caption
```

无 caption reflective r3 为 59.22%，低于 caption r3 的 59.60%；answer-first locked no-caption 为 59.51%。这说明基础 caption 对 reflective 复核有帮助，但 answer-first 本身不依赖 caption 也很强。

### 3. 多轮是否有用？

比较：

```text
reflective_answer_first_caption_r3
vs
reflective_answer_first_caption_r5
```

实际 r5 为 58.68%，明显低于 r3 的 59.60%。这说明 CoT 多轮仍然会累积噪声，下一步应该选择性触发复核，而不是继续增加轮数。

## 当前判断

纯 CoT 不是无药可救，但传统 step-by-step 基本不适合这个工程。最有希望的路线是：

```text
先锁答案 -> 再解释 -> 解释不能自由改答案
```

也就是说，CoT 的价值应当体现在可解释性和少量纠错，而不是替代 direct answer 做主生成链。

## 下一步代码方向

基于 r3/r5 的对比，下一步不应该继续加轮数，而应该让复核变得更少、更硬、更可控：

- adaptive：只有高风险或低置信问题进入复核。
- keep/revise：复核只能输出 keep/revise，不允许自由生成新答案。
- visible-only review：复核只检查图像中是否有直接矛盾，不做常识扩展。
- answer-context split：Round 1 使用基础 caption，复核阶段不再看 caption。
- answer-first ensemble：多个第一答案投票后，再做一次保守复核。

## 当前 follow-up 5 实验

代码版本：

```text
d6d35bb add adaptive reflective answer first ablations
```

本轮在 `reflective_answer_first_caption_r3 = 59.60%` 的基础上做 5 个小改动，每个实验 3 shards，分别放在 GPU3-7：

| 实验 | 准确率 | 正确数 | 主要参数 | 结论 |
|---|---:|---:|---|---|
| `reflective_adaptive_highrisk_lowconf_caption_r3_3shards_gpu3` | 58.59% | 670/1145 | `--cot_style adaptive_reflective_answer_first --reflect_trigger_mode high_risk_or_low_confidence` | 选择性触发复核不够稳，掉点明显 |
| `reflective_keep_revise_caption_r3_3shards_gpu4` | 59.24% | 678/1145 | `--reflect_review_format keep_revise` | 强约束 reviewer 有保护作用，但低于原始 r3 |
| `reflective_visible_only_caption_r3_3shards_gpu5` | 59.21% | 677/1145 | `--reflect_evidence_mode visible_only` | 完全禁止常识/用途证据略伤效果 |
| `reflective_review_empty_context_caption_r3_3shards_gpu6` | 59.46% | 680/1145 | `--reflect_review_context empty` | 本轮最好，复核阶段去 caption 有收益 |
| `reflective_initial_ensemble3_keep_revise_caption_r3_3shards_gpu7` | 58.76% | 672/1145 | `--reflect_initial_ensemble 3 --reflect_review_format keep_revise` | 初答三投票没有帮助 |

这些实验仍围绕同一个核心原则：CoT 不做主生成链，只做最小必要审查。

本轮没有超过 `reflective_answer_first_caption_r3` 的 59.60%，但给出了一个更细的机制判断：caption 最好用于第一轮答案，不一定适合继续喂给复核阶段。下一步若继续做，优先围绕 `review_empty_context` 做微调，而不是 adaptive 跳过复核或 ensemble 初答。
