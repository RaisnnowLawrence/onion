# 代码版本演化与实验可比性说明

生成时间：2026-06-14

这份文档专门解释一个重要问题：当前 `onion_output` 里的实验很多，但它们并不是在完全一致的代码版本上跑出来的。实验是边改代码、边补消融、边修运行脚本和 merge 逻辑做出来的。因此，读结果时要同时看“实验设置”和“当时代码主要做了什么”。

## 总体原则

结论先说清楚：

- 可以比较同一阶段、同一方法族内部的实验，例如 `context_empty` vs `context_no_round_state`，或者 `reviewer_evidence_caption_only` vs `reviewer_evidence_all`。
- 跨阶段比较时要谨慎，例如早期 `forward_baseline` 和后期 `pure_cot_answer_first_locked` 不只是参数不同，代码里的 prompt、answer extraction、reviewer 逻辑也已经变了。
- 最终写论文或报告时，最好把这些实验称为“开发过程中的消融探索”，而不是一个完全统一代码版本下的 benchmark。
- 如果需要最严格公平比较，应该在当前最新代码上挑关键方法重跑一版。

## Git 版本脉络

当前版本控制仓库：

```text
/data2/lizhengxue/WorkSpace/onion
git@github.com:RaisnnowLawrence/visualcot_experiment_repo.git
```

近期主要提交：

| Commit | 大致阶段 | 主要内容 |
|---|---|---|
| `4248a3d` | 合并初始 forward 与早期总结 | 将 forward 代码和早期消融总结合并进统一仓库 |
| `a94cf80` | MCTS n=20 / ablation controls | 同步 MCTS 参数和消融控制 |
| `30086e6` | marker MCTS / CoT diagnostics | 同步 marker MCTS 与 CoT 诊断结果 |
| `d3ae983` | structured compact CoT | 加入结构化 compact CoT 风格 |
| `fb14076` | CoT rescue results | 保存 CoT rescue 消融结果 |
| `5b2cee0` | direct_verify | 新增 direct answer verification CoT style |
| `84bf7fb` | direct_verify followup | direct_verify 后续消融 |
| `86a8c91` | reviewer evidence | 新增 reviewer evidence CoT mode |
| `95fb67d` | reviewer evidence scope | reviewer evidence 范围消融 |
| `156e508` | selective reviewer | 新增 selective reviewer evidence mode |
| `fbecea0` | selective reviewer results | 保存 selective reviewer 实验结果 |
| `5ddcce2` | constrained pure CoT | 新增受约束 pure CoT 消融 |
| `f24759c` | pure CoT / direct results | 保存 pure CoT 与 direct 对照结果 |
| `a9a7a18` | reflective answer-first | 新增 reflective answer-first 与脚本 |

## 阶段 1：早期 forward baseline

代表实验：

- `qwen3-VL-4B_forward_baseline`
- `qwen3-VL-4B_forward_rounds1`
- `qwen3-VL-4B_forward_rounds3`
- `qwen3-VL-4B_forward_remove_caption`
- `qwen3-VL-4B_forward_nshot0`
- `qwen3-VL-4B_forward_nshot4`
- `qwen3-VL-4B_forward_ocr`
- `qwen3-VL-4B_forward_clip_thought`
- `qwen3-VL-4B_forward_qwen_caption`
- `qwen3-VL-4B_forward_qwen_thought`

代码主要逻辑：

- 使用原始 forward 流程。
- 保留传统多轮 Onion/CoT 思路。
- 每一轮根据模型/策略选择 image、caption、knowledge 等增强方向。
- prompt 中会包含 VinVL caption、round state、selected objects、previous thoughts 等信息。
- Qwen-VL 最终生成答案，CoT 回答通过当时的 answer extraction 逻辑抽取。

这批实验主要用于发现：

- 原始 CoT baseline 约 52.72%。
- rounds 增加并没有帮助。
- OCR、CLIP thought、Qwen thought 等小模块收益很弱。
- Qwen caption 注入最终 prompt 会严重伤害效果。

可比性风险：

- 早期 `rounds1` 有 1147 样本的重复 shard 问题。
- answer extraction 和 merge 逻辑后来做过修正。
- 这批结果适合当“早期系统 baseline”，不适合和后期每个新结构做逐题严格公平比较。

## 阶段 2：clean / no-CoT / context / answer extraction

代表实验：

- `qwen3-VL-4B_forward2_clean_rounds1`
- `qwen3-VL-4B_forward2_no_cot_rounds1`
- `qwen3-VL-4B_forward2_no_cot_rounds3`
- `qwen3-VL-4B_forward2_context_empty`
- `qwen3-VL-4B_forward2_context_caption_only`
- `qwen3-VL-4B_forward2_context_objects_only`
- `qwen3-VL-4B_forward2_context_no_round_state`
- `qwen3-VL-4B_forward2_answer_strict_final`
- `qwen3-VL-4B_forward2_answer_raw`
- `qwen3-VL-4B_forward2_answer_last_line`

代码主要逻辑：

- 更清晰地区分 `chain_of_thoughts` 开关。
- 增加 `context_mode`，控制最终 answer prompt 的 brief context：
  - `empty`
  - `caption_only`
  - `objects_only`
  - `no_round_state`
  - `full`
- 增加 `answer_extraction_strategy`，控制 CoT 输出如何抽答案：
  - 当前策略
  - strict final
  - last line
  - raw
- 对 shard/merge 的结果更谨慎，后续以 1145 样本为准。

这批实验主要用于发现：

- no-CoT rounds=1 达到 59.24%，成为当前最强。
- answer extraction 不是主要瓶颈。
- CoT 中 round state 和冗长 context 会带来噪声。

可比性说明：

- 这批比早期 baseline 更适合作为当前方法比较基准。
- 后续 direct_verify、reviewer、pure CoT 多数都建立在这个 forward2 代码基础上。

## 阶段 3：MCTS / SAM 图像增强参数化

代表实验：

- `mcts_image_no_cot_rounds1_n5`
- `mcts_narrow_no_cot_rounds1_n5/n10/n20`
- `mcts_marker_narrow_no_cot_rounds1_n1/n5/n10/n20`
- `mcts_marker_visual_no_cot_rounds1_n10`
- `mcts_safe_no_cot_rounds1_n5`
- `compact_marker_mcts`
- `conflict_marker_mcts`

代码主要逻辑：

- 将原本较固定的 MCTS/SAM 图像增强改成 args 可控：
  - `--use_image_enhance`
  - `--mcts_n_simulations`
  - `--mcts_trigger_mode`
  - `--mcts_action_mode`
  - `--mcts_filter_objects`
- MCTS 根据问题和候选对象寻找增强动作。
- 动作包括 crop、outline、marker、no_crop 等不同图像处理方式。
- 后续 narrow/safe 版本减少动作范围，避免过度改变图像。

这批实验主要用于发现：

- 原始 MCTS 改图像会明显掉点。
- 收窄动作集合后能接近 direct no-CoT，但仍没稳定超过 direct。
- 增加 n_simulations 有轻微变化，但不是决定性因素。

可比性风险：

- MCTS 实验由于搜索慢，很多是分片并行、不同 shard 数完成。
- 不同 MCTS 实验的动作集合和触发条件不同，不能只看 `n_simulations` 一个参数。
- 和 no-CoT 比较时要说明它改变了输入图像或图像提示。

## 阶段 4：Compact / Conflict CoT Rescue

代表实验：

- `cot_rescue_compact_rounds1/3/5`
- `cot_rescue_compact_marker_mcts`
- `cot_rescue_compact_outline_mcts`
- `cot_rescue_conflict_no_round_state`
- `cot_rescue_conflict_caption_only`
- `cot_rescue_conflict_objects_only`
- `cot_rescue_conflict_ocr_context`
- `cot_rescue_conflict_marker_mcts`

代码主要逻辑：

- 传统 step-by-step CoT 改成更短、更结构化的 CoT。
- compact 风格要求模型只列少量 visual cues，再给 final answer。
- conflict 风格更保守，主要判断当前答案是否被证据明确推翻。
- 这阶段开始形成“不要让 CoT 自由长推理”的经验。

这批实验主要用于发现：

- compact 比旧 step-by-step 好，但仍不如 direct。
- conflict-only 更稳，尤其 `conflict_no_round_state` 接近 direct best。
- 多轮 compact 仍然可能累积噪声。

可比性说明：

- 这批已经不是原始 CoT，而是被约束过的 CoT。
- 它适合证明“约束推理链比长 CoT 更好”，不适合和旧 CoT 直接混为同一种方法。

## 阶段 5：Direct-Verify

代表实验：

- `cot_rescue_direct_verify_rounds1`
- `cot_rescue_direct_verify_conflict_only`
- `cot_rescue_direct_verify_no_fallback`
- `cot_rescue_direct_verify_keep_stronger`
- `cot_rescue_direct_verify_qwen_caption`
- `cot_rescue_direct_verify_marker_mcts`
- `cot_rescue_direct_verify_outline_mcts`

代码主要逻辑：

- 新增 `--cot_style direct_verify`。
- 每个样本通常至少两次 Qwen 调用：
  1. 先直接短答。
  2. 再用 verifier prompt 判断证据是否支持或矛盾。
- 新增 `--direct_verify_policy`：
  - `balanced`
  - `keep_stronger`
  - `conflict_only`
  - `revise_freely`
  - `no_fallback`
- 加入 fallback 逻辑：如果 verifier 输出像 object list / visual cue list，可以回退到初始答案。

这批实验主要用于发现：

- `conflict_only` 最好，说明 verifier 最大问题是过度修改。
- Qwen caption evidence 会明显伤害。
- direct_verify 是有希望的方向，但整体还没有超过 direct no-CoT。

可比性风险：

- direct_verify 调用模型次数更多，计算成本不同。
- 它不是“普通 CoT”，而是两阶段答案审查。
- 与 direct no-CoT 比较时要明确它增加了 verifier 调用。

## 阶段 6：Reviewer Evidence

代表实验：

- `reviewer_evidence_all`
- `reviewer_evidence_caption_only`
- `reviewer_evidence_caption_object`
- `reviewer_evidence_no_objects`
- `reviewer_evidence_selective`
- `reviewer_evidence_caption_only_strict`

代码主要逻辑：

- 新增 `--cot_style reviewer_evidence`。
- 初始答案仍然先由模型直接给出。
- 三增强模块或已有上下文不直接生成最终答案，而是整理成 evidence：
  - base caption
  - selected objects
  - regional captions
  - OCR
  - enhance caption
  - enhance knowledge
  - enhanced image
  - Qwen caption
- reviewer prompt 输出：
  - `Evidence Check: supported / contradicted / uncertain`
  - evidence points
  - final answer
- 新增 `--reviewer_evidence_scope` 控制 reviewer 能看到哪些证据。

这批实验主要用于发现：

- caption-only reviewer 最强，说明干净短证据比多证据更好。
- all/selective/object evidence 没有显著提升，说明证据噪声仍然存在。
- 这条线最能自然结合“三增强模块负责找证据，reviewer 负责判证据”。

可比性说明：

- reviewer evidence 是方法结构变化，不只是参数变化。
- 它适合和 direct_verify 以及 answer-first 比较“答案审查是否有效”。
- 不应和早期 image/caption/knowledge 三增强直接生成式用法混为一类。

## 阶段 7：Selective Reviewer

代表实验：

- `reviewer_evidence_selective_rounds1_3shards_gpu456`
- `reviewer_evidence_selective_rounds1_3shards_gpu7`

代码主要逻辑：

- 根据问题类型选择 evidence 种类。
- 例如涉及 OCR、颜色、数量、对象局部细节时，只打开更相关证据。
- 目标是减少 reviewer 看到无关证据的概率。

这批实验主要用于发现：

- selective 目前没有明显超过 caption-only。
- 说明简单的问题类型规则还不够强，或者证据本身质量仍是瓶颈。

可比性说明：

- selective 是 reviewer evidence 的子版本，应主要和 reviewer evidence scope 系列比较。

## 阶段 8：Pure CoT / Answer-First

代表实验：

- `pure_cot_answer_first_locked`
- `pure_cot_visual_facts`
- `pure_cot_visual_facts_no_caption`
- `direct_image_question_only`

代码主要逻辑：

- 新增 `--cot_style answer_first_locked`：
  - 第一行必须输出 `Answer: <short answer>`。
  - 后面可以给 reasons。
  - 最终评测只抽第一行答案。
- 新增 `--cot_style visual_facts`：
  - 先列最多 2 条 visible facts。
  - 再输出 answer。
- `direct_image_question_only` 通过 `context_mode empty` 且不开 CoT，作为裸图像+问题 direct 对照。

这批实验主要用于发现：

- `answer_first_locked` 只比 direct best 少 1 题。
- 只要先锁答案，CoT 不一定会掉点。
- 先列 visual facts 会严重掉点，说明先推理后答案仍然危险。

可比性说明：

- `answer_first_locked` 的高分依赖“只抽第一行答案”的代码逻辑。
- 它是纯 CoT prompt 结构消融，不是 reviewer，也没有三增强模块。

## 阶段 9：Reflective Answer-First

代表实验：

- 当前正在运行的 `reflective_answer_first_caption_r3`
- 当前正在运行的 `reflective_answer_first_no_caption_r3`
- 当前正在运行的 `reflective_answer_first_caption_r5`
- 当前正在运行的 `answer_first_locked_no_caption`

代码主要逻辑：

- 新增 `--cot_style reflective_answer_first`。
- 新增 `--reflect_rounds`。
- 结构是：

```text
Round 1：先短答
Round 2：补最小必要视觉证据，不允许改答案
Round 3：保守复核，只在 contradicted 时修改
Round 4/5：重复证据和复核
```

- 复核阶段复用 conflict-only 思路，避免 uncertain 时乱改答案。

这批实验要验证：

- 有无基础 caption 对 answer-first reflective 是否关键。
- r5 是否比 r3 更好，还是继续累积噪声。
- `answer_first_locked_no_caption` 能否说明 answer-first 的收益不依赖 VinVL caption。

可比性说明：

- 这是当前最新代码结构。
- 它比 `answer_first_locked` 多模型调用，多了证据和复核步骤。
- 最适合直接对比 `answer_first_locked`、`direct no-CoT`、`reviewer_evidence_caption_only`。

## 建议如何在论文/报告中引用这些实验

建议分成三层：

### 第一层：严格主结果

选当前最新代码，重跑关键配置：

- direct no-CoT
- answer-first locked
- reviewer evidence caption-only
- 最佳 MCTS narrow
- reflective answer-first r3/r5

这些可以作为严格表格。

### 第二层：开发消融

当前已有大量结果可以作为开发过程证据：

- 传统 CoT 掉点。
- Qwen caption 注入有害。
- answer extraction 不是主因。
- MCTS 改图像不稳。
- reviewer 过度修改有害。
- answer-first 是最有希望的纯 CoT。

### 第三层：机制分析

用于解释为什么：

- CoT 先行会污染答案。
- 多证据会引入噪声。
- 三增强模块更适合作为 evidence provider。
- reviewer 应该保守，只在明确矛盾时推翻初始答案。

## 最后结论

当前实验最合理的解释不是“所有方法在同一代码下 direct 永远最好”，而是：

```text
随着代码逐步改进，我们发现传统生成式 CoT 不适合当前 A-OKVQA + Qwen3-VL-4B 设置；
把 CoT 改成 answer-first 或 reviewer 后，性能可以接近 direct；
三增强模块最好不要直接改变答案，而应该提供少量、干净、可审查的证据。
```
