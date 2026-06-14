# Pure CoT / Image-Question Direct Ablation Summary

## 实验进度

四组实验均已完成全量 1145 题并完成 merge。

| 实验 | GPU | Shards | 样本数 | Accuracy |
|---|---:|---:|---:|---:|
| pure_cot_answer_first_locked_rounds1_3shards_gpu4 | 4 | 3 | 1145/1145 | 59.17% |
| pure_cot_visual_facts_rounds1_3shards_gpu5 | 5 | 3 | 1145/1145 | 53.20% |
| pure_cot_visual_facts_no_caption_rounds1_3shards_gpu6 | 6 | 3 | 1145/1145 | 53.31% |
| direct_image_question_only_rounds1_3shards_gpu7 | 7 | 3 | 1145/1145 | 58.54% |

## 实验设置

共同设置：

| 参数 | 设置 |
|---|---|
| model | qwen3-VL-4B |
| dataset | A-OKVQA val |
| rounds | 1 |
| n_shot | 1 |
| n_ensemble | 1 |
| caption_type | vinvl |
| 三增强模块 | 不使用 |
| reviewer/direct_verify | 不使用 |

差异设置：

| 实验 | cot_style | context_mode | 结构 |
|---|---|---|---|
| answer_first_locked | answer_first_locked | no_round_state | 先输出 Answer，再给 reasons；最终只抽第一行 Answer |
| visual_facts | visual_facts | no_round_state | 先列最多 2 条 visible facts，再输出 Answer |
| visual_facts_no_caption | visual_facts | empty | 不给 caption，只看图像+问题，先列 visible facts 再答 |
| direct_image_question_only | no-CoT | empty | 只给图像+问题，直接短答 |

## 与关键历史结果对比

| 实验 | Accuracy | 正确数 |
|---|---:|---:|
| direct no-CoT rounds1 | 59.24% | 678/1145 |
| pure_cot_answer_first_locked | 59.17% | 677/1145 |
| reviewer_evidence_caption_only | 59.13% | 676/1145 |
| conflict-only + no_round_state | 59.10% | 676/1145 |
| direct_image_question_only | 58.54% | 670/1145 |
| old CoT rounds1 | 55.33% | 633/1145 |
| visual_facts_no_caption | 53.31% | 610/1145 |
| visual_facts | 53.20% | 609/1145 |

## 结论

1. `answer_first_locked` 是目前最好的纯 CoT 版本，达到 59.17%，只比 direct no-CoT 少 1 题。
2. 这说明 CoT 并不是天然无效，关键是必须先锁定答案，不能让后续推理链污染最终答案。
3. `answer_first_locked` 也超过了 reviewer caption-only 的 59.13%，说明“单阶段 answer-first CoT”目前比 reviewer CoT 更接近 direct best。
4. `visual_facts` 两个版本都明显失败，只有约 53.2%。这说明让模型先显式列视觉事实会严重扰乱答案生成，尤其容易产生过度描述或错误 grounding。
5. `direct_image_question_only` 为 58.54%，低于带基础 caption/context 的 direct no-CoT 59.24%。说明基础 caption/context 对 direct answer 仍有正收益。

## 方法判断

这轮实验改变了之前的判断：不是所有纯 CoT 都比 direct 差。真正有希望的是：

`Answer-first CoT = 先保留 direct answer 能力，再让模型给很短理由，但最终答案只取第一行。`

它本质上避免了传统 CoT 的两个问题：

- 推理链先行导致答案被语言先验带偏；
- 后续理由或格式内容污染答案抽取。

因此，后续如果要继续探索“纯 CoT 超过 direct”，应优先围绕 answer-first locked 做小改动，而不是 visual facts 或 step-by-step。

## 下一步建议

1. 复跑 `answer_first_locked` 一版不同 shard/随机性的确认实验，确认 59.17% 是否稳定。
2. 尝试 `answer_first_locked_no_caption`，判断它是否依赖基础 caption。
3. 尝试 `answer_first_locked_ensemble3`，看答案第一行投票是否能超过 direct no-CoT。
