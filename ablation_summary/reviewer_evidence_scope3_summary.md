# Reviewer Evidence Scope Ablation Summary

## 实验进度

三组 3-shard 实验均已完成并 merge。

| 实验 | GPU | Shards | 样本数 | Accuracy |
|---|---:|---:|---:|---:|
| reviewer_evidence_caption_only_rounds1_3shards_gpu4 | 4 | 3 | 1145/1145 | 59.13% |
| reviewer_evidence_no_objects_rounds1_3shards_gpu5 | 5 | 3 | 1145/1145 | 58.45% |
| reviewer_evidence_caption_object_rounds1_3shards_gpu6 | 6 | 3 | 1145/1145 | 58.59% |

## 参数设置

三组共同设置：

| 参数 | 设置 |
|---|---|
| model | qwen3-VL-4B |
| dataset | A-OKVQA val |
| rounds | 1 |
| n_shot | 1 |
| n_ensemble | 1 |
| caption_type | vinvl |
| cot_style | reviewer_evidence |
| direct_verify_policy | conflict_only |
| context_mode | no_round_state |
| MCTS / image enhance | false |

差异设置：

| 实验 | reviewer_evidence_scope | caption/knowledge enhance |
|---|---|---|
| caption_only | caption_only | false |
| no_objects | no_objects | true |
| caption_object | caption_object | false |

## 与关键历史结果对比

| 实验 | Accuracy | 正确数 |
|---|---:|---:|
| direct no-CoT rounds1 | 59.24% | 678/1145 |
| reviewer_evidence_caption_only | 59.13% | 676/1145 |
| conflict-only + no_round_state | 59.10% | 676/1145 |
| direct_verify conflict_only | 58.96% | 675/1145 |
| reviewer_evidence_caption_object | 58.59% | 670/1145 |
| reviewer_evidence_all | 58.49% | 669/1145 |
| reviewer_evidence_no_objects | 58.45% | 669/1145 |
| old CoT rounds1 | 55.33% | 633/1145 |

## 结论

1. 本轮最好的版本是 `reviewer_evidence_caption_only`，达到 59.13%，只比 direct no-CoT 的 59.24% 少 1 题。
2. `caption_only` 比上一版 `all` 高 0.64 个百分点，也就是多 7 题，说明 evidence 越干净越好。
3. `caption_object` 只有 58.59%，比 `caption_only` 低 6 题，说明 selected object evidence 仍然可能引入噪声或诱导 reviewer 编造视觉支持。
4. `no_objects` 只有 58.45%，低于 `caption_only`。这说明 targeted caption / knowledge enhance 当前没有稳定收益，甚至可能带来额外噪声。
5. 当前最强的 reviewer CoT 形态不是“证据越多越好”，而是“只给基础 caption 做保守复核”。

## 当前判断

这轮结果对方法很关键：CoT 并不是完全没救，真正有效的是极简 reviewer CoT。它和 direct no-CoT 已经只差 1 题，比传统 CoT 高 43 题。

后续如果要继续提升，不建议再堆 object/knowledge。更值得做的是：

- 在 `caption_only` 上加入更严格的 reviewer prompt，禁止 reviewer 声称 caption 中没有出现的视觉细节。
- 只对少数不确定题触发 reviewer，而不是每题都 review。
- 在 `caption_only` 基础上加轻量 MCTS marker，只作为图像提示，不加入 object list。
