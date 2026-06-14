# Reviewer Evidence Selective / Strict Summary

## 当前进度

两组实验均已完成全量 1145 题并完成 merge。当前没有 reviewer 实验 tmux 会话在运行。

| 实验 | 切片/GPU | 样本数 | Accuracy |
|---|---|---:|---:|
| reviewer_evidence_selective_rounds1_3shards_gpu456 | shard0->GPU4, shard1->GPU5, shard2->GPU6 | 1145/1145 | 58.49% |
| reviewer_evidence_caption_only_strict_rounds1_2shards_gpu7 | 2 shards on GPU7 | 1145/1145 | 58.51% |

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
| cot_style | reviewer_evidence |
| direct_verify_policy | conflict_only |
| context_mode | no_round_state |

差异：

| 实验 | reviewer_evidence_scope | 说明 |
|---|---|---|
| selective | selective | 默认 caption；按问题类型触发 caption_enhance / knowledge；image 入口需显式 MCTS，本次未开 MCTS |
| caption_only_strict | caption_only | 只给基础 caption；使用新版更严格 reviewer prompt |

## 与关键历史结果对比

| 实验 | Accuracy | 正确数 |
|---|---:|---:|
| direct no-CoT rounds1 | 59.24% | 678/1145 |
| reviewer_evidence_caption_only | 59.13% | 676/1145 |
| conflict-only + no_round_state | 59.10% | 676/1145 |
| direct_verify conflict_only | 58.96% | 675/1145 |
| reviewer_evidence_caption_object | 58.59% | 670/1145 |
| reviewer_evidence_caption_only_strict | 58.51% | 669/1145 |
| reviewer_evidence_selective | 58.49% | 669/1145 |
| reviewer_evidence_all | 58.49% | 669/1145 |
| old CoT rounds1 | 55.33% | 633/1145 |

## 结论

1. `selective` 没有带来收益，结果和之前 `all` 基本一样，都是 58.49%。
2. `caption_only_strict` 也没有复现上一轮 `caption_only` 的 59.13%，反而下降到 58.51%。
3. 这说明新增的强约束句：
   `Do not invent visual details that are not visible or listed.`
   可能让 reviewer 变得过度保守，减少了它从图像中补充判断的能力。
4. 选择性触发 caption/knowledge enhance 没有解决噪声问题，反而仍然低于纯 `caption_only`。
5. 当前最可靠的结果仍然是旧版 `reviewer_evidence_caption_only`，59.13%，只比 no-CoT best 少 1 题。

## 判断

这轮实验说明，三增强模块接回 reviewer 链时仍然会带来噪声或额外不稳定性。更严格的 reviewer prompt 也不是无条件有利，它可能压制了模型原本有用的图像理解。

目前最稳的方向不是继续加证据，而是保留 `caption_only` 的简洁结构，并做更小幅的 prompt 改动。下一步如果继续救三增强，建议只针对错误样本触发增强，而不是按问题类型全局触发。
