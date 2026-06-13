# Reviewer Evidence Initial Experiment Summary

## 实验进度

本次已完成 1 个全量初步实验：

| 实验 | 状态 | 样本数 | 耗时 | Accuracy |
|---|---|---:|---:|---:|
| reviewer_evidence_all_rounds1 | 已完成并 merge | 1145/1145 | 约 2h01m | 58.49% |

输出目录：

`/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_reviewer_evidence_all_rounds1_tmux_debug`

accuracy log：

`/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_reviewer_evidence_all_rounds1_tmux_debug/accuracy.log`

## 主要参数

| 参数 | 设置 |
|---|---|
| model | qwen3-VL-4B |
| dataset | A-OKVQA val |
| samples | 1145 |
| rounds | 1 |
| n_shot | 1 |
| n_ensemble | 1 |
| caption_type | vinvl |
| cot_style | reviewer_evidence |
| direct_verify_policy | conflict_only |
| context_mode | no_round_state |
| reviewer_evidence_scope | all |
| use_caption_enhance | true |
| use_knowledge_enhance | true |
| use_image_enhance / MCTS | false |

## 方法逻辑

这版代码把增强模块改成 evidence provider：

1. Qwen 先直接回答，使用原始图像和基础 caption。
2. reviewer 再看到 evidence，包括基础 caption、选中的 object、caption enhance、knowledge enhance 等。
3. reviewer 只判断 evidence 是否明确推翻初始答案。
4. `conflict_only` 策略下，只有 `Evidence Check: contradicted` 才允许改答案；`supported` 或 `uncertain` 都保留初始答案。

## 与历史结果对比

| 实验 | Accuracy | 正确数 | 对比 |
|---|---:|---:|---|
| direct no-CoT rounds1 | 59.24% | 678/1145 | 当前 best |
| direct_verify conflict_only + no_round_state | 59.10% | 676/1145 | 最强 verification 版本 |
| direct_verify conflict_only | 58.96% | 675/1145 | 稳定接近 best |
| reviewer_evidence_all_rounds1 | 58.49% | 669/1145 | 本次新实验 |
| compact marker MCTS n10 | 58.40% | 668/1145 | compact CoT 系列最好 |
| old CoT rounds1 | 55.33% | 633/1145 | 原始 CoT |

## 结论

1. reviewer-evidence 这版没有超过 direct no-CoT，也没有超过之前的 `conflict_no_round_state`。
2. 但它明显高于原始 CoT，说明“审稿人 CoT”方向不是失败的；只是当前 evidence 质量还不够强。
3. 当前 `all` scope 会把 object evidence 放进去，而历史结果显示 object-only 很伤结果；这可能拉低了 reviewer-evidence 的表现。
4. 这版没有启用 MCTS/image enhance，所以“图像最小提示”还没真正进入 evidence provider 流程。
5. 日志里可以看到 reviewer 有时会把并不在 evidence 文本里的视觉细节说成 supported，例如直接声称看到了某个数字/文字。这说明 reviewer prompt 仍然需要更强的证据约束，避免它自己补证据。

## 下一步建议

优先补跑两个更干净的消融：

| 建议实验 | 目的 |
|---|---|
| reviewer_evidence_caption_only | 验证只保留 caption evidence 是否比 all 更稳 |
| reviewer_evidence_no_objects | 去掉 object evidence，测试 object 噪声是否是主要掉点来源 |

如果这两个高于 58.49%，说明 evidence scope 需要收窄。之后再加一版轻量 MCTS/marker evidence，不建议马上并发跑多进程。
