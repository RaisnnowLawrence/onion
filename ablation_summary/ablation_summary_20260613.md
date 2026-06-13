# VisualCoT A-OKVQA Ablation Summary - 2026-06-13

## 当前进度

本轮新跑的 8 个 `conflict_only` 消融，以及之前补跑的 3 个 direct-verify MCTS 实验，都已经完成全量 1145 题并完成 merge。

当前没有正在运行的 `onion.py` 实验进程。`nvidia-smi` 上 GPU2 一度显示约 12G 占用，但对应 PID 在系统进程表中不可见，像是驱动侧残留或瞬时状态；和本轮实验结果无关。

## 最高结果排行

| 排名 | 实验 | Accuracy | 正确数 |
|---:|---|---:|---:|
| 1 | direct no-CoT rounds1 | 59.24% | 678/1145 |
| 2 | direct-verify conflict-only + no round state | 59.10% | 676/1145 |
| 3 | no-CoT MCTS narrow n=10 | 59.09% | 676/1145 |
| 4 | no-CoT marker narrow n=20 | 58.97% | 675/1145 |
| 5 | direct-verify conflict-only | 58.96% | 675/1145 |
| 6 | direct-verify conflict-only + caption only | 58.93% | 674/1145 |
| 7 | direct-verify conflict-only + marker MCTS n=5 | 58.90% | 674/1145 |
| 8 | direct-verify conflict-only + marker MCTS n=20 | 58.79% | 673/1145 |

## 本轮 8 个新实验

| 实验 | 目的 | Accuracy | 正确数 | 结论 |
|---|---|---:|---:|---|
| conflict_no_round_state_rounds1 | 去掉 round state，只保留更干净的验证上下文 | 59.10% | 676/1145 | 本轮最好，几乎追平 direct no-CoT |
| conflict_caption_only_rounds1 | 只给 caption 类上下文 | 58.93% | 674/1145 | 很稳，说明 caption 比 object list 更适合作为验证证据 |
| conflict_objects_only_rounds1 | 只给 objects/scene graph 类上下文 | 54.64% | 625/1145 | 明显伤害结果，object list 噪声较大 |
| conflict_marker_mcts_n5 | conflict-only + marker MCTS，搜索数 5 | 58.90% | 674/1145 | MCTS 没超过干净文本验证，但比旧 CoT 好很多 |
| conflict_marker_mcts_n10 | conflict-only + marker MCTS，搜索数 10 | 58.72% | 672/1145 | 搜索数增加没有带来稳定收益 |
| conflict_marker_mcts_n20 | conflict-only + marker MCTS，搜索数 20 | 58.79% | 673/1145 | n=20 也没有超过 n=5/no-MCTS |
| conflict_outline_mcts_n10 | conflict-only + outline MCTS | 58.66% | 671/1145 | outline 略低于 marker |
| conflict_ocr_context_rounds1 | 加 OCR 上下文 | 58.72% | 672/1145 | 没有明显收益，可能 OCR 覆盖/质量不足 |

## Direct Verify 系列

| 实验 | Accuracy | 正确数 | 说明 |
|---|---:|---:|---|
| direct_verify_rounds1 | 57.34% | 656/1145 | 直接答案后再做验证 |
| direct_verify_no_fallback | 57.57% | 659/1145 | 关闭 cue-list fallback 后略升 |
| direct_verify_keep_stronger | 56.51% | 646/1145 | 更保守策略反而下降 |
| direct_verify_conflict_only | 58.96% | 675/1145 | 只在明确冲突时改答案，最稳 |
| direct_verify_qwen_caption | 49.48% | 566/1145 | Qwen caption 噪声很大 |
| direct_verify_marker_mcts_n10 | 57.28% | 655/1145 | balanced 策略 + MCTS 不理想 |
| direct_verify_outline_mcts_n10 | 56.97% | 652/1145 | outline 更低 |
| direct_verify_keep_stronger_marker_mcts_n10 | 56.38% | 645/1145 | keep_stronger + MCTS 不推荐 |

## Compact CoT Rescue 系列

| 实验 | Accuracy | 正确数 | 说明 |
|---|---:|---:|---|
| old CoT rounds1 / clean rounds1 | 55.33% | 633/1145 | 原始 CoT 表现 |
| compact_rounds1 | 57.70% | 660/1145 | 收窄 CoT 后明显提升 |
| compact_rounds3 | 58.10% | 665/1145 | round3 略升 |
| compact_rounds5 | 57.38% | 657/1145 | round5 反而下降 |
| compact_marker_mcts_n10 | 58.40% | 668/1145 | compact 系列最好 |
| compact_outline_mcts_n10 | 58.36% | 668/1145 | 和 marker 接近 |
| answer_first_rounds1 | 1.23% | 14/1145 | prompt 失败，模型把 cue list 当答案 |

## No-CoT / MCTS 系列

| 实验 | Accuracy | 正确数 | 说明 |
|---|---:|---:|---|
| no_cot_rounds1 | 59.24% | 678/1145 | 当前总体最好 |
| no_cot_rounds3 | 58.18% | 666/1145 | 多轮无收益 |
| no_cot_ensemble1 | 58.18% | 666/1145 | 低于 direct no-CoT |
| no_cot_ensemble3 | 57.36% | 656/1145 | ensemble 没收益 |
| mcts_image_no_cot_n5 | 53.92% | 617/1145 | 直接改图像伤害明显 |
| mcts_safe_no_cot_n5 | 57.48% | 658/1145 | safe 版本仍低 |
| mcts_narrow_no_cot_n5 | 58.72% | 672/1145 | 收窄动作后恢复 |
| mcts_narrow_no_cot_n10 | 59.09% | 676/1145 | no-CoT MCTS 系列最好 |
| mcts_narrow_no_cot_n20 | 58.66% | 671/1145 | n=20 没继续提升 |
| mcts_marker_narrow_no_cot_n1 | 58.60% | 670/1145 | 少量搜索也接近 |
| mcts_marker_narrow_no_cot_n5 | 58.72% | 672/1145 | 稳定但不超 direct |
| mcts_marker_narrow_no_cot_n10 | 58.76% | 672/1145 | 稳定但不超 direct |
| mcts_marker_narrow_no_cot_n20 | 58.97% | 675/1145 | marker narrow 中最好 |
| mcts_marker_visual_no_cot_n10 | 58.59% | 670/1145 | visual marker 没有额外收益 |

## 早期增强/上下文实验

| 实验 | Accuracy | 正确数 | 说明 |
|---|---:|---:|---|
| forward_baseline | 52.72% | 603/1145 | 早期 forward baseline |
| forward_no_cot | 58.04% | 664/1145 | 早期 no-CoT 已显著优于 CoT |
| forward_rounds3 | 53.27% | 609/1145 | 多轮 CoT 较差 |
| forward_qwen_caption | 44.46% | 509/1145 | Qwen caption 明显伤害 |
| forward_qwen_thought | 53.09% | 607/1145 | 小幅高于 baseline |
| forward_clip_thought | 52.86% | 605/1145 | 基本持平 |
| forward_ocr | 52.90% | 605/1145 | OCR 基本无收益 |
| forward_all_added | 43.90% | 502/1145 | 全部增强堆叠严重伤害 |
| context_empty | 55.90% | 640/1145 | 去上下文反而比旧 CoT 稳 |
| context_no_round_state | 55.48% | 635/1145 | 早期版本小幅变化 |
| context_caption_only | 55.17% | 631/1145 | 早期版本无明显收益 |
| context_objects_only | 55.10% | 630/1145 | 早期版本无明显收益 |

## 总结

1. 当前总体最好仍然是 `direct no-CoT rounds1`，59.24%。
2. 最接近 best 的增强链是 `direct_verify conflict_only + no_round_state`，59.10%，只差 0.14 个百分点，即 2 题。
3. CoT 的主要问题不是“不会推理”，而是中间链条会引入格式漂移、噪声证据和错误修正。收窄 prompt 后，CoT 从 55.33% 提到 58% 左右，但仍略低于直接回答。
4. MCTS 如果直接改变图像，会明显掉点；改成“最小必要提示/marker/outline”后损害变小，但目前没有稳定超过 direct no-CoT。
5. `object list / scene graph only` 和 `Qwen caption` 是最危险的两类上下文，容易引入噪声；`caption only` 和 `no round state` 更稳。
6. `n_simulations` 不是越大越好：n=5、10、20 的差距很小，甚至 n=5 更稳。继续堆搜索数大概率不是性价比最高的方向。

## 推荐下一步

优先围绕 `direct_verify_policy=conflict_only` 继续做小步优化：保留 direct answer 的强先验，只允许 verifier 在非常明确的视觉矛盾时改答案。下一步最值得做的是逐题对比 `no_cot_rounds1` 和 `conflict_no_round_state`，找出 2 类样本：验证链救回的问题，以及验证链改错的问题。这个分析比继续盲目加 MCTS 搜索数更可能带来真实提升。
