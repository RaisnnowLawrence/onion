# Safe MCTS n=5 6-shard Result Summary

## Experiment

- Name: `mcts_safe_no_cot_rounds1_n5_6shards`
- Output: `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_mcts_safe_no_cot_rounds1_n5_6shards`
- Model: `qwen3-VL-4B`
- Dataset: A-OKVQA validation, 1145 samples
- Main setting: no CoT, rounds=1, n_ensemble=5, n_shot=1, caption_type=vinvl
- MCTS setting: `--use_image_enhance --mcts_n_simulations 5 --mcts_trigger_mode visual_detail_only --mcts_action_mode outline_only --mcts_filter_objects`
- Runtime: about 50 minutes including merge, 6 shards

## Result

| Experiment | Accuracy | Correct | Delta vs baseline | Delta vs best no-CoT |
| --- | ---: | ---: | ---: | ---: |
| baseline CoT | 52.72% | 603/1145 | +0.00 | -6.52 |
| old MCTS n=5 | 53.92% | 617/1145 | +1.20 | -5.32 |
| safe MCTS n=5 | 57.48% | 658/1145 | +4.76 | -1.76 |
| best no-CoT rounds1 | 59.24% | 678/1145 | +6.52 | +0.00 |

## Interpretation

The safe MCTS change is much better than the old MCTS setting. It gains +3.56 points over old MCTS n=5 and recovers most of the gap to the best direct no-CoT result.

However, it is still 1.76 points below `no_cot_rounds1`, so MCTS should not replace the current best default setting.

Log counts show the conservative trigger worked:

- Skipped by trigger rule: 849 samples
- Entered SAM/MCTS object setup: 280 samples
- No usable MCTS action after detection: 6 samples
- Approximate enhanced samples: 274 samples

This supports the hypothesis that MCTS was previously hurting because it ran too broadly and sometimes changed/cropped the image in ways that removed useful global context. Restricting it to visual-detail questions, filtering objects through the scene graph, and using outline-only actions makes it much safer.

## Chinese Notes

这一版 Safe MCTS 的效果比之前原始 MCTS 好很多，但还没有超过最好的直接 no-CoT。

关键结论：

- 原始 MCTS n=5：53.92%，比最佳 no-CoT 低 5.32 个点。
- Safe MCTS n=5：57.48%，比原始 MCTS 高 3.56 个点。
- 但 Safe MCTS 仍然比最佳 `no_cot_rounds1` 的 59.24% 低 1.76 个点。

说明这次改动方向是有效的：不要让 MCTS 对所有问题都做图像增强，只在更可能需要局部视觉细节的问题上触发，而且只做描边，不裁剪、不缩放。这样能避免破坏全局图像上下文。

推荐使用策略：

- 默认结果仍然使用 `no_cot_rounds1`。
- 如果论文里想讨论 MCTS，可以把 Safe MCTS 作为“改进后的 MCTS 版本”，说明它显著优于 naive MCTS，但当前还没有超过直接问模型。
