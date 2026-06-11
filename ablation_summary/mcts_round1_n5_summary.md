# MCTS n=5 Ablation Summary

Generated: 2026-06-11

## Experiment

| Item | Value |
|---|---|
| Experiment name | `mcts_image_no_cot_rounds1_n5_4shards` |
| Output dir | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_mcts_image_no_cot_rounds1_n5_4shards` |
| Base setting | `no_cot_rounds1` |
| MCTS setting | `--use_image_enhance --mcts_n_simulations 5` |
| Shards | 4 shards |
| GPU allocation | shard0 -> GPU0, shard1 -> GPU1, shard2/3 -> GPU6 |
| Start time | 2026-06-11 14:29:32 |
| Merge time | 2026-06-11 17:02:03 |
| Samples | 1145/1145 |

## Result

| Experiment | Accuracy | Score | Delta vs baseline | Delta vs no_cot_rounds1 |
|---|---:|---:|---:|---:|
| baseline CoT | 52.72 | 603/1145 | +0.00 | -6.52 |
| no_cot_rounds1 | 59.24 | 678/1145 | +6.52 | +0.00 |
| MCTS image enhance, n=5 | 53.92 | 617/1145 | +1.20 | -5.32 |

## Takeaways

1. MCTS image enhancement with `n_simulations=5` improves over the original CoT baseline by `+1.20`, but it is far below the best no-CoT setting.
2. Compared with `no_cot_rounds1`, MCTS drops by `5.32` points, so this image-enhancement path is not helpful for the current best direct-answer pipeline.
3. The run took about `2.5` hours with 4 shards. This is much slower than ordinary no-CoT inference, although much faster than the original `n_simulations=20` estimate.
4. The likely failure mode is that MCTS/SAM-enhanced images do not reliably preserve the information needed for A-OKVQA. Some questions depend on global context, text, commonsense, or details outside the selected mask.

## Recommendation

Do not use `--use_image_enhance` as a default for the best-performing pipeline. If MCTS is kept for further study, it should be tested selectively:

- only on questions routed to visual-detail reasoning,
- only when object selection confidence is high,
- or with a fallback to the original image when SAM/MCTS confidence is weak.

For accuracy-oriented experiments, continue using `no_cot_rounds1` as the main baseline.
