# CoT Rescue Ablation Summary

Date: 2026-06-12

Runtime code baseline: `forward_code/onion.py` with `--cot_style compact` and `--cot_style answer_first`.

## Results

| Experiment | Accuracy | Notes |
|---|---:|---|
| `cot_rescue_compact_rounds1_2shards` | 57.70% (660/1145) | Compact visual cues + final answer, 1 round |
| `cot_rescue_answer_first_rounds1_2shards` | 1.23% (14/1145) | Prompt failed; model placed visual cue lists in the answer field |
| `cot_rescue_compact_outline_mcts_rounds1_n10_2shards` | 58.36% (668/1145) | Compact CoT + outline-only MCTS n=10 |
| `cot_rescue_compact_marker_mcts_rounds1_n10_2shards` | 58.40% (668/1145) | Compact CoT + marker-only MCTS n=10 |
| `cot_rescue_compact_rounds3_4shards` | 58.10% (665/1145) | Compact CoT, 3 rounds |
| `cot_rescue_compact_rounds5_4shards` | 57.38% (657/1145) | Compact CoT, 5 rounds |

## Takeaways

- Compact CoT improves over the previous explicit CoT baseline, but does not beat direct no-CoT.
- MCTS gives a small additional gain on top of compact CoT, with marker-only slightly ahead of outline-only in this run.
- More interaction rounds still add noise: 3 rounds is below the best MCTS combination, and 5 rounds drops further.
- The `answer_first` prompt is invalid as tested and should be redesigned rather than rerun unchanged.
