# 所有已有 Accuracy Log 结果索引

生成时间：2026-06-14

本文件索引 `/data2/lizhengxue/WorkSpace/onion_output/aokvqa` 下已生成 `accuracy.log` 的实验。正在运行且尚未 merge 的实验不在本表最终结果中，见 `04_pure_cot_answer_first_reflective.md`。

## 代码版本提醒

本表是结果索引，不是严格同代码 leaderboard。实验目录来自多个开发阶段：早期 forward baseline、forward2 clean/context、MCTS 参数化、direct_verify、reviewer evidence、pure CoT、reflective answer-first 等。每个阶段的主要代码差异见 `06_code_evolution_and_comparability.md`。

## 完整结果表

| 实验目录 | Accuracy | 正确数 |
|---|---:|---:|
| `qwen3-VL-4B` | 47.24% | 554/1173 |
| `qwen3-VL-4B_forward` | 45.07% | 516/1145 |
| `qwen3-VL-4B_forward_baseline` | 52.72% | 603/1145 |
| `qwen3-VL-4B_forward_no_cot` | 58.04% | 664/1145 |
| `qwen3-VL-4B_forward_rounds1` | 54.85% | 629/1147 |
| `qwen3-VL-4B_forward_rounds3` | 53.27% | 609/1145 |
| `qwen3-VL-4B_forward_remove_caption` | 52.85% | 605/1145 |
| `qwen3-VL-4B_forward_ensemble1` | 52.92% | 605/1145 |
| `qwen3-VL-4B_forward_ensemble3` | 52.86% | 605/1145 |
| `qwen3-VL-4B_forward_ensemble_norm` | 52.43% | 600/1145 |
| `qwen3-VL-4B_forward_nshot0` | 52.52% | 601/1145 |
| `qwen3-VL-4B_forward_nshot4` | 52.48% | 600/1145 |
| `qwen3-VL-4B_forward_sim_question` | 52.45% | 600/1145 |
| `qwen3-VL-4B_forward_caption_vinvl_sg` | 52.69% | 603/1145 |
| `qwen3-VL-4B_forward_caption_vinvl_tag` | 51.71% | 592/1145 |
| `qwen3-VL-4B_forward_ocr` | 52.90% | 605/1145 |
| `qwen3-VL-4B_forward_clip_thought` | 52.86% | 605/1145 |
| `qwen3-VL-4B_forward_qwen_thought` | 53.09% | 607/1145 |
| `qwen3-VL-4B_forward_qwen_caption` | 44.46% | 509/1145 |
| `qwen3-VL-4B_forward_all_regions` | 52.82% | 604/1145 |
| `qwen3-VL-4B_forward_all_added` | 43.90% | 502/1145 |
| `qwen3-VL-4B_forward2_clean_rounds1` | 55.33% | 633/1145 |
| `qwen3-VL-4B_forward2_no_cot_rounds1` | 59.24% | 678/1145 |
| `qwen3-VL-4B_forward2_no_cot_rounds3` | 58.18% | 666/1145 |
| `qwen3-VL-4B_forward2_no_cot_ensemble1` | 58.18% | 666/1145 |
| `qwen3-VL-4B_forward2_no_cot_ensemble3` | 57.36% | 656/1145 |
| `qwen3-VL-4B_forward2_context_empty` | 55.90% | 640/1145 |
| `qwen3-VL-4B_forward2_context_no_round_state` | 55.48% | 635/1145 |
| `qwen3-VL-4B_forward2_context_caption_only` | 55.17% | 631/1145 |
| `qwen3-VL-4B_forward2_context_objects_only` | 55.10% | 630/1145 |
| `qwen3-VL-4B_forward2_answer_strict_final` | 52.43% | 600/1145 |
| `qwen3-VL-4B_forward2_answer_raw` | 52.37% | 599/1145 |
| `qwen3-VL-4B_forward2_answer_last_line` | 52.25% | 598/1145 |
| `qwen3-VL-4B_forward2_qwen_caption_no_final` | 52.66% | 602/1145 |
| `qwen3-VL-4B_forward2_qwen_caption_local` | 45.23% | 517/1145 |
| `qwen3-VL-4B_forward2_qwen_caption_global` | 44.06% | 504/1145 |
| `qwen3-VL-4B_forward2_qwen_caption_short` | 43.87% | 502/1145 |
| `qwen3-VL-4B_forward2_mcts_image_no_cot_rounds1_n5_4shards` | 53.92% | 617/1145 |
| `qwen3-VL-4B_forward2_mcts_safe_no_cot_rounds1_n5_6shards` | 57.48% | 658/1145 |
| `qwen3-VL-4B_forward2_mcts_narrow_no_cot_rounds1_n5_6shards` | 58.72% | 672/1145 |
| `qwen3-VL-4B_forward2_mcts_narrow_no_cot_rounds1_n10_6shards` | 59.09% | 676/1145 |
| `qwen3-VL-4B_forward2_mcts_narrow_no_cot_rounds1_n20_12shards` | 58.66% | 671/1145 |
| `qwen3-VL-4B_forward2_mcts_marker_narrow_no_cot_rounds1_n1_2shards` | 58.60% | 670/1145 |
| `qwen3-VL-4B_forward2_mcts_marker_narrow_no_cot_rounds1_n5_2shards` | 58.72% | 672/1145 |
| `qwen3-VL-4B_forward2_mcts_marker_narrow_no_cot_rounds1_n10_4shards` | 58.76% | 672/1145 |
| `qwen3-VL-4B_forward2_mcts_marker_narrow_no_cot_rounds1_n20_2shards` | 58.97% | 675/1145 |
| `qwen3-VL-4B_forward2_mcts_marker_visual_no_cot_rounds1_n10_2shards` | 58.59% | 670/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_compact_rounds1_2shards` | 57.70% | 660/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_compact_rounds3_4shards` | 58.10% | 665/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_compact_rounds5_4shards` | 57.38% | 657/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_compact_marker_mcts_rounds1_n10_2shards` | 58.40% | 668/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_compact_outline_mcts_rounds1_n10_2shards` | 58.36% | 668/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_answer_first_rounds1_2shards` | 1.23% | 14/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_no_round_state_rounds1_2shards` | 59.10% | 676/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_caption_only_rounds1_2shards` | 58.93% | 674/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_objects_only_rounds1_2shards` | 54.64% | 625/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_ocr_context_rounds1_3shards` | 58.72% | 672/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_marker_mcts_rounds1_n5_3shards` | 58.90% | 674/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_marker_mcts_rounds1_n10_3shards` | 58.72% | 672/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_marker_mcts_rounds1_n20_3shards` | 58.79% | 673/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_conflict_outline_mcts_rounds1_n10_3shards` | 58.66% | 671/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_rounds1_4shards` | 57.34% | 656/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_conflict_only_rounds1_4shards` | 58.96% | 675/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_no_fallback_rounds1_4shards` | 57.57% | 659/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_keep_stronger_rounds1_4shards` | 56.51% | 646/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_qwen_caption_rounds1_4shards` | 49.48% | 566/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards` | 57.28% | 655/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards` | 56.97% | 652/1145 |
| `qwen3-VL-4B_forward2_cot_rescue_direct_verify_keep_stronger_marker_mcts_rounds1_n10_4shards` | 56.38% | 645/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_all_rounds1_1shard` | 58.49% | 669/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_all_rounds1_2shards` | 58.54% | 670/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_all_rounds1_tmux_debug` | 58.49% | 669/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_caption_only_rounds1_2shards` | 57.39% | 657/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_caption_only_rounds1_3shards_gpu4` | 59.13% | 676/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_caption_only_strict_rounds1_2shards_gpu7` | 58.51% | 669/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_caption_object_rounds1_2shards` | 57.97% | 663/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_caption_object_rounds1_3shards_gpu6` | 58.59% | 670/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_no_objects_rounds1_2shards` | 58.59% | 670/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_no_objects_rounds1_3shards_gpu5` | 58.45% | 669/1145 |
| `qwen3-VL-4B_forward2_reviewer_evidence_selective_rounds1_3shards_gpu456` | 58.49% | 669/1145 |
| `qwen3-VL-4B_forward2_pure_cot_answer_first_locked_rounds1_3shards_gpu4` | 59.17% | 677/1145 |
| `qwen3-VL-4B_forward2_pure_cot_visual_facts_rounds1_3shards_gpu5` | 53.20% | 609/1145 |
| `qwen3-VL-4B_forward2_pure_cot_visual_facts_no_caption_rounds1_3shards_gpu6` | 53.31% | 610/1145 |
| `qwen3-VL-4B_forward2_direct_image_question_only_rounds1_3shards_gpu7` | 58.54% | 670/1145 |
| `qwen3-VL-4B_forward2_reflective_answer_first_caption_r3_rounds1_3shards_gpu4` | 59.60% | 682/1145 |
| `qwen3-VL-4B_forward2_reflective_answer_first_no_caption_r3_rounds1_3shards_gpu5` | 59.22% | 678/1145 |
| `qwen3-VL-4B_forward2_answer_first_locked_no_caption_rounds1_3shards_gpu6` | 59.51% | 681/1145 |
| `qwen3-VL-4B_forward2_reflective_answer_first_caption_r5_rounds1_3shards_gpu7` | 58.68% | 671/1145 |
| `qwen3-VL-4B_forward2_reflective_adaptive_highrisk_lowconf_caption_r3_3shards_gpu3` | 58.59% | 670/1145 |
| `qwen3-VL-4B_forward2_reflective_keep_revise_caption_r3_3shards_gpu4` | 59.24% | 678/1145 |
| `qwen3-VL-4B_forward2_reflective_visible_only_caption_r3_3shards_gpu5` | 59.21% | 677/1145 |
| `qwen3-VL-4B_forward2_reflective_review_empty_context_caption_r3_3shards_gpu6` | 59.46% | 680/1145 |
| `qwen3-VL-4B_forward2_reflective_initial_ensemble3_keep_revise_caption_r3_3shards_gpu7` | 58.76% | 672/1145 |

## 读取说明

- 表内只列已完成 merge 并写出 `accuracy.log` 的实验。
- `rounds1` 旧实验为 `629/1147`，存在重复 shard 影响；更建议引用 `clean_rounds1`。
- 几个差异小于 0.2 个百分点的结果，只对应 1-3 道题，建议谨慎解释。
