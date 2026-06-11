# VisualCoT Follow-up 2 Ablation Report

| ID | Experiment | Status | Samples | Accuracy | Score | Params |
|---|---|---:|---:|---:|---:|---|
| C1 | clean_rounds1 | merged | 1145 | 55.33 | 633/1145 | `--rounds 1 --chain_of_thoughts` |
| NC1 | no_cot_rounds1 | merged | 1145 | 59.24 | 678/1145 | `--rounds 1` |
| NC3 | no_cot_rounds3 | merged | 1145 | 58.18 | 666/1145 | `--rounds 3` |
| NE1 | no_cot_ensemble1 | merged | 1145 | 58.18 | 666/1145 | `--n_ensemble 1` |
| NE3 | no_cot_ensemble3 | merged | 1145 | 57.36 | 656/1145 | `--n_ensemble 3` |
| CTX0 | context_empty | merged | 1145 | 55.90 | 640/1145 | `--context_mode empty --chain_of_thoughts` |
| CTXC | context_caption_only | merged | 1145 | 55.17 | 631/1145 | `--context_mode caption_only --chain_of_thoughts` |
| CTXO | context_objects_only | merged | 1145 | 55.10 | 630/1145 | `--context_mode objects_only --chain_of_thoughts` |
| CTXN | context_no_round_state | running_or_partial | 1042 |  |  | `--context_mode no_round_state --chain_of_thoughts` |
| QG | qwen_caption_global | merged | 1145 | 44.06 | 504/1145 | `--use_qwen_blip2_caption --qwen_caption_mode global --chain_of_thoughts` |
| QL | qwen_caption_local | running_or_partial | 620 |  |  | `--use_qwen_blip2_caption --qwen_caption_mode local --chain_of_thoughts` |
| QS | qwen_caption_short | running_or_partial | 569 |  |  | `--use_qwen_blip2_caption --qwen_caption_max_tokens 48 --qwen_caption_final_max_chars 220 --chain_of_thoughts` |
| QNF | qwen_caption_no_final | merged | 1145 | 52.66 | 602/1145 | `--use_qwen_blip2_caption --qwen_caption_no_final_context --chain_of_thoughts` |
| AS | answer_strict_final | running_or_partial | 637 |  |  | `--answer_extraction_strategy strict_final --chain_of_thoughts` |
| AL | answer_last_line | merged | 1145 | 52.25 | 598/1145 | `--answer_extraction_strategy last_line --chain_of_thoughts` |
| AR | answer_raw | merged | 1145 | 52.37 | 599/1145 | `--answer_extraction_strategy raw --chain_of_thoughts` |
