# VisualCoT A-OKVQA Ablation Report

Generated: 2026-06-10 11:08:47

| ID | Experiment | Status | Samples | Accuracy | Score | Params | Output |
|---|---|---:|---:|---:|---:|---|---|
| 0 | baseline | merged | 1145 | 52.72 | 603/1145 | `--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_baseline` |
| A1 | remove_caption | merged | 1145 | 52.85 | 605/1145 | `--remove_caption --caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_remove_caption` |
| A2 | no_cot | merged | 1145 | 58.04 | 664/1145 | `--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_no_cot` |
| A3.1 | rounds1 | merged | 1147 | 54.85 | 629/1147 | `--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 1 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_rounds1` |
| A3.2 | rounds3 | merged | 1145 | 53.27 | 609/1145 | `--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 3 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_rounds3` |
| A4.1 | ensemble1 | merged | 1145 | 52.92 | 605/1145 | `--caption_type vinvl --n_shot 1 --n_ensemble 1 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_ensemble1` |
| A4.2 | ensemble3 | merged | 1145 | 52.86 | 605/1145 | `--caption_type vinvl --n_shot 1 --n_ensemble 3 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_ensemble3` |
| A5.1 | nshot0 | merged | 1145 | 52.52 | 601/1145 | `--caption_type vinvl --n_shot 0 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_nshot0` |
| A5.2 | nshot4 | merged | 1145 | 52.48 | 600/1145 | `--caption_type vinvl --n_shot 4 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_nshot4` |
| A6 | sim_question | merged | 1145 | 52.45 | 600/1145 | `--similarity_metric question --caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_sim_question` |
| A7.1 | caption_vinvl_tag | merged | 1145 | 51.71 | 592/1145 | `--caption_type vinvl_tag --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_caption_vinvl_tag` |
| A7.2 | caption_vinvl_sg | merged | 1145 | 52.69 | 603/1145 | `--caption_type vinvl_sg --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_caption_vinvl_sg` |
| B1 | ocr | merged | 1145 | 52.90 | 605/1145 | `--use_ocr_context` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_ocr` |
| B2 | clip_thought | merged | 1145 | 52.86 | 605/1145 | `--use_clip_thought_verify` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_clip_thought` |
| B3 | qwen_caption | merged | 1145 | 44.46 | 509/1145 | `--use_qwen_blip2_caption` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_qwen_caption` |
| B4 | qwen_thought | merged | 1145 | 53.09 | 607/1145 | `--use_qwen_blip2_thought_verify` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_qwen_thought` |
| B5 | all_regions | merged | 1145 | 52.82 | 604/1145 | `--use_all_regional_captions` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_all_regions` |
| B6 | ensemble_norm | merged | 1145 | 52.43 | 600/1145 | `--ensemble_strategy normalized_majority` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_ensemble_norm` |
| B7 | all_added | merged | 1145 | 43.90 | 502/1145 | `--use_ocr_context --use_clip_thought_verify --use_qwen_blip2_caption --use_qwen_blip2_thought_verify --use_all_regional_captions --ensemble_strategy normalized_majority` | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_all_added` |

## Notes

- A1-A7 are the no-code ablations discussed after the added-module ablation run.
- A8 image enhancement is intentionally excluded from the follow-up run.
- B-series entries are the added-module ablations already run in this session.
