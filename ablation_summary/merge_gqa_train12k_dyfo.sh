#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
DATASET_ROOT="/data2/lizhengxue/datasets"
SUBSET_JSON="${GQA_QUESTION_FILE:-${REPO}/ablation_summary/router_data/gqa_train_balanced_seed42_12000_questions.json}"
EXP_OUT="/data2/lizhengxue/WorkSpace/onion_output/gqa/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"

"$PY" "$REPO/forward_code/onion.py" \
  --dataset_name gqa \
  --split_name train \
  --engine qwen3-VL-4B \
  --caption_type vinvl \
  --n_shot 0 \
  --n_ensemble 1 \
  --rounds 1 \
  --iterative_strategy caption \
  --dataset_root "$DATASET_ROOT" \
  --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations \
  --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa \
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text \
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags \
  --train_sim_metric answer \
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk \
  --output_path "$EXP_OUT" \
  --cache_path "${EXP_OUT}/cache_merge" \
  --num_shards 3 \
  --gqa_question_file "$SUBSET_JSON" \
  --context_mode no_round_state \
  --use_image_enhance \
  --mcts_action_mode dyfo_evidence \
  --use_dyfo_visual_evidence \
  --dyfo_decision_mode weighted_vote \
  --dyfo_trigger_mode visual_detail \
  --dyfo_n_simulations 6 \
  --dyfo_max_depth 3 \
  --dyfo_area_reward compact \
  --merge_only \
  --summary_log "$SUMMARY_LOG"
