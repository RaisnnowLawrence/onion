#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 2
fi

RUN_ID="$1"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
EXP_OUT="/data2/lizhengxue/WorkSpace/onion_output/gqa/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
NUM_SHARDS="${NUM_SHARDS:-6}"
SPLIT_NAME="${SPLIT_NAME:-testdev}"
DYFO_DECISION_MODE="${DYFO_DECISION_MODE:-evidence_inject}"
DYFO_ANSWER_IMAGE_MODE="${DYFO_ANSWER_IMAGE_MODE:-concat_horizontal}"

COMMON_ARGS=(
  "$REPO/forward_code/onion.py"
  --dataset_name gqa
  --split_name "$SPLIT_NAME"
  --engine qwen3-VL-4B
  --caption_type vinvl
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --context_mode no_round_state
  --use_image_enhance
  --mcts_action_mode dyfo_evidence
  --use_dyfo_visual_evidence
  --dyfo_use_focus_image_as_answer
  --dyfo_answer_image_mode "$DYFO_ANSWER_IMAGE_MODE"
  --dyfo_decision_mode "$DYFO_DECISION_MODE"
  --dyfo_trigger_mode visual_detail
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
  --dataset_root /data2/lizhengxue/datasets
  --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations
  --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
  --train_sim_metric answer
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
  --output_path "$EXP_OUT"
  --cache_path "${EXP_OUT}/cache"
  --num_shards "$NUM_SHARDS"
)

count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
echo "[merge] RUN_ID=${RUN_ID}"
echo "[merge] samples=${count}"
echo "[merge] summary=${SUMMARY_LOG}"

if [[ "$count" -eq 0 ]]; then
  echo "[merge] skip: no prompt_samples yet"
  exit 0
fi

"$PY" "${COMMON_ARGS[@]}" --merge_only --summary_log "$SUMMARY_LOG"
