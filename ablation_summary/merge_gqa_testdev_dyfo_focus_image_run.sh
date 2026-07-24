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
DYFO_NODE_ANSWER_IMAGE_MODE="${DYFO_NODE_ANSWER_IMAGE_MODE:-concat_horizontal}"
DYFO_TRIGGER_MODE="${DYFO_TRIGGER_MODE:-visual_detail}"
DYFO_FORCE_RUN_ALL_SAMPLES="${DYFO_FORCE_RUN_ALL_SAMPLES:-0}"
DYFO_OVERRIDE_CONFIDENCE_THRESHOLD="${DYFO_OVERRIDE_CONFIDENCE_THRESHOLD:-95}"
DYFO_OVERRIDE_REQUIRED_STRENGTH="${DYFO_OVERRIDE_REQUIRED_STRENGTH:-extreme}"
DYFO_TOKEN_CONFIDENCE_THRESHOLD="${DYFO_TOKEN_CONFIDENCE_THRESHOLD:-0.95}"
DYFO_TOKEN_CONFIDENCE_MARGIN="${DYFO_TOKEN_CONFIDENCE_MARGIN:-0.0}"
DYFO_NODE_CONFIDENCE_THRESHOLD="${DYFO_NODE_CONFIDENCE_THRESHOLD:-0.80}"
DYFO_NODE_CONFIDENCE_MARGIN="${DYFO_NODE_CONFIDENCE_MARGIN:-0.10}"
DYFO_NODE_CONFIDENCE_SUPPORT_RATIO="${DYFO_NODE_CONFIDENCE_SUPPORT_RATIO:-0.60}"
DYFO_NODE_CONFIDENCE_MIN_SUPPORT="${DYFO_NODE_CONFIDENCE_MIN_SUPPORT:-2}"
DYFO_TEXT_FOCUS_USE_IMAGE="${DYFO_TEXT_FOCUS_USE_IMAGE:-0}"
DYFO_FOCUS_PADDING="${DYFO_FOCUS_PADDING:-1.2}"
DYFO_DUAL_VISUAL_EXPERTS="${DYFO_DUAL_VISUAL_EXPERTS:-0}"
DYFO_OWLV2_MODEL_PATH="${DYFO_OWLV2_MODEL_PATH:-/data2/lizhengxue/WorkSpace/PreTrainModel/owlv2/owlv2-large-patch14-ensemble}"
DYFO_OWLV2_THRESHOLD="${DYFO_OWLV2_THRESHOLD:-0.10}"
DYFO_DUAL_IOU_THRESHOLD="${DYFO_DUAL_IOU_THRESHOLD:-0.60}"
DYFO_DUAL_IOU_DELTA="${DYFO_DUAL_IOU_DELTA:-0.10}"
DYFO_NODE_ANSWER_IMAGE_MODE="${DYFO_NODE_ANSWER_IMAGE_MODE:-concat_horizontal}"
DYFO_CLIP_STATEMENT_MARGIN="${DYFO_CLIP_STATEMENT_MARGIN:-0.0}"
DYFO_CLIP_STATEMENT_FOCUS_GAIN="${DYFO_CLIP_STATEMENT_FOCUS_GAIN:-0.0}"

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
  --dyfo_node_answer_image_mode "$DYFO_NODE_ANSWER_IMAGE_MODE"
  --dyfo_decision_mode "$DYFO_DECISION_MODE"
  --dyfo_trigger_mode "$DYFO_TRIGGER_MODE"
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
  --dyfo_focus_padding "$DYFO_FOCUS_PADDING"
  --dyfo_owlv2_model_path "$DYFO_OWLV2_MODEL_PATH"
  --dyfo_owlv2_threshold "$DYFO_OWLV2_THRESHOLD"
  --dyfo_dual_iou_threshold "$DYFO_DUAL_IOU_THRESHOLD"
  --dyfo_dual_iou_delta "$DYFO_DUAL_IOU_DELTA"
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
  --dyfo_override_confidence_threshold "$DYFO_OVERRIDE_CONFIDENCE_THRESHOLD"
  --dyfo_override_required_strength "$DYFO_OVERRIDE_REQUIRED_STRENGTH"
  --dyfo_token_confidence_threshold "$DYFO_TOKEN_CONFIDENCE_THRESHOLD"
  --dyfo_token_confidence_margin "$DYFO_TOKEN_CONFIDENCE_MARGIN"
  --dyfo_node_confidence_threshold "$DYFO_NODE_CONFIDENCE_THRESHOLD"
  --dyfo_node_confidence_margin "$DYFO_NODE_CONFIDENCE_MARGIN"
  --dyfo_node_confidence_support_ratio "$DYFO_NODE_CONFIDENCE_SUPPORT_RATIO"
  --dyfo_node_confidence_min_support "$DYFO_NODE_CONFIDENCE_MIN_SUPPORT"
  --dyfo_clip_statement_margin "$DYFO_CLIP_STATEMENT_MARGIN"
  --dyfo_clip_statement_focus_gain "$DYFO_CLIP_STATEMENT_FOCUS_GAIN"
)

if [[ "$DYFO_FORCE_RUN_ALL_SAMPLES" == "1" ]]; then
  COMMON_ARGS+=(--dyfo_force_run_all_samples)
fi

if [[ "$DYFO_TEXT_FOCUS_USE_IMAGE" == "1" ]]; then
  COMMON_ARGS+=(--dyfo_text_focus_use_image)
fi

if [[ "$DYFO_DUAL_VISUAL_EXPERTS" == "1" ]]; then
  COMMON_ARGS+=(--dyfo_dual_visual_experts)
fi

count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
echo "[merge] RUN_ID=${RUN_ID}"
echo "[merge] samples=${count}"
echo "[merge] summary=${SUMMARY_LOG}"

if [[ "$count" -eq 0 ]]; then
  echo "[merge] skip: no prompt_samples yet"
  exit 0
fi

"$PY" "${COMMON_ARGS[@]}" --merge_only --summary_log "$SUMMARY_LOG"
