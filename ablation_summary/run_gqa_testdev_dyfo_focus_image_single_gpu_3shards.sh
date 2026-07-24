#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <gpu_id> [run_id]" >&2
  exit 2
fi

GPU_ID="$1"
RUN_ID="${2:-gqa_testdev_dyfo_focus_concat_gpu${GPU_ID}_3shards_$(date +%Y%m%d_%H%M%S)}"
LOG_TAG="${LOG_TAG:-$RUN_ID}"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/gqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_gqa_testdev_dyfo_focus_image"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
NUM_SHARDS="${NUM_SHARDS:-3}"
SHARD_IDS="${SHARD_IDS:-}"
SPLIT_NAME="${SPLIT_NAME:-testdev}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-150}"
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
DYFO_REGION_AUDIT="${DYFO_REGION_AUDIT:-0}"
DYFO_REGION_AUDIT_SAVE_CROPS="${DYFO_REGION_AUDIT_SAVE_CROPS:-0}"
DYFO_CLIP_STATEMENT_MARGIN="${DYFO_CLIP_STATEMENT_MARGIN:-0.0}"
DYFO_CLIP_STATEMENT_FOCUS_GAIN="${DYFO_CLIP_STATEMENT_FOCUS_GAIN:-0.0}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-}"
MERGE_ON_DONE="${MERGE_ON_DONE:-1}"
CHILDREN=()

mkdir -p "$LOG_ROOT" "$EXP_OUT"

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

if [[ "$DYFO_REGION_AUDIT" == "1" ]]; then
  COMMON_ARGS+=(--dyfo_region_audit)
fi

if [[ "$DYFO_REGION_AUDIT_SAVE_CROPS" == "1" ]]; then
  COMMON_ARGS+=(--dyfo_region_audit_save_crops)
fi

if [[ -n "$MAX_SAMPLES_PER_SHARD" ]]; then
  COMMON_ARGS+=(--max_samples_per_shard "$MAX_SAMPLES_PER_SHARD")
fi

merge_partial() {
  local count
  count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
  echo "[merge] samples=${count} out=${EXP_OUT}"
  if [[ "${count}" -eq 0 ]]; then
    echo "[merge] skip: no prompt_samples yet"
    return 0
  fi
  "$PY" "${COMMON_ARGS[@]}" --merge_only --summary_log "$SUMMARY_LOG"
}

stop_children() {
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

on_term() {
  echo "[signal] received TERM/INT, stopping children and merging partial results"
  stop_children
  wait || true
  merge_partial || true
  exit 143
}
trap on_term TERM INT

run_shard() {
  local shard_id="$1"
  local log_file="${LOG_ROOT}/${LOG_TAG}_gpu${GPU_ID}_shard${shard_id}.log"
  echo "[launch] gpu=${GPU_ID} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PY" "${COMMON_ARGS[@]}" --shard_id "$shard_id" > "$log_file" 2>&1 &
  CHILDREN+=("$!")
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] LOG_TAG=${LOG_TAG}"
echo "[run] GPU_ID=${GPU_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] NUM_SHARDS=${NUM_SHARDS}"
echo "[run] SHARD_IDS=${SHARD_IDS:-auto:0..$((NUM_SHARDS - 1))}"
echo "[run] SPLIT_NAME=${SPLIT_NAME}"
echo "[run] DYFO_DECISION_MODE=${DYFO_DECISION_MODE}"
echo "[run] DYFO_ANSWER_IMAGE_MODE=${DYFO_ANSWER_IMAGE_MODE}"
echo "[run] DYFO_NODE_ANSWER_IMAGE_MODE=${DYFO_NODE_ANSWER_IMAGE_MODE}"
echo "[run] DYFO_TRIGGER_MODE=${DYFO_TRIGGER_MODE}"
echo "[run] DYFO_FORCE_RUN_ALL_SAMPLES=${DYFO_FORCE_RUN_ALL_SAMPLES}"
echo "[run] DYFO_OVERRIDE_CONFIDENCE_THRESHOLD=${DYFO_OVERRIDE_CONFIDENCE_THRESHOLD}"
echo "[run] DYFO_OVERRIDE_REQUIRED_STRENGTH=${DYFO_OVERRIDE_REQUIRED_STRENGTH}"
echo "[run] DYFO_NODE_CONFIDENCE_THRESHOLD=${DYFO_NODE_CONFIDENCE_THRESHOLD}"
echo "[run] DYFO_NODE_CONFIDENCE_MARGIN=${DYFO_NODE_CONFIDENCE_MARGIN}"
echo "[run] DYFO_NODE_CONFIDENCE_SUPPORT_RATIO=${DYFO_NODE_CONFIDENCE_SUPPORT_RATIO}"
echo "[run] DYFO_NODE_CONFIDENCE_MIN_SUPPORT=${DYFO_NODE_CONFIDENCE_MIN_SUPPORT}"
echo "[run] DYFO_TEXT_FOCUS_USE_IMAGE=${DYFO_TEXT_FOCUS_USE_IMAGE}"
echo "[run] DYFO_FOCUS_PADDING=${DYFO_FOCUS_PADDING}"
echo "[run] DYFO_DUAL_VISUAL_EXPERTS=${DYFO_DUAL_VISUAL_EXPERTS}"
echo "[run] DYFO_OWLV2_MODEL_PATH=${DYFO_OWLV2_MODEL_PATH}"
echo "[run] DYFO_OWLV2_THRESHOLD=${DYFO_OWLV2_THRESHOLD}"
echo "[run] DYFO_DUAL_IOU_THRESHOLD=${DYFO_DUAL_IOU_THRESHOLD}"
echo "[run] DYFO_DUAL_IOU_DELTA=${DYFO_DUAL_IOU_DELTA}"
echo "[run] DYFO_NODE_ANSWER_IMAGE_MODE=${DYFO_NODE_ANSWER_IMAGE_MODE}"
echo "[run] DYFO_REGION_AUDIT=${DYFO_REGION_AUDIT}"
echo "[run] DYFO_REGION_AUDIT_SAVE_CROPS=${DYFO_REGION_AUDIT_SAVE_CROPS}"
echo "[run] DYFO_CLIP_STATEMENT_MARGIN=${DYFO_CLIP_STATEMENT_MARGIN}"
echo "[run] DYFO_CLIP_STATEMENT_FOCUS_GAIN=${DYFO_CLIP_STATEMENT_FOCUS_GAIN}"
echo "[run] MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-full}"
echo "[run] dyfo: evidence + focus-image answer, n_sim=6, max_depth=3, area_reward=compact"
echo "[run] merge after all shards finish; partial merge on TERM/INT"

if [[ -z "$SHARD_IDS" ]]; then
  SHARD_IDS="$(seq 0 $((NUM_SHARDS - 1)))"
fi

shard_count="$(wc -w <<< "$SHARD_IDS" | awk '{print $1}')"
shard_idx=0
for shard_id in $SHARD_IDS; do
  run_shard "$shard_id"
  shard_idx=$((shard_idx + 1))
  if [[ "$shard_idx" -lt "$shard_count" ]]; then
    sleep "$SHARD_LAUNCH_DELAY"
  fi
done

status=0
for pid in "${CHILDREN[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$MERGE_ON_DONE" == "1" ]]; then
  merge_partial
else
  echo "[done] MERGE_ON_DONE=0; merge is expected to run as a separate task"
fi
echo "[done] summary=${SUMMARY_LOG}"
exit "$status"
