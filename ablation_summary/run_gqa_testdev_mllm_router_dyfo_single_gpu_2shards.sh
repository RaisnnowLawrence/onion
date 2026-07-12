#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <gpu_id> [run_id]" >&2
  exit 2
fi

GPU_ID="$1"
RUN_ID="${2:-gqa_testdev_mllm_router_dyfo_gpu${GPU_ID}_2shards_$(date +%Y%m%d_%H%M%S)}"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/gqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_gqa_testdev_mllm_router_dyfo"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
NUM_SHARDS="${NUM_SHARDS:-6}"
SHARD_IDS="${SHARD_IDS:-}"
SPLIT_NAME="${SPLIT_NAME:-testdev}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-120}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-100}"
DYFO_DECISION_MODE="${DYFO_DECISION_MODE:-weighted_vote}"
DYFO_ANSWER_IMAGE_MODE="${DYFO_ANSWER_IMAGE_MODE:-concat_horizontal}"
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
  --dyfo_decision_mode "$DYFO_DECISION_MODE"
  --dyfo_trigger_mode visual_detail
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
  --chain_of_thoughts
  --cot_style multi_strategy_router
  --multi_strategy_router_source mllm
  --multi_strategy_names direct,dyfo
  --multi_strategy_default direct
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
  --max_samples_per_shard "$MAX_SAMPLES_PER_SHARD"
)

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
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${GPU_ID}_shard${shard_id}.log"
  echo "[launch] gpu=${GPU_ID} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PY" "${COMMON_ARGS[@]}" --shard_id "$shard_id" > "$log_file" 2>&1 &
  CHILDREN+=("$!")
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] GPU_ID=${GPU_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] NUM_SHARDS=${NUM_SHARDS}"
echo "[run] SHARD_IDS=${SHARD_IDS:-auto:0..$((NUM_SHARDS - 1))}"
echo "[run] MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD}"
echo "[run] router=multi_strategy_router_source:mllm strategies=direct,dyfo"

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
