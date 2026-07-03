#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-vqav2_qwen4b_visual_enhance_gpu57_6shards_$(date +%Y%m%d_%H%M%S)}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/vqav2"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_vqav2_qwen4b_visual_enhance"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
NUM_SHARDS="${NUM_SHARDS:-6}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-120}"
CHILDREN=()

mkdir -p "$LOG_ROOT" "$EXP_OUT"

merge_partial() {
  local count
  count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
  echo "[merge] samples=${count} out=${EXP_OUT}"
  if [[ "${count}" -eq 0 ]]; then
    echo "[merge] skip: no prompt_samples yet"
    return 0
  fi
  "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name vqav2 \
    --split_name val \
    --engine qwen3-VL-4B \
    --caption_type vinvl \
    --n_shot 0 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode no_round_state \
    --use_image_enhance \
    --mcts_n_simulations 5 \
    --output_path "$EXP_OUT" \
    --coco_path /data2/lizhengxue/datasets/vqav2 \
    --raw_image_dir /data2/lizhengxue/datasets/coco14 \
    --merge_only \
    --summary_log "$SUMMARY_LOG"
}

on_term() {
  echo "[signal] received TERM/INT, stopping children and merging partial results"
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait || true
  merge_partial || true
  exit 143
}
trap on_term TERM INT

gpu_for_shard() {
  case "$1" in
    0|1|2) echo 5 ;;
    3|4|5) echo 7 ;;
    *) echo "unsupported shard id: $1" >&2; return 1 ;;
  esac
}

run_shard() {
  local shard_id="$1"
  local gpu_id
  gpu_id="$(gpu_for_shard "$shard_id")"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${gpu_id}_shard${shard_id}.log"
  echo "[launch] gpu=${gpu_id} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name vqav2 \
    --split_name val \
    --engine qwen3-VL-4B \
    --caption_type vinvl \
    --n_shot 0 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode no_round_state \
    --use_image_enhance \
    --mcts_n_simulations 5 \
    --raw_image_dir /data2/lizhengxue/datasets/coco14 \
    --coco_path /data2/lizhengxue/datasets/vqav2 \
    --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations \
    --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa \
    --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text \
    --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags \
    --train_sim_metric answer \
    --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk \
    --output_path "$EXP_OUT" \
    --cache_path "${EXP_OUT}/cache" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$shard_id" \
    ${MAX_SAMPLES_PER_SHARD:+--max_samples_per_shard "$MAX_SAMPLES_PER_SHARD"} \
    > "$log_file" 2>&1 &
  CHILDREN+=("$!")
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] NUM_SHARDS=${NUM_SHARDS}"
echo "[run] assignment: gpu5=shards0,1,2 gpu7=shards3,4,5"
echo "[run] visual enhancement: --use_image_enhance --mcts_n_simulations 5"
echo "[run] partial merge on TERM/INT"

for shard_id in 0 3 1 4 2 5; do
  run_shard "$shard_id"
  sleep "$SHARD_LAUNCH_DELAY"
done

wait
merge_partial
echo "[done] summary=${SUMMARY_LOG}"
