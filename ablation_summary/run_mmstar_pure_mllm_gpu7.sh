#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
GPU_ID="${GPU_ID:-7}"
RUN_BATCH_ID="${1:-mmstar_pure_mllm_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/mmstar"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_mmstar_pure_mllm"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-120}"
CHILDREN=()

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

common_args() {
  local engine="$1"
  local exp_out="$2"
  local cache_path="$3"
  local num_shards="$4"

  printf '%s\n' \
    "$REPO/forward_code/onion.py" \
    --dataset_name mmstar \
    --split_name val \
    --engine "$engine" \
    --caption_type vinvl \
    --n_shot 0 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode empty \
    --remove_caption \
    --dataset_root /data2/lizhengxue/datasets \
    --coco_path /data2/lizhengxue/datasets/mmstar \
    --raw_image_dir /data2/lizhengxue/datasets/mmstar \
    --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations \
    --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa \
    --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text \
    --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags \
    --train_sim_metric answer \
    --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk \
    --output_path "$exp_out" \
    --cache_path "$cache_path" \
    --num_shards "$num_shards"
}

merge_exp() {
  local label="$1"
  local engine="$2"
  local exp_out="$3"
  local num_shards="$4"
  local summary_log="${exp_out}/summary.log"
  local count

  count="$(find "${exp_out}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
  echo "[merge] ${label} samples=${count} out=${exp_out}"
  if [[ "$count" -eq 0 ]]; then
    echo "[merge] ${label} skip: no prompt_samples yet"
    return 0
  fi

  mapfile -t args < <(common_args "$engine" "$exp_out" "${exp_out}/cache_merge" "$num_shards")
  "$PY" "${args[@]}" --merge_only --summary_log "$summary_log"
}

stop_children() {
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

merge_all() {
  merge_exp "qwen8b" "qwen3-VL-8B" "${OUT_ROOT}/mmstar_qwen8b_pure_mllm_gpu${GPU_ID}_1shard_${RUN_BATCH_ID}" 1
  merge_exp "qwen4b" "qwen3-VL-4B" "${OUT_ROOT}/mmstar_qwen4b_pure_mllm_gpu${GPU_ID}_2shards_${RUN_BATCH_ID}" 2
}

on_term() {
  echo "[signal] received TERM/INT, stopping children and merging partial results"
  stop_children
  wait || true
  merge_all || true
  exit 143
}
trap on_term TERM INT

run_shard() {
  local label="$1"
  local engine="$2"
  local num_shards="$3"
  local shard_id="$4"
  local exp_out="$5"
  local cache_path="${exp_out}/cache_shard${shard_id}"
  local log_file="${LOG_ROOT}/${label}_${RUN_BATCH_ID}_gpu${GPU_ID}_shard${shard_id}.log"

  mkdir -p "$exp_out"
  mapfile -t args < <(common_args "$engine" "$exp_out" "$cache_path" "$num_shards")
  echo "[launch] ${label} gpu=${GPU_ID} shard=${shard_id}/${num_shards} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PY" "${args[@]}" --shard_id "$shard_id" > "$log_file" 2>&1 &
  CHILDREN+=("$!")
}

EXP_8B="${OUT_ROOT}/mmstar_qwen8b_pure_mllm_gpu${GPU_ID}_1shard_${RUN_BATCH_ID}"
EXP_4B="${OUT_ROOT}/mmstar_qwen4b_pure_mllm_gpu${GPU_ID}_2shards_${RUN_BATCH_ID}"

echo "[run] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[run] GPU_ID=${GPU_ID}"
echo "[run] 8B out=${EXP_8B} shards=1"
echo "[run] 4B out=${EXP_4B} shards=2"
echo "[run] pure MLLM flags: --context_mode empty --remove_caption --n_shot 0"

run_shard "mmstar_qwen8b_pure" "qwen3-VL-8B" 1 0 "$EXP_8B"
sleep "$SHARD_LAUNCH_DELAY"
run_shard "mmstar_qwen4b_pure" "qwen3-VL-4B" 2 0 "$EXP_4B"
sleep "$SHARD_LAUNCH_DELAY"
run_shard "mmstar_qwen4b_pure" "qwen3-VL-4B" 2 1 "$EXP_4B"

status=0
for pid in "${CHILDREN[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

merge_all
echo "[done] 8B summary=${EXP_8B}/summary.log"
echo "[done] 4B summary=${EXP_4B}/summary.log"
exit "$status"
