#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-vqav2_qwen8b_pure_mllm_gpu2_4shards_$(date +%Y%m%d_%H%M%S)}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/vqav2"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_vqav2_qwen8b_pure_mllm"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
GPU_ID="${GPU_ID:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CURRENT_CHILD=""

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
    --engine qwen3-VL-8B \
    --caption_type vinvl \
    --n_shot 0 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode empty \
    --remove_caption \
    --output_path "$EXP_OUT" \
    --coco_path /data2/lizhengxue/datasets/vqav2 \
    --raw_image_dir /data2/lizhengxue/datasets/coco14 \
    --merge_only \
    --summary_log "$SUMMARY_LOG"
}

on_term() {
  echo "[signal] received TERM/INT, stopping current shard and merging partial results"
  if [[ -n "${CURRENT_CHILD}" ]] && kill -0 "${CURRENT_CHILD}" 2>/dev/null; then
    kill -TERM "${CURRENT_CHILD}" 2>/dev/null || true
    wait "${CURRENT_CHILD}" 2>/dev/null || true
  fi
  merge_partial || true
  exit 143
}
trap on_term TERM INT

run_shard() {
  local shard_id="$1"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${GPU_ID}_shard${shard_id}.log"
  echo "[launch] gpu=${GPU_ID} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name vqav2 \
    --split_name val \
    --engine qwen3-VL-8B \
    --caption_type vinvl \
    --n_shot 0 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode empty \
    --remove_caption \
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
  CURRENT_CHILD="$!"
  wait "$CURRENT_CHILD"
  CURRENT_CHILD=""
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] GPU_ID=${GPU_ID} NUM_SHARDS=${NUM_SHARDS}"
echo "[run] mode=sequential shards; partial merge on TERM/INT"

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "$shard_id"
done

merge_partial
echo "[done] summary=${SUMMARY_LOG}"
