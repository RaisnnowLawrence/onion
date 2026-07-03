#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-vqav2_pure_mllm_gpu3_3shards_$(date +%Y%m%d_%H%M%S)}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/vqav2"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_vqav2_pure_mllm"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
GPU_ID="${GPU_ID:-3}"
NUM_SHARDS="${NUM_SHARDS:-3}"

mkdir -p "$LOG_ROOT" "$EXP_OUT"

run_shard() {
  local shard_id="$1"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${GPU_ID}_shard${shard_id}.log"
  echo "[launch] gpu=${GPU_ID} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name vqav2 \
    --split_name val \
    --engine qwen3-VL-4B \
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
    > "$log_file" 2>&1
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] GPU_ID=${GPU_ID} NUM_SHARDS=${NUM_SHARDS}"

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "$shard_id" &
  if [[ "$shard_id" -lt $((NUM_SHARDS - 1)) ]]; then
    sleep "${SHARD_LAUNCH_DELAY:-90}"
  fi
done

wait

echo "[merge] merging ${EXP_OUT}"
"$PY" "$REPO/forward_code/onion.py" \
  --dataset_name vqav2 \
  --split_name val \
  --caption_type vinvl \
  --output_path "$EXP_OUT" \
  --coco_path /data2/lizhengxue/datasets/vqav2 \
  --raw_image_dir /data2/lizhengxue/datasets/coco14 \
  --merge_only \
  --summary_log "$SUMMARY_LOG"

echo "[done] summary=${SUMMARY_LOG}"
