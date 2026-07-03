#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_pure_mllm_$(date +%Y%m%d_%H%M%S)}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/okvqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_pure_mllm"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"

mkdir -p "$LOG_ROOT" "$EXP_OUT"

run_shard() {
  local gpu_id="$1"
  local shard_id="$2"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${gpu_id}_shard${shard_id}.log"
  echo "[launch] gpu=${gpu_id} shard=${shard_id} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name okvqa \
    --split_name val \
    --engine qwen3-VL-4B \
    --caption_type vinvl \
    --n_shot 1 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --context_mode empty \
    --remove_caption \
    --raw_image_dir /data2/lizhengxue/datasets/coco14 \
    --coco_path /data2/lizhengxue/datasets/okvqa \
    --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations \
    --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa \
    --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text \
    --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags \
    --train_sim_metric answer \
    --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk \
    --output_path "$EXP_OUT" \
    --cache_path "${EXP_OUT}/cache" \
    --num_shards 3 \
    --shard_id "$shard_id" \
    > "$log_file" 2>&1
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] card0: shard0; card4: shard1,shard2"

run_shard 0 0 &
sleep "${SHARD_LAUNCH_DELAY:-90}"
run_shard 4 1 &
sleep "${SHARD_LAUNCH_DELAY:-90}"
run_shard 4 2 &

wait

echo "[merge] merging ${EXP_OUT}"
"$PY" "$REPO/forward_code/onion.py" \
  --dataset_name okvqa \
  --split_name val \
  --caption_type vinvl \
  --output_path "$EXP_OUT" \
  --coco_path /data2/lizhengxue/datasets/okvqa \
  --raw_image_dir /data2/lizhengxue/datasets/coco14 \
  --merge_only \
  --summary_log "$SUMMARY_LOG"

echo "[done] summary=${SUMMARY_LOG}"
