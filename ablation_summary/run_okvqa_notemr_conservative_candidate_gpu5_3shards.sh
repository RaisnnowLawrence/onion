#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_notemr_conservative_candidate_gpu5_3shards_$(date +%Y%m%d_%H%M%S)}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/okvqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_notemr_conservative"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
KNOWLEDGE_CACHE="/data2/lizhengxue/WorkSpace/onion_output/okvqa/preflmr_gs112k_cache_val_train_clean/preflmr_gs112k_retriever.json"

mkdir -p "$LOG_ROOT" "$EXP_OUT"

run_shard() {
  local shard_id="$1"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu5_shard${shard_id}.log"
  echo "[launch] gpu=5 shard=${shard_id} log=${log_file}"
  CUDA_VISIBLE_DEVICES=5 "$PY" "$REPO/forward_code/onion.py" \
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
    --chain_of_thoughts \
    --cot_style notemr_conservative_candidate \
    --notemr_candidate_trigger knowledge_qtype_or_weak \
    --knowledge_notes_mode notes \
    --knowledge_cache_file "$KNOWLEDGE_CACHE" \
    --knowledge_cache_only \
    --knowledge_notes_use_image \
    --knowledge_top_k 5 \
    --knowledge_notes_max_words 60 \
    --knowledge_notes_max_tokens 96 \
    --knowledge_notes_max_chars 600 \
    --knowledge_raw_max_chars 1000 \
    --notemr_relevance_max_tokens 48 \
    --notemr_candidate_max_tokens 16 \
    --notemr_judge_max_tokens 80 \
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
echo "[run] card5: shard0, shard1, shard2"

run_shard 0 &
sleep "${SHARD_LAUNCH_DELAY:-90}"
run_shard 1 &
sleep "${SHARD_LAUNCH_DELAY:-90}"
run_shard 2 &

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
