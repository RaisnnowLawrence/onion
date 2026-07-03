#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_preflmr_notes_knowledge_only_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-5}"
REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
SAM_ENV="/data2/lizhengxue/anaconda3/envs/sam"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/okvqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_preflmr_notes"
RETRIEVAL_DIR="${OUT_ROOT}/preflmr_gs112k_cache_val_train_clean"
RETRIEVAL_JSON="${RETRIEVAL_DIR}/preflmr_gs112k_retriever.json"
NOTES_JSONL="${RETRIEVAL_DIR}/preflmr_gs112k_knowledge_notes.jsonl"
NOTES_SHARD_DIR="${RETRIEVAL_DIR}/knowledge_notes_shards_${RUN_ID}"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"

mkdir -p "$LOG_ROOT" "$EXP_OUT"

export HF_HOME="/data2/lizhengxue/WorkSpace/.cache/huggingface"
export PYTHONPATH="/data2/lizhengxue/WorkSpace/opensource/FLMR-main:/data2/lizhengxue/WorkSpace/opensource/FLMR-main/third_party/ColBERT:${PYTHONPATH:-}"

echo "[pipeline] RUN_ID=${RUN_ID}"
echo "[pipeline] GPU_ID=${GPU_ID}"
echo "[pipeline] RETRIEVAL_JSON=${RETRIEVAL_JSON}"
echo "[pipeline] NOTES_JSONL=${NOTES_JSONL}"
echo "[pipeline] EXP_OUT=${EXP_OUT}"

if [[ ! -s "$RETRIEVAL_JSON" ]]; then
  echo "[pipeline] waiting for retrieval output..."
  while [[ ! -s "$RETRIEVAL_JSON" ]]; do
    if ! pgrep -f "build_preflmr_gs112k_cache.py.*preflmr_gs112k_cache_val_train_clean" >/dev/null; then
      echo "[pipeline] retrieval process is not running; start it now."
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
        --device cuda \
        --gs112k_corpus /data2/lizhengxue/datasets/gs112k/okvqa_train_clean_corpus.csv \
        --question_file /data2/lizhengxue/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json \
        --coco_root /data2/lizhengxue/datasets/coco14 \
        --top_k 10 \
        --index_experiment okvqa_val_gs112k_train_clean_gpu5 \
        --index_name preflmr_vitb_gs112k_train_clean \
        --output_dir "$RETRIEVAL_DIR"
    fi
    sleep 60
  done
fi

if [[ "$(wc -l < "$NOTES_JSONL" 2>/dev/null || echo 0)" -lt 5046 ]]; then
  echo "[pipeline] generating/resuming Knowledge Notes with 3 parallel shards..."
  mkdir -p "$NOTES_SHARD_DIR"
  for note_shard_id in 0 1 2; do
    note_log="${LOG_ROOT}/${RUN_ID}_notes_shard${note_shard_id}.log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/generate_knowledge_notes_from_retrieval.py" \
      --retrieval_file "$RETRIEVAL_JSON" \
      --output_file "${NOTES_SHARD_DIR}/notes_shard${note_shard_id}.jsonl" \
      --engine qwen3-VL-4B \
      --top_k 10 \
      --num_shards 3 \
      --shard_id "$note_shard_id" \
      --resume \
      > "$note_log" 2>&1 &
    echo "[pipeline] launched notes shard ${note_shard_id}, log=${note_log}"
    if [[ "$note_shard_id" != "2" ]]; then
      sleep "${NOTES_SHARD_LAUNCH_DELAY:-90}"
    fi
  done
  wait
  cat "${NOTES_SHARD_DIR}"/notes_shard*.jsonl > "${NOTES_JSONL}.tmp"
  mv "${NOTES_JSONL}.tmp" "$NOTES_JSONL"
else
  echo "[pipeline] Knowledge Notes already complete, skip generation."
fi

echo "[pipeline] launching OK-VQA val knowledge-only ablation with 3 shards..."
for shard_id in 0 1 2; do
  shard_log="${LOG_ROOT}/${RUN_ID}_shard${shard_id}.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/onion.py" \
    --dataset_name okvqa \
    --split_name val \
    --engine qwen3-VL-4B \
    --caption_type vinvl \
    --n_shot 1 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --use_knowledge_enhance \
    --knowledge_notes_mode notes \
    --knowledge_enhance_trigger always \
    --knowledge_cache_file "$NOTES_JSONL" \
    --knowledge_cache_only \
    --knowledge_top_k 10 \
    --knowledge_raw_max_chars 1800 \
    --knowledge_notes_max_chars 700 \
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
    > "$shard_log" 2>&1 &
  echo "[pipeline] launched shard ${shard_id}, log=${shard_log}"
  if [[ "$shard_id" != "2" ]]; then
    sleep "${SHARD_LAUNCH_DELAY:-90}"
  fi
done

wait

echo "[pipeline] merging results..."
"$PY" "$REPO/forward_code/onion.py" \
  --dataset_name okvqa \
  --split_name val \
  --caption_type vinvl \
  --output_path "$EXP_OUT" \
  --coco_path /data2/lizhengxue/datasets/okvqa \
  --raw_image_dir /data2/lizhengxue/datasets/coco14 \
  --merge_only \
  --summary_log "$SUMMARY_LOG"

echo "[pipeline] done. summary=${SUMMARY_LOG}"
