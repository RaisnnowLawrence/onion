#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_preflmr_vitl_notes_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-5}"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
POLL_SECONDS="${POLL_SECONDS:-120}"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/okvqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_preflmr_vitl"

PREFLMR_CKPT="/data2/lizhengxue/WorkSpace/PreTrainModel/PreFLMR/PreFLMR_ViT-L"
IMAGE_PROCESSOR="/data2/lizhengxue/WorkSpace/PreTrainModel/OpenAI/clip-vit-large-patch14"
GS112K="/data2/lizhengxue/datasets/gs112k/okvqa_train_clean_corpus.csv"
QUESTION_FILE="/data2/lizhengxue/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
COCO_ROOT="/data2/lizhengxue/datasets/coco14"

RETRIEVAL_DIR="${OUT_ROOT}/preflmr_vitl_gs112k_cache_val_train_clean"
RETRIEVAL_JSON="${RETRIEVAL_DIR}/preflmr_vitl_gs112k_retriever.json"
RETRIEVAL_JSONL="${RETRIEVAL_DIR}/preflmr_vitl_gs112k_knowledge_cache.jsonl"
NOTES_JSONL="${RETRIEVAL_DIR}/preflmr_vitl_gs112k_knowledge_notes.jsonl"

mkdir -p "$LOG_ROOT" "$RETRIEVAL_DIR"

export HF_HOME="/data2/lizhengxue/WorkSpace/.cache/huggingface"
export PYTHONPATH="/data2/lizhengxue/WorkSpace/opensource/FLMR-main:/data2/lizhengxue/WorkSpace/opensource/FLMR-main/third_party/ColBERT:${PYTHONPATH:-}"

wait_for_gpu() {
  while true; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d '[:space:]')"
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d '[:space:]')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d '[:space:]')"
    echo "[vitl_pipeline] gpu=${GPU_ID} used=${used_mb}MB free=${free_mb}MB util=${util}% need_free>=${MIN_FREE_MB}MB"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      break
    fi
    sleep "$POLL_SECONDS"
  done
}

echo "[vitl_pipeline] RUN_ID=${RUN_ID}"
echo "[vitl_pipeline] checkpoint=${PREFLMR_CKPT}"
echo "[vitl_pipeline] image_processor=${IMAGE_PROCESSOR}"
echo "[vitl_pipeline] corpus=${GS112K}"
echo "[vitl_pipeline] output_dir=${RETRIEVAL_DIR}"

if [[ ! -d "$PREFLMR_CKPT" ]]; then
  echo "[vitl_pipeline] missing checkpoint: $PREFLMR_CKPT" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_PROCESSOR" ]]; then
  echo "[vitl_pipeline] missing image processor: $IMAGE_PROCESSOR" >&2
  exit 1
fi

wait_for_gpu

if [[ ! -s "$RETRIEVAL_JSON" ]]; then
  echo "[vitl_pipeline] extracting PreFLMR ViT-L knowledge..."
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
    --preflmr_checkpoint "$PREFLMR_CKPT" \
    --image_processor "$IMAGE_PROCESSOR" \
    --device cuda \
    --gs112k_corpus "$GS112K" \
    --question_file "$QUESTION_FILE" \
    --coco_root "$COCO_ROOT" \
    --top_k 10 \
    --index_experiment okvqa_val_gs112k_train_clean_vitl_gpu5 \
    --index_name preflmr_vitl_gs112k_train_clean \
    --note_mr_output_name "$(basename "$RETRIEVAL_JSON")" \
    --onion_output_name "$(basename "$RETRIEVAL_JSONL")" \
    --output_dir "$RETRIEVAL_DIR"
else
  echo "[vitl_pipeline] retrieval output exists, skip extraction: $RETRIEVAL_JSON"
fi

wait_for_gpu

echo "[vitl_pipeline] generating/resuming ViT-L Knowledge Notes..."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/generate_knowledge_notes_from_retrieval.py" \
  --retrieval_file "$RETRIEVAL_JSON" \
  --output_file "$NOTES_JSONL" \
  --engine qwen3-VL-4B \
  --top_k 10 \
  --resume

echo "[vitl_pipeline] done"
echo "[vitl_pipeline] retrieval_json=${RETRIEVAL_JSON}"
echo "[vitl_pipeline] retrieval_jsonl=${RETRIEVAL_JSONL}"
echo "[vitl_pipeline] notes_jsonl=${NOTES_JSONL}"
