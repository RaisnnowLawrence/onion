#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-aokvqa_method4_wiki_subcorpus_flmrb_gpu5_$(date +%Y%m%d_%H%M%S)}"

GPU_ID="${GPU_ID:-5}"
NUM_SHARDS="${NUM_SHARDS:-10}"
QUERY_PARALLEL="${QUERY_PARALLEL:-2}"
TOP_K="${TOP_K:-10}"
TOP_K_PER_QUERY="${TOP_K_PER_QUERY:-50}"
MAX_SUBCORPUS="${MAX_SUBCORPUS:-300000}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
QUERY_SCOPE="${QUERY_SCOPE:-val}"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/aokvqa"
INDEX_ROOT="/data2/lizhengxue/WorkSpace/onion_output/preflmr_indexes"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_aokvqa_method4_wiki_subcorpus"

WIKI21M="/data2/lizhengxue/datasets/wiki21m/psgs_w100.tsv"
AOKVQA_TRAIN="/data2/lizhengxue/datasets/aokvqa/aokvqa_v1p0_train.json"
AOKVQA_VAL="/data2/lizhengxue/datasets/aokvqa/aokvqa_v1p0_val.json"
COCO_ROOT="/data2/lizhengxue/datasets/coco17"

PREFLMR_CKPT="/data2/lizhengxue/WorkSpace/PreTrainModel/PreFLMR/PreFLMR_ViT-B"
IMAGE_PROCESSOR="/data2/lizhengxue/WorkSpace/PreTrainModel/OpenAI/clip-vit-base-patch32"

EXP_ROOT="${OUT_ROOT}/method4_wiki_${QUERY_SCOPE}_sub${MAX_SUBCORPUS}_flmrb_${RUN_ID}"
SUBCORPUS_CSV="${EXP_ROOT}/wiki21m_aokvqa_${QUERY_SCOPE}_sub${MAX_SUBCORPUS}.csv"
SUBCORPUS_MANIFEST="${EXP_ROOT}/wiki21m_aokvqa_${QUERY_SCOPE}_sub${MAX_SUBCORPUS}_manifest.json"
RETRIEVAL_DIR="${EXP_ROOT}/retrieval"
SHARD_ROOT="${RETRIEVAL_DIR}/shards"
INDEX_EXPERIMENT="aokvqa_method4_wiki_sub${MAX_SUBCORPUS}_flmrb_${RUN_ID}"
INDEX_NAME="preflmr_vitb_wiki21m_sub${MAX_SUBCORPUS}"

mkdir -p "$EXP_ROOT" "$RETRIEVAL_DIR" "$SHARD_ROOT" "$LOG_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="/data2/lizhengxue/WorkSpace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_EXTENSIONS_DIR="/data2/lizhengxue/WorkSpace/.cache/torch_extensions"
export PYTHONPATH="/data2/lizhengxue/WorkSpace/opensource/FLMR-main:/data2/lizhengxue/WorkSpace/opensource/FLMR-main/third_party/ColBERT:${PYTHONPATH:-}"

wait_for_gpu() {
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d '[:space:]')"
    local used_mb
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d '[:space:]')"
    echo "[$(date '+%F %T')] gpu=${GPU_ID} used=${used_mb}MB free=${free_mb}MB need_free>=${MIN_FREE_MB}MB"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      break
    fi
    sleep 120
  done
}

merge_outputs() {
  "$PY" - "$RETRIEVAL_DIR" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_shards = int(sys.argv[2])
records = []
jsonl_records = []
for shard_id in range(num_shards):
    shard_dir = root / "shards" / f"shard{shard_id}"
    retriever = shard_dir / f"method4_wiki_sub_flmrb_retriever_shard{shard_id}.json"
    cache = shard_dir / f"method4_wiki_sub_flmrb_knowledge_cache_shard{shard_id}.jsonl"
    if not retriever.exists():
        raise FileNotFoundError(retriever)
    records.extend(json.loads(retriever.read_text(encoding="utf-8")))
    if not cache.exists():
        raise FileNotFoundError(cache)
    with cache.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jsonl_records.append(json.loads(line))

(root / "method4_wiki_sub_flmrb_retriever.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (root / "method4_wiki_sub_flmrb_knowledge_cache.jsonl").open("w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

manifest = {
    "num_shards": num_shards,
    "num_records_json": len(records),
    "num_records_jsonl": len(jsonl_records),
    "retriever_json": str(root / "method4_wiki_sub_flmrb_retriever.json"),
    "knowledge_cache_jsonl": str(root / "method4_wiki_sub_flmrb_knowledge_cache.jsonl"),
}
(root / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PY
}

echo "RUN_ID=${RUN_ID}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "SUBCORPUS_CSV=${SUBCORPUS_CSV}"
echo "RETRIEVAL_DIR=${RETRIEVAL_DIR}"

if [[ ! -s "$SUBCORPUS_CSV" ]]; then
  echo "[$(date '+%F %T')] stage1: build wiki21m subcorpus"
  "$PY" "$REPO/forward_code/build_aokvqa_wiki21m_subcorpus.py" \
    --wiki21m_file "$WIKI21M" \
    --train_file "$AOKVQA_TRAIN" \
    --val_file "$AOKVQA_VAL" \
    --output_csv "$SUBCORPUS_CSV" \
    --manifest_file "$SUBCORPUS_MANIFEST" \
    --top_k_per_query "$TOP_K_PER_QUERY" \
    --max_subcorpus "$MAX_SUBCORPUS" \
    --query_scope "$QUERY_SCOPE" \
    > "${LOG_ROOT}/${RUN_ID}_stage1_subcorpus.log" 2>&1
else
  echo "[$(date '+%F %T')] stage1 skip: ${SUBCORPUS_CSV}"
fi

wait_for_gpu

if [[ ! -d "${INDEX_ROOT}/${INDEX_EXPERIMENT}/indexes/${INDEX_NAME}.nbits=8" ]] || [[ ! -s "${INDEX_ROOT}/${INDEX_EXPERIMENT}/indexes/${INDEX_NAME}.nbits=8/plan.json" ]]; then
  echo "[$(date '+%F %T')] stage2: build FLMR-B index on subcorpus"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
    --preflmr_checkpoint "$PREFLMR_CKPT" \
    --image_processor "$IMAGE_PROCESSOR" \
    --device cuda \
    --dataset_name aokvqa \
    --question_file "$AOKVQA_VAL" \
    --coco_root "$COCO_ROOT" \
    --corpus_format gs112k \
    --corpus_file "$SUBCORPUS_CSV" \
    --knowledge_source_name "preflmr_vitb_wiki21m_sub${MAX_SUBCORPUS}" \
    --top_k "$TOP_K" \
    --index_root "$INDEX_ROOT" \
    --index_experiment "$INDEX_EXPERIMENT" \
    --index_name "$INDEX_NAME" \
    --output_dir "${EXP_ROOT}/index_build" \
    --build_index_only \
    > "${LOG_ROOT}/${RUN_ID}_stage2_index_gpu${GPU_ID}.log" 2>&1
else
  echo "[$(date '+%F %T')] stage2 skip existing index"
fi

wait_for_gpu

echo "[$(date '+%F %T')] stage3: retrieve A-OKVQA val with ${NUM_SHARDS} shards, parallel=${QUERY_PARALLEL}"
running=0
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_dir="${SHARD_ROOT}/shard${shard_id}"
  mkdir -p "$shard_dir"
  (
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
      --preflmr_checkpoint "$PREFLMR_CKPT" \
      --image_processor "$IMAGE_PROCESSOR" \
      --device cuda \
      --dataset_name aokvqa \
      --question_file "$AOKVQA_VAL" \
      --coco_root "$COCO_ROOT" \
      --corpus_format gs112k \
      --corpus_file "$SUBCORPUS_CSV" \
      --knowledge_source_name "preflmr_vitb_wiki21m_sub${MAX_SUBCORPUS}" \
      --top_k "$TOP_K" \
      --index_root "$INDEX_ROOT" \
      --index_experiment "$INDEX_EXPERIMENT" \
      --index_name "$INDEX_NAME" \
      --output_dir "$shard_dir" \
      --note_mr_output_name "method4_wiki_sub_flmrb_retriever_shard${shard_id}.json" \
      --onion_output_name "method4_wiki_sub_flmrb_knowledge_cache_shard${shard_id}.jsonl" \
      --num_shards "$NUM_SHARDS" \
      --shard_id "$shard_id" \
      --skip_index_build \
      > "${LOG_ROOT}/${RUN_ID}_stage3_retrieve_shard${shard_id}_gpu${GPU_ID}.log" 2>&1
  ) &
  running=$((running + 1))
  if (( running >= QUERY_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
  sleep 5
done
wait

merge_outputs

echo "[$(date '+%F %T')] all done"
echo "retriever=${RETRIEVAL_DIR}/method4_wiki_sub_flmrb_retriever.json"
echo "knowledge_cache=${RETRIEVAL_DIR}/method4_wiki_sub_flmrb_knowledge_cache.jsonl"
