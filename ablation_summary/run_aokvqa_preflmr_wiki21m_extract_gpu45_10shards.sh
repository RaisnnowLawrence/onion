#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-aokvqa_preflmr_wiki21m_extract_gpu45_10shards_$(date +%Y%m%d_%H%M%S)}"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/aokvqa"
INDEX_ROOT="/data2/lizhengxue/WorkSpace/onion_output/preflmr_indexes"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_aokvqa_preflmr_wiki21m"

WIKI21M="/data2/lizhengxue/datasets/wiki21m/psgs_w100.tsv"
AOKVQA_VAL="/data2/lizhengxue/datasets/aokvqa/aokvqa_v1p0_val.json"
COCO_ROOT="/data2/lizhengxue/datasets/coco17"
NUM_SHARDS="${NUM_SHARDS:-10}"
QUERY_PARALLEL="${QUERY_PARALLEL:-10}"
TOP_K="${TOP_K:-10}"

mkdir -p "$LOG_ROOT"

export HF_HOME="/data2/lizhengxue/WorkSpace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_EXTENSIONS_DIR="/data2/lizhengxue/WorkSpace/.cache/torch_extensions"
export PYTHONPATH="/data2/lizhengxue/WorkSpace/opensource/FLMR-main:/data2/lizhengxue/WorkSpace/opensource/FLMR-main/third_party/ColBERT:${PYTHONPATH:-}"

wait_for_gpu() {
  local gpu_id="$1"
  local min_free_mb="$2"
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | tr -d '[:space:]')"
    local used_mb
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" | tr -d '[:space:]')"
    echo "[$(date '+%F %T')] gpu=${gpu_id} used=${used_mb}MB free=${free_mb}MB need_free>=${min_free_mb}MB"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= min_free_mb )); then
      break
    fi
    sleep 120
  done
}

merge_outputs() {
  local model_tag="$1"
  local out_dir="$2"
  "$PY" - "$model_tag" "$out_dir" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

model_tag = sys.argv[1]
out_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])

records = []
jsonl_records = []
for shard_id in range(num_shards):
    shard_dir = out_dir / "shards" / f"shard{shard_id}"
    retriever = shard_dir / f"preflmr_{model_tag}_wiki21m_retriever_shard{shard_id}.json"
    cache = shard_dir / f"preflmr_{model_tag}_wiki21m_knowledge_cache_shard{shard_id}.jsonl"
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

(out_dir / f"preflmr_{model_tag}_wiki21m_retriever.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (out_dir / f"preflmr_{model_tag}_wiki21m_knowledge_cache.jsonl").open("w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

manifest = {
    "model_tag": model_tag,
    "num_shards": num_shards,
    "num_records_json": len(records),
    "num_records_jsonl": len(jsonl_records),
}
(out_dir / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PY
}

run_model() {
  local model_tag="$1"
  local gpu_id="$2"
  local checkpoint="$3"
  local image_processor="$4"
  local min_free_mb="$5"

  local out_dir="${OUT_ROOT}/preflmr_${model_tag}_wiki21m_cache_val_${RUN_ID}"
  local shard_root="${out_dir}/shards"
  local index_experiment="aokvqa_val_wiki21m_${model_tag}"
  local index_name="preflmr_${model_tag}_wiki21m"
  local offsets_file="${OUT_ROOT}/wiki21m_psgs_w100.offsets"

  mkdir -p "$out_dir" "$shard_root"

  echo "[$(date '+%F %T')] start model=${model_tag} gpu=${gpu_id} out=${out_dir}"
  wait_for_gpu "$gpu_id" "$min_free_mb"

  local index_log="${LOG_ROOT}/${RUN_ID}_${model_tag}_index_gpu${gpu_id}.log"
  if [[ ! -d "${INDEX_ROOT}/${index_experiment}/indexes/${index_name}.nbits=8" ]]; then
    echo "[$(date '+%F %T')] build index model=${model_tag} log=${index_log}"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
      --preflmr_checkpoint "$checkpoint" \
      --image_processor "$image_processor" \
      --device cuda \
      --dataset_name aokvqa \
      --question_file "$AOKVQA_VAL" \
      --coco_root "$COCO_ROOT" \
      --corpus_format wiki21m \
      --corpus_file "$WIKI21M" \
      --knowledge_source_name "preflmr_${model_tag}_wiki21m" \
      --top_k "$TOP_K" \
      --index_root "$INDEX_ROOT" \
      --index_experiment "$index_experiment" \
      --index_name "$index_name" \
      --output_dir "${out_dir}/index_build" \
      --build_index_only \
      > "$index_log" 2>&1
  else
    echo "[$(date '+%F %T')] index exists, skip model=${model_tag}: ${INDEX_ROOT}/${index_experiment}/indexes/${index_name}.nbits=8"
  fi

  echo "[$(date '+%F %T')] launch query shards model=${model_tag} gpu=${gpu_id} shards=${NUM_SHARDS} parallel=${QUERY_PARALLEL}"
  local running=0
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_dir="${shard_root}/shard${shard_id}"
    mkdir -p "$shard_dir"
    local shard_log="${LOG_ROOT}/${RUN_ID}_${model_tag}_shard${shard_id}_gpu${gpu_id}.log"
    (
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" "$REPO/forward_code/build_preflmr_gs112k_cache.py" \
        --preflmr_checkpoint "$checkpoint" \
        --image_processor "$image_processor" \
        --device cuda \
        --dataset_name aokvqa \
        --question_file "$AOKVQA_VAL" \
        --coco_root "$COCO_ROOT" \
        --corpus_format wiki21m \
        --corpus_file "$WIKI21M" \
        --knowledge_source_name "preflmr_${model_tag}_wiki21m" \
        --top_k "$TOP_K" \
        --index_root "$INDEX_ROOT" \
        --index_experiment "$index_experiment" \
        --index_name "$index_name" \
        --output_dir "$shard_dir" \
        --note_mr_output_name "preflmr_${model_tag}_wiki21m_retriever_shard${shard_id}.json" \
        --onion_output_name "preflmr_${model_tag}_wiki21m_knowledge_cache_shard${shard_id}.jsonl" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$shard_id" \
        --skip_index_build \
        --use_lazy_wiki_lookup \
        --wiki_offsets_file "$offsets_file" \
        > "$shard_log" 2>&1
    ) &
    running=$((running + 1))
    if (( running >= QUERY_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
    sleep 5
  done
  wait

  merge_outputs "$model_tag" "$out_dir"
  echo "[$(date '+%F %T')] done model=${model_tag} out=${out_dir}"
}

echo "RUN_ID=${RUN_ID}"
echo "LOG_ROOT=${LOG_ROOT}"

run_model "vitb" "4" \
  "/data2/lizhengxue/WorkSpace/PreTrainModel/PreFLMR/PreFLMR_ViT-B" \
  "/data2/lizhengxue/WorkSpace/PreTrainModel/OpenAI/clip-vit-base-patch32" \
  "${MIN_FREE_MB_VITB:-24000}" &

run_model "vitl" "5" \
  "/data2/lizhengxue/WorkSpace/PreTrainModel/PreFLMR/PreFLMR_ViT-L" \
  "/data2/lizhengxue/WorkSpace/PreTrainModel/OpenAI/clip-vit-large-patch14" \
  "${MIN_FREE_MB_VITL:-32000}" &

wait
echo "[$(date '+%F %T')] all done RUN_ID=${RUN_ID}"
