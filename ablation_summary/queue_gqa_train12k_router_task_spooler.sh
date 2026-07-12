#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
TS_BIN="/data2/lizhengxue/.local/bin/ts"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_train12k_router_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
SCRIPT="${REPO}/ablation_summary/run_wait_free_gpu_single_exp.sh"
SUBSET_DIR="${REPO}/ablation_summary/router_data"
SUBSET_JSON="${SUBSET_DIR}/gqa_train_balanced_seed42_12000_questions.json"
SUBSET_IDS="${SUBSET_DIR}/gqa_train_balanced_seed42_12000_ids.tsv"
ROUTER_DIR="${SUBSET_DIR}/${RUN_BATCH_ID}"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-5 7 6 2 3 0 1 4}"
TS_CONCURRENCY="${TS_CONCURRENCY:-2}"
PURE_MIN_FREE_MB="${PURE_MIN_FREE_MB:-34000}"
DYFO_MIN_FREE_MB="${DYFO_MIN_FREE_MB:-30000}"

export TS_SOCKET

"$TS_BIN" -S "$TS_CONCURRENCY" >/dev/null
mkdir -p "$SUBSET_DIR" "$ROUTER_DIR"

echo "RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "TS_SOCKET=${TS_SOCKET}"
echo "CANDIDATE_GPUS=${CANDIDATE_GPUS}"
echo "TS_CONCURRENCY=${TS_CONCURRENCY}"
echo "SUBSET_JSON=${SUBSET_JSON}"

if [[ ! -f "$SUBSET_JSON" ]]; then
  "$PY" "${REPO}/ablation_summary/sample_gqa_train12k.py" \
    --seed 42 \
    --count 12000 \
    --output "$SUBSET_JSON" \
    --ids-output "$SUBSET_IDS"
else
  echo "[subset] exists, reuse ${SUBSET_JSON}"
fi

PURE_LABEL="gqa_train12k_seed42_pure4b_3shards"
DYFO_LABEL="gqa_train12k_seed42_dyfo4b_weighted_vote_3shards"
PURE_RUN_ID="${PURE_LABEL}_${RUN_BATCH_ID}"
DYFO_RUN_ID="${DYFO_LABEL}_${RUN_BATCH_ID}"
PURE_OUT="/data2/lizhengxue/WorkSpace/onion_output/gqa/${PURE_RUN_ID}"
DYFO_OUT="/data2/lizhengxue/WorkSpace/onion_output/gqa/${DYFO_RUN_ID}"

pure_job="$("$TS_BIN" -L "$PURE_LABEL" env \
  CANDIDATE_GPUS="$CANDIDATE_GPUS" \
  SPLIT_NAME=train \
  GQA_QUESTION_FILE="$SUBSET_JSON" \
  SHARD_LAUNCH_DELAY=90 \
  bash "$SCRIPT" "$PURE_RUN_ID" gqa qwen3-VL-4B pure 3 "$PURE_MIN_FREE_MB")"
echo "${PURE_LABEL} job_id=${pure_job} run_id=${PURE_RUN_ID}"

dyfo_job="$("$TS_BIN" -L "$DYFO_LABEL" env \
  CANDIDATE_GPUS="$CANDIDATE_GPUS" \
  SPLIT_NAME=train \
  GQA_QUESTION_FILE="$SUBSET_JSON" \
  DYFO_DECISION_MODE=weighted_vote \
  SHARD_LAUNCH_DELAY=150 \
  bash "$SCRIPT" "$DYFO_RUN_ID" gqa qwen3-VL-4B dyfo 3 "$DYFO_MIN_FREE_MB")"
echo "${DYFO_LABEL} job_id=${dyfo_job} run_id=${DYFO_RUN_ID}"

router_job="$("$TS_BIN" -W "${pure_job},${dyfo_job}" -L gqa_train12k_router_table env \
  PURE_OUT="$PURE_OUT" \
  DYFO_OUT="$DYFO_OUT" \
  ROUTER_DIR="$ROUTER_DIR" \
  "$PY" "${REPO}/ablation_summary/build_gqa_router_training_table.py" \
    --pure-dir "$PURE_OUT" \
    --dyfo-dir "$DYFO_OUT" \
    --out-csv "${ROUTER_DIR}/gqa_train12k_router_training_table.csv" \
    --out-jsonl "${ROUTER_DIR}/gqa_train12k_router_training_table.jsonl" \
    --out-md "${ROUTER_DIR}/gqa_train12k_router_training_summary.md")"
echo "gqa_train12k_router_table job_id=${router_job} out_dir=${ROUTER_DIR}"

"$TS_BIN" -l
