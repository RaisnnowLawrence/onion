#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/data2/lizhengxue/WorkSpace/onion}
OUT=${OUT:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
RUN="${BASE}/ablation_summary/run_train_guided_complex_router_4exp_3shards.sh"
RUN_ID=${PROFILE_RUN_ID:-trainrag10k_$(date +%Y%m%d_%H%M%S)}

mkdir -p "${OUT}"

LOG="${OUT}/train_guided_complex_router_4exp_3shards_${RUN_ID}.master.out"

nohup env \
  PROFILE_RUN_ID="${RUN_ID}" \
  TRAIN_PROFILE_TOTAL=10000 \
  NUM_SHARDS=3 \
  "${RUN}" all \
  > "${LOG}" 2>&1 &

PID=$!
echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${LOG}"
