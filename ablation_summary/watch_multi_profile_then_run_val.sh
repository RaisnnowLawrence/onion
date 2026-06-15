#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
PROFILE_DIR=${PROFILE_DIR:-${REPORT_DIR}/strategy_rag_profiles}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_multi_strategy_router_val}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-400}
PROFILE_PATH=${PROFILE_PATH:-${PROFILE_DIR}/multi_combined_train_n${MAX_TRAIN_SAMPLES}.jsonl}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}

mkdir -p "${LOG_DIR}"

echo "[watch-multi] waiting for ${PROFILE_PATH}"
while [[ ! -s "${PROFILE_PATH}" ]]; do
  sleep "${CHECK_INTERVAL}"
done

echo "[watch-multi] profile found; launching multi-strategy val router"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES}" PROFILE_PATH="${PROFILE_PATH}" \
  "${REPO}/ablation_summary/run_multi_strategy_router_val_3shards.sh"

