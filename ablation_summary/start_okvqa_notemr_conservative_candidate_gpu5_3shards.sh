#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_notemr_conservative_candidate_gpu5_3shards_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_notemr_conservative"
CTRL_LOG="${LOG_ROOT}/${RUN_ID}_controller.log"

mkdir -p "$LOG_ROOT"
cd /data2/lizhengxue/WorkSpace/onion

setsid env SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-90}" \
  bash /data2/lizhengxue/WorkSpace/onion/ablation_summary/run_okvqa_notemr_conservative_candidate_gpu5_3shards.sh "$RUN_ID" \
  > "$CTRL_LOG" 2>&1 < /dev/null &

echo "RUN_ID=${RUN_ID}"
echo "PID=$!"
echo "CTRL_LOG=${CTRL_LOG}"
echo "EXP_OUT=/data2/lizhengxue/WorkSpace/onion_output/okvqa/${RUN_ID}"
