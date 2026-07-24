#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_clip_statement_full4_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_clip_statement_mmfocus_pad12_full_12shards_${RUN_BATCH_ID}}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
CONTROLLER="/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_gqa_testdev_dyfo_clip_statement_wait_four_gpus.sh"

export TS_SOCKET
"$TS_BIN" -S 1
job_id=$(
  "$TS_BIN" -L clip_statement_full4 env \
    CANDIDATE_GPUS="0 1 2 3 4 5 6 7" \
    MIN_FREE_MB=40960 GPU_WAIT_SECONDS=60 SHARD_LAUNCH_DELAY=180 \
    bash "$CONTROLLER" "$RUN_ID"
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] JOB_ID=${job_id}"
echo "[queue] waits for four GPUs among 0-7 with at least 40960 MiB free each"
"$TS_BIN" -l
