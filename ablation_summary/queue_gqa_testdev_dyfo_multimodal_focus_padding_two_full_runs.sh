#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_mmfocus_padding_ablation_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
CONTROLLER="/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_gqa_testdev_dyfo_token_confidence_wait_two_gpus.sh"
RUN_ID_MM12="${RUN_ID_MM12:-gqa_testdev_dyfo_token_conf_mmfocus_pad12_full_6shards_${RUN_BATCH_ID}}"
RUN_ID_MM14="${RUN_ID_MM14:-gqa_testdev_dyfo_token_conf_mmfocus_pad14_full_6shards_${RUN_BATCH_ID}}"

export TS_SOCKET
"$TS_BIN" -S 1

COMMON_ENV=(
  CANDIDATE_GPUS="0 1 2 3 4 5 6 7"
  MIN_FREE_MB=40960
  GPU_WAIT_SECONDS=60
  SHARD_LAUNCH_DELAY=180
  DYFO_TEXT_FOCUS_USE_IMAGE=1
)

job_mm12=$(
  "$TS_BIN" -L mmfocus_pad12_full env \
    "${COMMON_ENV[@]}" DYFO_FOCUS_PADDING=1.2 \
    bash "$CONTROLLER" "$RUN_ID_MM12"
)

job_mm14=$(
  "$TS_BIN" -D "$job_mm12" -L mmfocus_pad14_full env \
    "${COMMON_ENV[@]}" DYFO_FOCUS_PADDING=1.4 \
    bash "$CONTROLLER" "$RUN_ID_MM14"
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] experiment 1: job=${job_mm12} run_id=${RUN_ID_MM12} mmfocus=1 padding=1.2"
echo "[queue] experiment 2: job=${job_mm14} run_id=${RUN_ID_MM14} mmfocus=1 padding=1.4"
echo "[queue] both runs use full GQA testdev, 2 GPUs, 3 shards per GPU"
"$TS_BIN" -l
