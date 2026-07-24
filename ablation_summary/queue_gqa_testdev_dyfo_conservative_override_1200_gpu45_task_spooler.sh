#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_conservative_override_1200_gpu45_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_conservative_override_6shards_1200_${RUN_BATCH_ID}}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${RUN_ID}_smoke}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"
WAIT_MERGE_SCRIPT="${SCRIPT_DIR}/wait_for_task_then_merge_gqa_testdev_dyfo.sh"

export TS_SOCKET
"$TS_BIN" -S 2

COMMON_ENV=(
  NUM_SHARDS=6
  SPLIT_NAME=testdev
  DYFO_DECISION_MODE=conservative_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_OVERRIDE_CONFIDENCE_THRESHOLD=95
  DYFO_OVERRIDE_REQUIRED_STRENGTH=extreme
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] policy=pure baseline -> DyFo weighted candidate -> extreme-confidence MLLM override"
echo "[queue] target=1200; 6 shards x 200 samples"
echo "[queue] gpu4 shards=0,2,4; gpu5 shards=1,3,5"

smoke_job=$(
  "$TS_BIN" -L smoke_gpu4 env \
    NUM_SHARDS=1 SHARD_IDS="0" MAX_SAMPLES_PER_SHARD=1 \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=0 \
    DYFO_DECISION_MODE=conservative_override \
    DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    DYFO_TRIGGER_MODE=always DYFO_FORCE_RUN_ALL_SAMPLES=1 \
    DYFO_OVERRIDE_CONFIDENCE_THRESHOLD=95 \
    DYFO_OVERRIDE_REQUIRED_STRENGTH=extreme \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 4 "$SMOKE_RUN_ID"
)

job4=$(
  "$TS_BIN" -D "$smoke_job" -L gpu4_shards0_2_4 env \
    "${COMMON_ENV[@]}" SHARD_IDS="0 2 4" MAX_SAMPLES_PER_SHARD=200 \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 \
    bash "$RUN_SCRIPT" 4 "$RUN_ID"
)

job5=$(
  "$TS_BIN" -D "$smoke_job" -L gpu5_shards1_3_5 env \
    "${COMMON_ENV[@]}" SHARD_IDS="1 3 5" MAX_SAMPLES_PER_SHARD=200 \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 \
    bash "$RUN_SCRIPT" 5 "$RUN_ID"
)

merge_job=$(
  "$TS_BIN" -D "$job5" -L merge_1200 env \
    "${COMMON_ENV[@]}" MAX_SAMPLES_PER_SHARD=200 \
    bash "$WAIT_MERGE_SCRIPT" "$job4" "$RUN_ID"
)

echo "[queue] jobs: smoke=${smoke_job} gpu4=${job4} gpu5=${job5} merge=${merge_job}"
"$TS_BIN" -l
