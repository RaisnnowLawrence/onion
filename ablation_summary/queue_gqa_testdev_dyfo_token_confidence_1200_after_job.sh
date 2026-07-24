#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_gqa_testdev_dyfo_conservative_override_resume_full_gpu45_20260713_175341.sock}"
WAIT_FOR_JOB_ID="${WAIT_FOR_JOB_ID:-2}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_token_confidence_1200_gpu45_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_token_confidence_6shards_1200_${RUN_BATCH_ID}}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
WAIT_MERGE_SCRIPT="${SCRIPT_DIR}/wait_for_task_then_merge_gqa_testdev_dyfo.sh"

export TS_SOCKET
"$TS_BIN" -S 2

COMMON_ENV=(
  NUM_SHARDS=6
  SPLIT_NAME=testdev
  DYFO_DECISION_MODE=token_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95
  DYFO_TOKEN_CONFIDENCE_MARGIN=0.0
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] dependency=${WAIT_FOR_JOB_ID}"
echo "[queue] confidence=exp(mean(answer-token logprob)); threshold=0.95; margin=0.0"
echo "[queue] target=1200; 6 shards x 200 samples"
echo "[queue] gpu4 shards=0,2,4; gpu5 shards=1,3,5"

job4=$(
  "$TS_BIN" -D "$WAIT_FOR_JOB_ID" -L token_conf_gpu4_shards0_2_4 env \
    "${COMMON_ENV[@]}" SHARD_IDS="0 2 4" MAX_SAMPLES_PER_SHARD=200 \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 \
    bash "$RUN_SCRIPT" 4 "$RUN_ID"
)

job5=$(
  "$TS_BIN" -D "$WAIT_FOR_JOB_ID" -L token_conf_gpu5_shards1_3_5 env \
    "${COMMON_ENV[@]}" SHARD_IDS="1 3 5" MAX_SAMPLES_PER_SHARD=200 \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 \
    bash "$RUN_SCRIPT" 5 "$RUN_ID"
)

merge_job=$(
  "$TS_BIN" -D "$job5" -L merge_token_conf_1200 env \
    "${COMMON_ENV[@]}" MAX_SAMPLES_PER_SHARD=200 \
    bash "$WAIT_MERGE_SCRIPT" "$job4" "$RUN_ID"
)

echo "[queue] jobs: gpu4=${job4} gpu5=${job5} merge=${merge_job}"
"$TS_BIN" -l
