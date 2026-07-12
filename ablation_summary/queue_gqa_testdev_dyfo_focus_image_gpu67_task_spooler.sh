#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/lizhengxue/WorkSpace/onion"
TS_BIN="/data2/lizhengxue/.local/bin/ts"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_focus_concat_gpu67_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_SCRIPT="${REPO}/ablation_summary/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${REPO}/ablation_summary/merge_gqa_testdev_dyfo_focus_image_run.sh"
TS_CONCURRENCY="${TS_CONCURRENCY:-2}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-150}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_focus_concat_6shards_${RUN_BATCH_ID}}"
DYFO_DECISION_MODE="${DYFO_DECISION_MODE:-best_focus_answer}"
DYFO_ANSWER_IMAGE_MODE="${DYFO_ANSWER_IMAGE_MODE:-concat_horizontal}"

export TS_SOCKET

"$TS_BIN" -S "$TS_CONCURRENCY" >/dev/null

echo "RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "RUN_ID=${RUN_ID}"
echo "TS_SOCKET=${TS_SOCKET}"
echo "TS_CONCURRENCY=${TS_CONCURRENCY}"
echo "SHARD_LAUNCH_DELAY=${SHARD_LAUNCH_DELAY}"
echo "DYFO_DECISION_MODE=${DYFO_DECISION_MODE}"
echo "DYFO_ANSWER_IMAGE_MODE=${DYFO_ANSWER_IMAGE_MODE}"

queue_gpu() {
  local gpu_id="$1"
  local shard_ids="$2"
  local label="gqa_testdev_dyfo_focus_concat_gpu${gpu_id}_${shard_ids// /-}"
  local job_id

  job_id="$("$TS_BIN" -L "$label" env \
    NUM_SHARDS=6 \
    SHARD_IDS="$shard_ids" \
    SHARD_LAUNCH_DELAY="$SHARD_LAUNCH_DELAY" \
    SPLIT_NAME=testdev \
    DYFO_DECISION_MODE="$DYFO_DECISION_MODE" \
    DYFO_ANSWER_IMAGE_MODE="$DYFO_ANSWER_IMAGE_MODE" \
    MERGE_ON_DONE=0 \
    bash "$RUN_SCRIPT" "$gpu_id" "$RUN_ID")"
  echo "${label} job_id=${job_id} run_id=${RUN_ID}"
}

job6="$(queue_gpu 6 "0 2 4" | tee /dev/stderr | awk -F'job_id=' '{print $2}' | awk '{print $1}')"
job7="$(queue_gpu 7 "1 3 5" | tee /dev/stderr | awk -F'job_id=' '{print $2}' | awk '{print $1}')"

merge_job="$("$TS_BIN" -L "gqa_testdev_dyfo_focus_concat_merge" -D "${job6},${job7}" env \
  NUM_SHARDS=6 \
  SPLIT_NAME=testdev \
  DYFO_DECISION_MODE="$DYFO_DECISION_MODE" \
  DYFO_ANSWER_IMAGE_MODE="$DYFO_ANSWER_IMAGE_MODE" \
  bash "$MERGE_SCRIPT" "$RUN_ID")"
echo "merge job_id=${merge_job} depends_on=${job6},${job7} run_id=${RUN_ID}"

"$TS_BIN" -l
