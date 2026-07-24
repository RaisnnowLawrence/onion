#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_conservative_override_resume_full_gpu45_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_ID="gqa_testdev_dyfo_conservative_override_6shards_1200_gqa_testdev_dyfo_conservative_override_1200_gpu45_20260713_155417"
LOG_TAG="${RUN_ID}_resume_full_${RUN_BATCH_ID}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
WAIT_MERGE_SCRIPT="${SCRIPT_DIR}/wait_for_task_then_merge_gqa_testdev_dyfo.sh"
EXP_OUT="/data2/lizhengxue/WorkSpace/onion_output/gqa/${RUN_ID}"

existing_count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l)"
if [[ "$existing_count" -lt 1200 ]]; then
  echo "[error] expected at least 1200 existing samples, found ${existing_count}" >&2
  exit 1
fi

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
echo "[queue] LOG_TAG=${LOG_TAG}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] existing=${existing_count}/12578; remaining=$((12578 - existing_count))"
echo "[queue] resume without MAX_SAMPLES_PER_SHARD"
echo "[queue] gpu4 shards=0,2,4; gpu5 shards=1,3,5"

job4=$(
  "$TS_BIN" -L resume_gpu4_shards0_2_4 env \
    "${COMMON_ENV[@]}" SHARD_IDS="0 2 4" \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 LOG_TAG="$LOG_TAG" \
    bash "$RUN_SCRIPT" 4 "$RUN_ID"
)

job5=$(
  "$TS_BIN" -L resume_gpu5_shards1_3_5 env \
    "${COMMON_ENV[@]}" SHARD_IDS="1 3 5" \
    SHARD_LAUNCH_DELAY=180 MERGE_ON_DONE=0 LOG_TAG="$LOG_TAG" \
    bash "$RUN_SCRIPT" 5 "$RUN_ID"
)

merge_job=$(
  "$TS_BIN" -D "$job5" -L merge_full_barrier env \
    "${COMMON_ENV[@]}" \
    bash "$WAIT_MERGE_SCRIPT" "$job4" "$RUN_ID"
)

echo "[queue] jobs: gpu4=${job4} gpu5=${job5} merge=${merge_job}"
"$TS_BIN" -l
