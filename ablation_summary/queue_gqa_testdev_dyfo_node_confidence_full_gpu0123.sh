#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_node_conf_gpu0123_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_node_conf_mmfocus_pad12_concat_full_12shards_${RUN_BATCH_ID}}"
SMOKE_RUN_ID="${RUN_ID}_smoke"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_SCRIPT="/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="/data2/lizhengxue/WorkSpace/onion/ablation_summary/merge_gqa_testdev_dyfo_focus_image_run.sh"

COMMON_ENV=(
  NUM_SHARDS=12
  SPLIT_NAME=testdev
  SHARD_LAUNCH_DELAY=180
  DYFO_DECISION_MODE=node_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TEXT_FOCUS_USE_IMAGE=1
  DYFO_FOCUS_PADDING=1.2
  DYFO_NODE_CONFIDENCE_THRESHOLD=0.80
  DYFO_NODE_CONFIDENCE_MARGIN=0.10
  DYFO_NODE_CONFIDENCE_SUPPORT_RATIO=0.60
  DYFO_NODE_CONFIDENCE_MIN_SUPPORT=2
)

export TS_SOCKET
"$TS_BIN" -S 4

smoke_id=$(
  "$TS_BIN" -L node_conf_smoke_gpu0 env \
    NUM_SHARDS=1 SHARD_IDS="0" MAX_SAMPLES_PER_SHARD=1 \
    SHARD_LAUNCH_DELAY=0 MERGE_ON_DONE=0 \
    DYFO_DECISION_MODE=node_confidence_override \
    DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal \
    DYFO_TRIGGER_MODE=always DYFO_FORCE_RUN_ALL_SAMPLES=1 \
    DYFO_TEXT_FOCUS_USE_IMAGE=1 DYFO_FOCUS_PADDING=1.2 \
    DYFO_NODE_CONFIDENCE_THRESHOLD=0.80 \
    DYFO_NODE_CONFIDENCE_MARGIN=0.10 \
    DYFO_NODE_CONFIDENCE_SUPPORT_RATIO=0.60 \
    DYFO_NODE_CONFIDENCE_MIN_SUPPORT=2 \
    bash "$RUN_SCRIPT" 0 "$SMOKE_RUN_ID"
)

job0=$("$TS_BIN" -D "$smoke_id" -L node_conf_gpu0_shards_0_4_8 env "${COMMON_ENV[@]}" SHARD_IDS="0 4 8" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 0 "$RUN_ID")
job1=$("$TS_BIN" -D "$smoke_id" -L node_conf_gpu1_shards_1_5_9 env "${COMMON_ENV[@]}" SHARD_IDS="1 5 9" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 1 "$RUN_ID")
job2=$("$TS_BIN" -D "$smoke_id" -L node_conf_gpu2_shards_2_6_10 env "${COMMON_ENV[@]}" SHARD_IDS="2 6 10" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 2 "$RUN_ID")
job3=$("$TS_BIN" -D "$smoke_id" -L node_conf_gpu3_shards_3_7_11 env "${COMMON_ENV[@]}" SHARD_IDS="3 7 11" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 3 "$RUN_ID")

merge_id=$(
  "$TS_BIN" -D "$job0" -D "$job1" -D "$job2" -D "$job3" \
    -L node_conf_merge_eval env "${COMMON_ENV[@]}" \
    bash "$MERGE_SCRIPT" "$RUN_ID"
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] SMOKE_RUN_ID=${SMOKE_RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] smoke=${smoke_id}"
echo "[queue] gpu_jobs=${job0},${job1},${job2},${job3}"
echo "[queue] merge=${merge_id}"
"$TS_BIN" -l
