#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_node_concat_gpu01234_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_token_conf_mmfocus_pad12_node_concat_full_15shards_${RUN_BATCH_ID}}"
SMOKE_RUN_ID="${RUN_ID}_smoke"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_SCRIPT="/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="/data2/lizhengxue/WorkSpace/onion/ablation_summary/merge_gqa_testdev_dyfo_focus_image_run.sh"

COMMON_ENV=(
  NUM_SHARDS=15
  SPLIT_NAME=testdev
  SHARD_LAUNCH_DELAY=180
  DYFO_DECISION_MODE=token_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TEXT_FOCUS_USE_IMAGE=1
  DYFO_FOCUS_PADDING=1.2
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95
  DYFO_TOKEN_CONFIDENCE_MARGIN=0.0
)

export TS_SOCKET
"$TS_BIN" -S 5

smoke_id=$(
  "$TS_BIN" -L node_concat_smoke_gpu0 env \
    NUM_SHARDS=1 SHARD_IDS="0" MAX_SAMPLES_PER_SHARD=1 \
    SHARD_LAUNCH_DELAY=0 MERGE_ON_DONE=0 \
    DYFO_DECISION_MODE=token_confidence_override \
    DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal \
    DYFO_TRIGGER_MODE=always DYFO_FORCE_RUN_ALL_SAMPLES=1 \
    DYFO_TEXT_FOCUS_USE_IMAGE=1 DYFO_FOCUS_PADDING=1.2 \
    DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95 DYFO_TOKEN_CONFIDENCE_MARGIN=0.0 \
    bash "$RUN_SCRIPT" 0 "$SMOKE_RUN_ID"
)

job0=$("$TS_BIN" -D "$smoke_id" -L node_concat_gpu0_shards_0_5_10 env "${COMMON_ENV[@]}" SHARD_IDS="0 5 10" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 0 "$RUN_ID")
job1=$("$TS_BIN" -D "$smoke_id" -L node_concat_gpu1_shards_1_6_11 env "${COMMON_ENV[@]}" SHARD_IDS="1 6 11" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 1 "$RUN_ID")
job2=$("$TS_BIN" -D "$smoke_id" -L node_concat_gpu2_shards_2_7_12 env "${COMMON_ENV[@]}" SHARD_IDS="2 7 12" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 2 "$RUN_ID")
job3=$("$TS_BIN" -D "$smoke_id" -L node_concat_gpu3_shards_3_8_13 env "${COMMON_ENV[@]}" SHARD_IDS="3 8 13" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 3 "$RUN_ID")
job4=$("$TS_BIN" -D "$smoke_id" -L node_concat_gpu4_shards_4_9_14 env "${COMMON_ENV[@]}" SHARD_IDS="4 9 14" MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 4 "$RUN_ID")

merge_id=$(
  "$TS_BIN" -D "$job0" -D "$job1" -D "$job2" -D "$job3" -D "$job4" \
    -L node_concat_merge_eval env "${COMMON_ENV[@]}" \
    bash "$MERGE_SCRIPT" "$RUN_ID"
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] SMOKE_RUN_ID=${SMOKE_RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] smoke=${smoke_id}"
echo "[queue] gpu_jobs=${job0},${job1},${job2},${job3},${job4}"
echo "[queue] merge=${merge_id}"
"$TS_BIN" -l
