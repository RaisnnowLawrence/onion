#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_dyfo_region_audit_gpu345_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_region_audit_9shards_900_${RUN_BATCH_ID}}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="$SCRIPT_DIR/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
FINISH_SCRIPT="$SCRIPT_DIR/finish_gqa_dyfo_region_audit.sh"

export TS_SOCKET
"$TS_BIN" -S 3

COMMON_ENV=(
  NUM_SHARDS=9
  SPLIT_NAME=testdev
  DYFO_DECISION_MODE=token_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_NODE_ANSWER_IMAGE_MODE=crop
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TEXT_FOCUS_USE_IMAGE=1
  DYFO_FOCUS_PADDING=1.2
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95
  DYFO_TOKEN_CONFIDENCE_MARGIN=0.0
  DYFO_REGION_AUDIT=1
  DYFO_REGION_AUDIT_SAVE_CROPS=1
  MAX_SAMPLES_PER_SHARD=100
  SHARD_LAUNCH_DELAY=120
  MERGE_ON_DONE=0
)

job3=$(
  "$TS_BIN" -L audit_gpu3_shards0_3_6 env \
    "${COMMON_ENV[@]}" SHARD_IDS="0 3 6" \
    bash "$RUN_SCRIPT" 3 "$RUN_ID"
)
job4=$(
  "$TS_BIN" -L audit_gpu4_shards1_4_7 env \
    "${COMMON_ENV[@]}" SHARD_IDS="1 4 7" \
    bash "$RUN_SCRIPT" 4 "$RUN_ID"
)
job5=$(
  "$TS_BIN" -L audit_gpu5_shards2_5_8 env \
    "${COMMON_ENV[@]}" SHARD_IDS="2 5 8" \
    bash "$RUN_SCRIPT" 5 "$RUN_ID"
)
finish_job=$(
  "$TS_BIN" -D "$job5" -L audit_merge_and_report \
    bash "$FINISH_SCRIPT" "$job3" "$job4" "$RUN_ID"
)

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] jobs gpu3=${job3} gpu4=${job4} gpu5=${job5} finish=${finish_job}"
echo "[queue] target=900 samples; 9 shards; crop-only node answers; persistent node crops"
"$TS_BIN" -l
