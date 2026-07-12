#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_free_key_gpu2345_2000_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_free_key_8shards_2000_${RUN_BATCH_ID}}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"

export TS_SOCKET

"$TS_BIN" -S 4

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] total shards=8, max samples per shard=250, target total=2000"
echo "[queue] mode=best_focus_answer answer_image=concat_horizontal"

job2=$(
  "$TS_BIN" -L gpu2_shards0_4 env \
    NUM_SHARDS=8 SHARD_IDS="0 4" MAX_SAMPLES_PER_SHARD=250 \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 \
    DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 2 "$RUN_ID"
)
job3=$(
  "$TS_BIN" -L gpu3_shards1_5 env \
    NUM_SHARDS=8 SHARD_IDS="1 5" MAX_SAMPLES_PER_SHARD=250 \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 \
    DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 3 "$RUN_ID"
)
job4=$(
  "$TS_BIN" -L gpu4_shards2_6 env \
    NUM_SHARDS=8 SHARD_IDS="2 6" MAX_SAMPLES_PER_SHARD=250 \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 \
    DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 4 "$RUN_ID"
)
job5=$(
  "$TS_BIN" -L gpu5_shards3_7 env \
    NUM_SHARDS=8 SHARD_IDS="3 7" MAX_SAMPLES_PER_SHARD=250 \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 \
    DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 5 "$RUN_ID"
)

merge_job=$(
  "$TS_BIN" -D "$job2" -D "$job3" -D "$job4" -D "$job5" -L merge env \
    NUM_SHARDS=8 SPLIT_NAME=testdev \
    DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    bash "$MERGE_SCRIPT" "$RUN_ID"
)

echo "[queue] jobs: gpu2=${job2} gpu3=${job3} gpu4=${job4} gpu5=${job5} merge=${merge_job}"
"$TS_BIN" -l
