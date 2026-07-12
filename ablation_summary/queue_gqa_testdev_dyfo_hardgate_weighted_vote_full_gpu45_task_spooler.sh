#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_hardgate_weighted_vote_full_gpu45_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
RUN_ID="${RUN_ID:-gqa_testdev_dyfo_hardgate_weighted_vote_6shards_full_${RUN_BATCH_ID}}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"

export TS_SOCKET

"$TS_BIN" -S 2

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] RUN_ID=${RUN_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] dataset=gqa testdev_balanced full"
echo "[queue] mode=weighted_vote answer_image=concat_horizontal"
echo "[queue] total shards=6; gpu4 shards=0,2,4; gpu5 shards=1,3,5"

job4=$(
  "$TS_BIN" -L gpu4_shards0_2_4 env \
    NUM_SHARDS=6 SHARD_IDS="0 2 4" \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=180 \
    DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 4 "$RUN_ID"
)

job5=$(
  "$TS_BIN" -L gpu5_shards1_3_5 env \
    NUM_SHARDS=6 SHARD_IDS="1 3 5" \
    SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=180 \
    DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 5 "$RUN_ID"
)

merge_job=$(
  "$TS_BIN" -D "$job4" -D "$job5" -L merge env \
    NUM_SHARDS=6 SPLIT_NAME=testdev \
    DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
    bash "$MERGE_SCRIPT" "$RUN_ID"
)

echo "[queue] jobs: gpu4=${job4} gpu5=${job5} merge=${merge_job}"
"$TS_BIN" -l
