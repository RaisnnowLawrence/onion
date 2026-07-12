#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_dyfo_hardgate_600_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"
ROUTER_RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_mllm_router_dyfo_single_gpu_2shards.sh"
ROUTER_MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_mllm_router_dyfo_run.sh"

WEIGHTED_RUN_ID="${WEIGHTED_RUN_ID:-gqa_testdev_dyfo_hardgate_weighted_vote_6shards_600_${RUN_BATCH_ID}}"
BEST_RUN_ID="${BEST_RUN_ID:-gqa_testdev_dyfo_hardgate_best_focus_6shards_600_${RUN_BATCH_ID}}"
ROUTER_RUN_ID="${ROUTER_RUN_ID:-gqa_testdev_mllm_router_dyfo_hardgate_6shards_600_${RUN_BATCH_ID}}"

export TS_SOCKET

"$TS_BIN" -S 6

echo "[queue] RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] weighted=${WEIGHTED_RUN_ID}"
echo "[queue] best=${BEST_RUN_ID}"
echo "[queue] router=${ROUTER_RUN_ID}"
echo "[queue] total shards=6, max samples per shard=100, target total=600"
echo "[queue] weighted GPUs=0,1,2; best GPUs=3,4,5; router waits then GPUs=0,1,2"

w0=$("$TS_BIN" -L weighted_gpu0_shards0_3 env NUM_SHARDS=6 SHARD_IDS="0 3" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 0 "$WEIGHTED_RUN_ID")
w1=$("$TS_BIN" -L weighted_gpu1_shards1_4 env NUM_SHARDS=6 SHARD_IDS="1 4" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 1 "$WEIGHTED_RUN_ID")
w2=$("$TS_BIN" -L weighted_gpu2_shards2_5 env NUM_SHARDS=6 SHARD_IDS="2 5" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 2 "$WEIGHTED_RUN_ID")
wm=$("$TS_BIN" -D "$w0" -D "$w1" -D "$w2" -L weighted_merge env NUM_SHARDS=6 SPLIT_NAME=testdev DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal bash "$MERGE_SCRIPT" "$WEIGHTED_RUN_ID")

b3=$("$TS_BIN" -L best_gpu3_shards0_3 env NUM_SHARDS=6 SHARD_IDS="0 3" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 3 "$BEST_RUN_ID")
b4=$("$TS_BIN" -L best_gpu4_shards1_4 env NUM_SHARDS=6 SHARD_IDS="1 4" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 4 "$BEST_RUN_ID")
b5=$("$TS_BIN" -L best_gpu5_shards2_5 env NUM_SHARDS=6 SHARD_IDS="2 5" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$RUN_SCRIPT" 5 "$BEST_RUN_ID")
bm=$("$TS_BIN" -D "$b3" -D "$b4" -D "$b5" -L best_merge env NUM_SHARDS=6 SPLIT_NAME=testdev DYFO_DECISION_MODE=best_focus_answer DYFO_ANSWER_IMAGE_MODE=concat_horizontal bash "$MERGE_SCRIPT" "$BEST_RUN_ID")

r0=$("$TS_BIN" -D "$wm" -D "$bm" -L router_gpu0_shards0_3 env NUM_SHARDS=6 SHARD_IDS="0 3" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$ROUTER_RUN_SCRIPT" 0 "$ROUTER_RUN_ID")
r1=$("$TS_BIN" -D "$wm" -D "$bm" -L router_gpu1_shards1_4 env NUM_SHARDS=6 SHARD_IDS="1 4" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$ROUTER_RUN_SCRIPT" 1 "$ROUTER_RUN_ID")
r2=$("$TS_BIN" -D "$wm" -D "$bm" -L router_gpu2_shards2_5 env NUM_SHARDS=6 SHARD_IDS="2 5" MAX_SAMPLES_PER_SHARD=100 SPLIT_NAME=testdev SHARD_LAUNCH_DELAY=120 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal MERGE_ON_DONE=0 bash "$ROUTER_RUN_SCRIPT" 2 "$ROUTER_RUN_ID")
rm=$("$TS_BIN" -D "$r0" -D "$r1" -D "$r2" -L router_merge env NUM_SHARDS=6 SPLIT_NAME=testdev MAX_SAMPLES_PER_SHARD=100 DYFO_DECISION_MODE=weighted_vote DYFO_ANSWER_IMAGE_MODE=concat_horizontal bash "$ROUTER_MERGE_SCRIPT" "$ROUTER_RUN_ID")

echo "[queue] jobs weighted: ${w0},${w1},${w2}, merge=${wm}"
echo "[queue] jobs best: ${b3},${b4},${b5}, merge=${bm}"
echo "[queue] jobs router: ${r0},${r1},${r2}, merge=${rm}"
"$TS_BIN" -l
