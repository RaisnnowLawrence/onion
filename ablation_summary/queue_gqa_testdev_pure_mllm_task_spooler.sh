#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/lizhengxue/WorkSpace/onion"
TS_BIN="/data2/lizhengxue/.local/bin/ts"
RUN_BATCH_ID="${RUN_BATCH_ID:-gqa_testdev_pure_mllm_queue_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
SCRIPT="${REPO}/ablation_summary/run_wait_free_gpu_single_exp.sh"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-5 7 6 2 3 0 1 4}"
TS_CONCURRENCY="${TS_CONCURRENCY:-2}"

export TS_SOCKET

"$TS_BIN" -S "$TS_CONCURRENCY" >/dev/null

echo "RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "TS_SOCKET=${TS_SOCKET}"
echo "CANDIDATE_GPUS=${CANDIDATE_GPUS}"
echo "TS_CONCURRENCY=${TS_CONCURRENCY}"

queue_exp() {
  local label="$1"
  local engine="$2"
  local min_free_mb="$3"
  local delay="$4"
  local run_id="${label}_${RUN_BATCH_ID}"
  local job_id

  job_id="$("$TS_BIN" -L "$label" env \
    CANDIDATE_GPUS="$CANDIDATE_GPUS" \
    SPLIT_NAME=testdev \
    SHARD_LAUNCH_DELAY="$delay" \
    bash "$SCRIPT" "$run_id" gqa "$engine" pure 3 "$min_free_mb")"
  echo "${label} job_id=${job_id} run_id=${run_id}"
}

queue_exp "gqa_testdev_pure4b_3shards" "qwen3-VL-4B" 34000 90
queue_exp "gqa_testdev_pure8b_3shards" "qwen3-VL-8B" 47000 120

"$TS_BIN" -l
