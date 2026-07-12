#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/lizhengxue/WorkSpace/onion"
TS_BIN="/data2/lizhengxue/.local/bin/ts"
RUN_BATCH_ID="${RUN_BATCH_ID:-textvqa_gqa_queue_$(date +%Y%m%d_%H%M%S)}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${RUN_BATCH_ID}.sock}"
SCRIPT="${REPO}/ablation_summary/run_wait_free_gpu_single_exp.sh"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-5 7 6 2 3}"
TS_CONCURRENCY="${TS_CONCURRENCY:-5}"

export TS_SOCKET

"$TS_BIN" -S "$TS_CONCURRENCY" >/dev/null

echo "RUN_BATCH_ID=${RUN_BATCH_ID}"
echo "TS_SOCKET=${TS_SOCKET}"
echo "CANDIDATE_GPUS=${CANDIDATE_GPUS}"
echo "TS_CONCURRENCY=${TS_CONCURRENCY}"

queue_exp() {
  local label="$1"
  local dataset="$2"
  local engine="$3"
  local variant="$4"
  local num_shards="$5"
  local min_free_mb="$6"
  local delay="$7"
  local run_id="${label}_${RUN_BATCH_ID}"

  local job_id
  job_id="$("$TS_BIN" -L "$label" env \
    CANDIDATE_GPUS="$CANDIDATE_GPUS" \
    SHARD_LAUNCH_DELAY="$delay" \
    bash "$SCRIPT" "$run_id" "$dataset" "$engine" "$variant" "$num_shards" "$min_free_mb")"
  echo "${label} job_id=${job_id} run_id=${run_id}"
}

queue_exp "textvqa_pure4b" "textvqa" "qwen3-VL-4B" "pure" 3 34000 90
queue_exp "textvqa_pure8b" "textvqa" "qwen3-VL-8B" "pure" 2 40000 100
queue_exp "gqa_pure4b" "gqa" "qwen3-VL-4B" "pure" 3 34000 90
queue_exp "gqa_pure8b" "gqa" "qwen3-VL-8B" "pure" 2 40000 100
queue_exp "gqa_dyfo4b" "gqa" "qwen3-VL-4B" "dyfo" 3 36000 120

"$TS_BIN" -l
