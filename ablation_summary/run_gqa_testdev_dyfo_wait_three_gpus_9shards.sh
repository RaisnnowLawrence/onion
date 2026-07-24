#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-46080}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-/tmp/onion_gpu_locks}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-180}"
DYFO_DUAL_VISUAL_EXPERTS="${DYFO_DUAL_VISUAL_EXPERTS:-0}"
DYFO_NODE_ANSWER_IMAGE_MODE="${DYFO_NODE_ANSWER_IMAGE_MODE:-concat_horizontal}"
SELECTED_GPUS=()
LOCK_DIRS=()
RUNNERS=()

mkdir -p "$GPU_LOCK_ROOT"

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk 'NR == 1 {print $1 + 0}'
}

clear_stale_lock() {
  local lock_dir="$1" owner_pid=""
  [[ -d "$lock_dir" ]] || return 0
  if [[ -f "${lock_dir}/owner" ]]; then
    owner_pid="$(awk 'NR == 1 {print $1}' "${lock_dir}/owner" 2>/dev/null || true)"
  fi
  if [[ -n "$owner_pid" ]] && kill -0 "$owner_pid" 2>/dev/null; then
    return 0
  fi
  rm -f "${lock_dir}/owner"
  rmdir "$lock_dir" 2>/dev/null || true
}

release_locks() {
  local lock_dir owner_pid
  for lock_dir in "${LOCK_DIRS[@]}"; do
    [[ -d "$lock_dir" ]] || continue
    owner_pid="$(awk 'NR == 1 {print $1}' "${lock_dir}/owner" 2>/dev/null || true)"
    if [[ "$owner_pid" == "$$" ]]; then
      rm -f "${lock_dir}/owner"
      rmdir "$lock_dir" 2>/dev/null || true
    fi
  done
  LOCK_DIRS=()
}

acquire_three_gpus() {
  local selected=() gpu free_mb lock_dir valid recheck
  while true; do
    selected=()
    LOCK_DIRS=()
    for gpu in $CANDIDATE_GPUS; do
      free_mb="$(gpu_free_mb "$gpu" 2>/dev/null || echo 0)"
      [[ "$free_mb" -ge "$MIN_FREE_MB" ]] || continue
      lock_dir="${GPU_LOCK_ROOT}/gpu${gpu}.lock"
      clear_stale_lock "$lock_dir"
      if mkdir "$lock_dir" 2>/dev/null; then
        printf '%s %s %s\n' "$$" "$RUN_ID" "$(date '+%F %T')" > "${lock_dir}/owner"
        selected+=("$gpu")
        LOCK_DIRS+=("$lock_dir")
        echo "[gpu-wait] provisional gpu=${gpu} free_mb=${free_mb}"
      fi
      [[ "${#selected[@]}" -eq 3 ]] && break
    done
    if [[ "${#selected[@]}" -eq 3 ]]; then
      valid=1
      for gpu in "${selected[@]}"; do
        recheck="$(gpu_free_mb "$gpu" 2>/dev/null || echo 0)"
        [[ "$recheck" -ge "$MIN_FREE_MB" ]] || valid=0
      done
      if [[ "$valid" -eq 1 ]]; then
        SELECTED_GPUS=("${selected[@]}")
        echo "[gpu-wait] acquired three GPUs: ${SELECTED_GPUS[*]}"
        return 0
      fi
    fi
    release_locks
    echo "[gpu-wait] need three GPUs with free_mb>=${MIN_FREE_MB}; candidates=${CANDIDATE_GPUS}; sleep=${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
}

stop_runners() {
  local pid
  for pid in "${RUNNERS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}

on_term() {
  echo "[signal] stopping three GPU runners"
  stop_runners
  release_locks
  exit 143
}

on_exit() {
  local code="$?"
  if [[ "$code" -ne 0 ]]; then
    stop_runners
  fi
  release_locks
  exit "$code"
}

trap on_term TERM INT
trap on_exit EXIT

COMMON_ENV=(
  NUM_SHARDS=9
  SPLIT_NAME=testdev
  DYFO_DECISION_MODE=node_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_NODE_ANSWER_IMAGE_MODE="$DYFO_NODE_ANSWER_IMAGE_MODE"
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TEXT_FOCUS_USE_IMAGE=1
  DYFO_FOCUS_PADDING=1.2
  DYFO_DUAL_VISUAL_EXPERTS="$DYFO_DUAL_VISUAL_EXPERTS"
  DYFO_NODE_CONFIDENCE_THRESHOLD=0.80
  DYFO_NODE_CONFIDENCE_MARGIN=0.10
  DYFO_NODE_CONFIDENCE_SUPPORT_RATIO=0.60
  DYFO_NODE_CONFIDENCE_MIN_SUPPORT=2
  SHARD_LAUNCH_DELAY="$SHARD_LAUNCH_DELAY"
  MERGE_ON_DONE=0
)

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] full GQA testdev_balanced; three GPUs x three shards"
echo "[run] dual_visual_experts=${DYFO_DUAL_VISUAL_EXPERTS} node_image=${DYFO_NODE_ANSWER_IMAGE_MODE}"
echo "[run] candidates=${CANDIDATE_GPUS} min_free_mb=${MIN_FREE_MB}"

acquire_three_gpus

SHARD_GROUPS=("0 3 6" "1 4 7" "2 5 8")
for index in 0 1 2; do
  gpu="${SELECTED_GPUS[$index]}"
  env "${COMMON_ENV[@]}" SHARD_IDS="${SHARD_GROUPS[$index]}" LOG_TAG="${RUN_ID}_gpu${gpu}" \
    bash "$RUN_SCRIPT" "$gpu" "$RUN_ID" &
  RUNNERS+=("$!")
done

status=0
for pid in "${RUNNERS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[error] at least one GPU runner failed; partial outputs preserved" >&2
  exit "$status"
fi

env "${COMMON_ENV[@]}" bash "$MERGE_SCRIPT" "$RUN_ID"
echo "[done] RUN_ID=${RUN_ID} GPUs=${SELECTED_GPUS[*]}"
