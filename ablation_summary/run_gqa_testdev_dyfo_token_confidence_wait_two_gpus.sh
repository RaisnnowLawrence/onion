#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-gqa_testdev_dyfo_token_confidence_full_6shards_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_SCRIPT="${SCRIPT_DIR}/run_gqa_testdev_dyfo_focus_image_single_gpu_3shards.sh"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gqa_testdev_dyfo_focus_image_run.sh"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-40960}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-/tmp/onion_gpu_locks}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-180}"
DYFO_TEXT_FOCUS_USE_IMAGE="${DYFO_TEXT_FOCUS_USE_IMAGE:-0}"
DYFO_FOCUS_PADDING="${DYFO_FOCUS_PADDING:-1.2}"
GPU_A=""
GPU_B=""
LOCK_DIRS=()
RUNNERS=()

mkdir -p "$GPU_LOCK_ROOT"

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk 'NR == 1 {print $1 + 0}'
}

clear_stale_lock() {
  local lock_dir="$1"
  local owner_pid=""
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

acquire_two_gpus() {
  local selected=()
  local gpu free_mb lock_dir recheck_a recheck_b
  while true; do
    selected=()
    LOCK_DIRS=()
    for gpu in $CANDIDATE_GPUS; do
      free_mb="$(gpu_free_mb "$gpu" 2>/dev/null || echo 0)"
      if [[ "$free_mb" -lt "$MIN_FREE_MB" ]]; then
        continue
      fi
      lock_dir="${GPU_LOCK_ROOT}/gpu${gpu}.lock"
      clear_stale_lock "$lock_dir"
      if mkdir "$lock_dir" 2>/dev/null; then
        printf '%s %s %s\n' "$$" "$RUN_ID" "$(date '+%F %T')" > "${lock_dir}/owner"
        selected+=("$gpu")
        LOCK_DIRS+=("$lock_dir")
        echo "[gpu-wait] provisional gpu=${gpu} free_mb=${free_mb}"
      fi
      if [[ "${#selected[@]}" -eq 2 ]]; then
        recheck_a="$(gpu_free_mb "${selected[0]}" 2>/dev/null || echo 0)"
        recheck_b="$(gpu_free_mb "${selected[1]}" 2>/dev/null || echo 0)"
        if [[ "$recheck_a" -ge "$MIN_FREE_MB" && "$recheck_b" -ge "$MIN_FREE_MB" ]]; then
          GPU_A="${selected[0]}"
          GPU_B="${selected[1]}"
          echo "[gpu-wait] acquired pair gpu_a=${GPU_A} free_a=${recheck_a} gpu_b=${GPU_B} free_b=${recheck_b}"
          return 0
        fi
        break
      fi
    done
    release_locks
    echo "[gpu-wait] need two GPUs with free_mb>=${MIN_FREE_MB}; candidates=${CANDIDATE_GPUS}; sleep=${GPU_WAIT_SECONDS}s"
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
  echo "[signal] stopping both GPU runners"
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
  NUM_SHARDS=6
  SPLIT_NAME=testdev
  DYFO_DECISION_MODE=token_confidence_override
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal
  DYFO_TRIGGER_MODE=always
  DYFO_FORCE_RUN_ALL_SAMPLES=1
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95
  DYFO_TOKEN_CONFIDENCE_MARGIN=0.0
  DYFO_TEXT_FOCUS_USE_IMAGE="$DYFO_TEXT_FOCUS_USE_IMAGE"
  DYFO_FOCUS_PADDING="$DYFO_FOCUS_PADDING"
  SHARD_LAUNCH_DELAY="$SHARD_LAUNCH_DELAY"
  MERGE_ON_DONE=0
)

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] candidates=${CANDIDATE_GPUS} min_free_mb=${MIN_FREE_MB}"
echo "[run] full GQA testdev; qwen3-VL-4B; token_confidence_override"
echo "[run] text_focus_use_image=${DYFO_TEXT_FOCUS_USE_IMAGE} focus_padding=${DYFO_FOCUS_PADDING}"
echo "[run] each selected GPU runs 3 staggered shards; total shards=6"

acquire_two_gpus

env "${COMMON_ENV[@]}" SHARD_IDS="0 2 4" LOG_TAG="${RUN_ID}_gpu${GPU_A}" \
  bash "$RUN_SCRIPT" "$GPU_A" "$RUN_ID" &
RUNNERS+=("$!")

env "${COMMON_ENV[@]}" SHARD_IDS="1 3 5" LOG_TAG="${RUN_ID}_gpu${GPU_B}" \
  bash "$RUN_SCRIPT" "$GPU_B" "$RUN_ID" &
RUNNERS+=("$!")

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

env NUM_SHARDS=6 SPLIT_NAME=testdev \
  DYFO_DECISION_MODE=token_confidence_override \
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
  DYFO_TRIGGER_MODE=always DYFO_FORCE_RUN_ALL_SAMPLES=1 \
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95 DYFO_TOKEN_CONFIDENCE_MARGIN=0.0 \
  DYFO_TEXT_FOCUS_USE_IMAGE="$DYFO_TEXT_FOCUS_USE_IMAGE" \
  DYFO_FOCUS_PADDING="$DYFO_FOCUS_PADDING" \
  bash "$MERGE_SCRIPT" "$RUN_ID"

echo "[done] RUN_ID=${RUN_ID} gpu_a=${GPU_A} gpu_b=${GPU_B}"
