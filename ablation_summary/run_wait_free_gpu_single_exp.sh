#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: $0 <run_id> <dataset> <engine> <variant:pure|dyfo> <num_shards> <min_free_mb>" >&2
  exit 2
fi

RUN_ID="$1"
DATASET="$2"
ENGINE="$3"
VARIANT="$4"
NUM_SHARDS="$5"
MIN_FREE_MB="$6"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
DATASET_ROOT="/data2/lizhengxue/datasets"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/${DATASET}"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_textvqa_gqa_queue"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
SUMMARY_LOG="${EXP_OUT}/summary.log"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-5 7 6 2 3}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-120}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-90}"
SPLIT_NAME="${SPLIT_NAME:-val}"
DYFO_DECISION_MODE="${DYFO_DECISION_MODE:-evidence_inject}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-/tmp/onion_gpu_locks}"
CHILDREN=()
GPU_ID=""
GPU_LOCK_DIR=""

mkdir -p "$LOG_ROOT" "$EXP_OUT" "$GPU_LOCK_ROOT"

COMMON_ARGS=(
  "$REPO/forward_code/onion.py"
  --dataset_name "$DATASET"
  --split_name "$SPLIT_NAME"
  --engine "$ENGINE"
  --caption_type vinvl
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --dataset_root "$DATASET_ROOT"
  --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations
  --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
  --train_sim_metric answer
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
  --output_path "$EXP_OUT"
  --cache_path "${EXP_OUT}/cache"
  --num_shards "$NUM_SHARDS"
)

if [[ -n "${GQA_QUESTION_FILE:-}" ]]; then
  COMMON_ARGS+=(--gqa_question_file "$GQA_QUESTION_FILE")
fi

case "$VARIANT" in
  pure)
    COMMON_ARGS+=(--context_mode empty --remove_caption)
    ;;
  dyfo)
    COMMON_ARGS+=(
      --context_mode no_round_state
      --use_image_enhance
      --mcts_action_mode dyfo_evidence
      --use_dyfo_visual_evidence
      --dyfo_decision_mode "$DYFO_DECISION_MODE"
      --dyfo_trigger_mode visual_detail
      --dyfo_n_simulations 6
      --dyfo_max_depth 3
      --dyfo_area_reward compact
    )
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ -n "${MAX_SAMPLES_PER_SHARD:-}" ]]; then
  COMMON_ARGS+=(--max_samples_per_shard "$MAX_SAMPLES_PER_SHARD")
fi

gpu_free_mb() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print $1 + 0; exit}'
}

clear_stale_lock() {
  local lock_dir="$1"
  local owner_file="${lock_dir}/owner"
  local owner_pid=""
  local now
  local mtime
  local age

  if [[ ! -d "$lock_dir" ]]; then
    return 0
  fi

  if [[ ! -f "$owner_file" ]]; then
    now="$(date +%s)"
    mtime="$(stat -c %Y "$lock_dir" 2>/dev/null || echo "$now")"
    age=$((now - mtime))
    if [[ "$age" -gt 300 ]]; then
      rmdir "$lock_dir" 2>/dev/null || true
    fi
    return 0
  fi

  owner_pid="$(awk 'NR == 1 {print $1}' "$owner_file" 2>/dev/null || true)"
  if [[ -n "$owner_pid" ]] && kill -0 "$owner_pid" 2>/dev/null; then
    return 0
  fi

  rm -f "$owner_file"
  rmdir "$lock_dir" 2>/dev/null || true
}

acquire_gpu() {
  while true; do
    for gpu in $CANDIDATE_GPUS; do
      local free_mb
      free_mb="$(gpu_free_mb "$gpu")"
      [[ -z "$free_mb" ]] && continue
      [[ "$free_mb" -lt "$MIN_FREE_MB" ]] && continue

      local lock_dir="${GPU_LOCK_ROOT}/gpu${gpu}.lock"
      clear_stale_lock "$lock_dir"
      if mkdir "$lock_dir" 2>/dev/null; then
        printf '%s %s %s\n' "$$" "$RUN_ID" "$(date '+%F %T')" > "${lock_dir}/owner"
        GPU_ID="$gpu"
        GPU_LOCK_DIR="$lock_dir"
        echo "[gpu] acquired gpu=${GPU_ID} free_mb=${free_mb} min_free_mb=${MIN_FREE_MB}"
        return 0
      fi
    done
    echo "[gpu] no eligible GPU yet; candidates=${CANDIDATE_GPUS} min_free_mb=${MIN_FREE_MB}; sleep=${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
}

release_gpu() {
  if [[ -n "$GPU_LOCK_DIR" && -d "$GPU_LOCK_DIR" ]]; then
    local owner_pid=""
    if [[ -f "${GPU_LOCK_DIR}/owner" ]]; then
      owner_pid="$(awk 'NR == 1 {print $1}' "${GPU_LOCK_DIR}/owner" 2>/dev/null || true)"
    fi
    if [[ "$owner_pid" == "$$" ]]; then
      rm -f "${GPU_LOCK_DIR}/owner"
      rmdir "$GPU_LOCK_DIR" 2>/dev/null || true
      echo "[gpu] released gpu=${GPU_ID}"
    fi
  fi
}

merge_partial() {
  local count
  count="$(find "${EXP_OUT}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l || true)"
  echo "[merge] samples=${count} out=${EXP_OUT}"
  if [[ "${count}" -eq 0 ]]; then
    echo "[merge] skip: no prompt_samples yet"
    return 0
  fi
  "$PY" "${COMMON_ARGS[@]}" --merge_only --summary_log "$SUMMARY_LOG"
}

on_term() {
  echo "[signal] received TERM/INT, stopping children and merging partial results"
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait || true
  merge_partial || true
  release_gpu
  exit 143
}

on_exit() {
  local code="$?"
  if [[ "$code" -ne 0 ]]; then
    echo "[exit] controller exiting with status=${code}; stopping child shards before releasing GPU"
    for pid in "${CHILDREN[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    wait || true
  fi
  release_gpu
  exit "$code"
}

trap on_term TERM INT
trap on_exit EXIT

run_shard() {
  local shard_id="$1"
  local log_file="${LOG_ROOT}/${RUN_ID}_gpu${GPU_ID}_shard${shard_id}.log"
  echo "[launch] gpu=${GPU_ID} shard=${shard_id}/${NUM_SHARDS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PY" "${COMMON_ARGS[@]}" --shard_id "$shard_id" > "$log_file" 2>&1 &
  CHILDREN+=("$!")
}

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] DATASET=${DATASET} SPLIT_NAME=${SPLIT_NAME} ENGINE=${ENGINE} VARIANT=${VARIANT}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] NUM_SHARDS=${NUM_SHARDS} MIN_FREE_MB=${MIN_FREE_MB}"
echo "[run] CANDIDATE_GPUS=${CANDIDATE_GPUS}"
echo "[run] GQA_QUESTION_FILE=${GQA_QUESTION_FILE:-}"
echo "[run] DYFO_DECISION_MODE=${DYFO_DECISION_MODE}"
echo "[run] partial merge on TERM/INT; final merge after all shards finish"

acquire_gpu

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "$shard_id"
  if [[ "$shard_id" -lt $((NUM_SHARDS - 1)) ]]; then
    sleep "$SHARD_LAUNCH_DELAY"
  fi
done

status=0
for pid in "${CHILDREN[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
merge_partial
echo "[done] summary=${SUMMARY_LOG}"
exit "$status"
