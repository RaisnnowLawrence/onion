#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
PROFILE_DIR=${PROFILE_DIR:-${REPORT_DIR}/strategy_rag_profiles}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_multi_strategy_profile_n400}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-400}
NUM_SHARDS=${NUM_SHARDS:-3}
SAMPLES_PER_SHARD=$(( (MAX_TRAIN_SAMPLES + NUM_SHARDS - 1) / NUM_SHARDS ))

mkdir -p "${PROFILE_DIR}" "${LOG_DIR}"

base_args=(
  --caption_type vinvl
  --n_shot 1
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --engine "${ENGINE}"
  --sg_path "${DATA_ROOT}/input_text/scene_graph_text"
  --train_sim_metric answer
  --train_sim_file "${DATA_ROOT}/input_text/scene_graph_text/train_object_select_answer.pk"
  --tag_path "${DATA_ROOT}/input_text/coco_caption_pred_tags"
  --split_name train
  --max_samples_per_shard "${SAMPLES_PER_SHARD}"
)

run_profile_shard() {
  local strategy="$1"
  local gpu="$2"
  local shard="$3"
  shift 3
  local out="${OUT_ROOT}/${ENGINE}_multi_strategy_profile_${strategy}_train_n${MAX_TRAIN_SAMPLES}_${NUM_SHARDS}shards"
  local cache="${DATA_ROOT}/image_cache_onion/cache_multi_strategy_profile_${strategy}_train_n${MAX_TRAIN_SAMPLES}_${NUM_SHARDS}shards"
  local profile="${PROFILE_DIR}/multi_${strategy}_train_n${MAX_TRAIN_SAMPLES}_shard${shard}.jsonl"
  rm -f "${profile}"
  mkdir -p "${out}" "${cache}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib_multi_strategy \
  "${PY}" "${REPO}/forward_code/onion.py" \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${base_args[@]}" \
    --strategy_name "${strategy}" \
    --strategy_profile_output "${profile}" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    "$@" \
    > "${LOG_DIR}/${strategy}_gpu${gpu}_shard${shard}.log" 2>&1
}

run_profile() {
  local strategy="$1"
  local gpu="$2"
  shift 2
  local out="${OUT_ROOT}/${ENGINE}_multi_strategy_profile_${strategy}_train_n${MAX_TRAIN_SAMPLES}_${NUM_SHARDS}shards"
  rm -rf "${out}/prompt_samples" "${out}/format_samples"
  local shard
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_profile_shard "${strategy}" "${gpu}" "${shard}" "$@" &
  done
}

echo "[multi-strategy] train profile slice: ${MAX_TRAIN_SAMPLES} samples per strategy, ${NUM_SHARDS} shards, ${SAMPLES_PER_SHARD} samples/shard"

rm -f "${PROFILE_DIR}/multi_"*"train_n${MAX_TRAIN_SAMPLES}"*.jsonl
rm -f "${PROFILE_DIR}/multi_combined_train_n${MAX_TRAIN_SAMPLES}.jsonl"

run_profile direct 0 \
  --context_mode no_round_state \
  --cot_style step_by_step

run_profile reflective_r3 1 \
  --context_mode no_round_state \
  --chain_of_thoughts \
  --cot_style reflective_answer_first \
  --reflect_rounds 3

run_profile answer_first_no_caption 2 \
  --context_mode empty \
  --chain_of_thoughts \
  --cot_style answer_first_locked

run_profile marker_mcts 3 \
  --context_mode no_round_state \
  --cot_style step_by_step \
  --use_image_enhance \
  --mcts_n_simulations 5 \
  --mcts_trigger_mode count_color_object_only \
  --mcts_action_mode marker_only \
  --mcts_filter_objects

wait

"${PY}" "${REPO}/ablation_summary/build_strategy_rag_profile.py" \
  --profile "${PROFILE_DIR}/multi_direct_train_n${MAX_TRAIN_SAMPLES}_shard*.jsonl" \
  --profile "${PROFILE_DIR}/multi_reflective_r3_train_n${MAX_TRAIN_SAMPLES}_shard*.jsonl" \
  --profile "${PROFILE_DIR}/multi_answer_first_no_caption_train_n${MAX_TRAIN_SAMPLES}_shard*.jsonl" \
  --profile "${PROFILE_DIR}/multi_marker_mcts_train_n${MAX_TRAIN_SAMPLES}_shard*.jsonl" \
  --output "${PROFILE_DIR}/multi_combined_train_n${MAX_TRAIN_SAMPLES}.jsonl" \
  > "${LOG_DIR}/build_multi_combined_profile.log" 2>&1

echo "[multi-strategy] combined profile: ${PROFILE_DIR}/multi_combined_train_n${MAX_TRAIN_SAMPLES}.jsonl"
