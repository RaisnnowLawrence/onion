#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
PROFILE_DIR=${PROFILE_DIR:-${REPORT_DIR}/strategy_rag_profiles}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_strategy_rag_profile_slice}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-400}

mkdir -p "${PROFILE_DIR}" "${LOG_DIR}"

common_args=(
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
  --context_mode no_round_state
  --split_name train
  --max_samples_per_shard "${MAX_TRAIN_SAMPLES}"
)

run_train_profile() {
  local strategy="$1"
  local gpu="$2"
  local chain_flag="$3"
  local cot_style="$4"

  local out="${OUT_ROOT}/${ENGINE}_strategy_profile_train_${strategy}_n${MAX_TRAIN_SAMPLES}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_strategy_profile_train_${strategy}_n${MAX_TRAIN_SAMPLES}"
  local profile="${PROFILE_DIR}/${strategy}_train_n${MAX_TRAIN_SAMPLES}.jsonl"
  rm -f "${profile}"
  mkdir -p "${out}" "${cache}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    ${chain_flag} \
    --cot_style "${cot_style}" \
    --strategy_name "${strategy}" \
    --strategy_profile_output "${profile}" \
    > "${LOG_DIR}/${strategy}_gpu${gpu}.log" 2>&1
}

echo "[strategy-rag] train profile slice: ${MAX_TRAIN_SAMPLES} samples per strategy"
run_train_profile direct 0 "" step_by_step &
run_train_profile protected_reflective 1 "--chain_of_thoughts" protected_reflective &
run_train_profile answer_first_locked 2 "--chain_of_thoughts" answer_first_locked &
wait

"${PY}" "${REPO}/ablation_summary/build_strategy_rag_profile.py" \
  --profile "${PROFILE_DIR}/direct_train_n${MAX_TRAIN_SAMPLES}.jsonl" \
  --profile "${PROFILE_DIR}/protected_reflective_train_n${MAX_TRAIN_SAMPLES}.jsonl" \
  --profile "${PROFILE_DIR}/answer_first_locked_train_n${MAX_TRAIN_SAMPLES}.jsonl" \
  --output "${PROFILE_DIR}/combined_train_n${MAX_TRAIN_SAMPLES}.jsonl" \
  > "${LOG_DIR}/build_combined_profile.log" 2>&1

echo "[strategy-rag] combined profile: ${PROFILE_DIR}/combined_train_n${MAX_TRAIN_SAMPLES}.jsonl"

