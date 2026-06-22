#!/usr/bin/env bash
set -euo pipefail

PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
CODE=/data2/lizhengxue/WorkSpace/onion/forward_code/onion.py
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa/rephrase_consistency_8exp
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_ROOT=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_rephrase_consistency_8exp_3shards
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_ROOT}/${RUN_ID}"

common_args() {
  local name="$1"
  local out="${OUT_ROOT}/${RUN_ID}/${ENGINE}_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_rephrase_consistency_${RUN_ID}_${name}"

  echo \
    --cache_path "${cache}" \
    --output_path "${out}" \
    --caption_type vinvl \
    --n_shot 1 \
    --n_ensemble 1 \
    --rounds 1 \
    --iterative_strategy caption \
    --engine "${ENGINE}" \
    --sg_path "${DATA_ROOT}/input_text/scene_graph_text" \
    --train_sim_metric answer \
    --train_sim_file "${DATA_ROOT}/input_text/scene_graph_text/train_object_select_answer.pk" \
    --context_mode no_round_state \
    --ensemble_strategy first \
    --answer_postprocess safe_rules \
    --chain_of_thoughts \
    --cot_style direct_rephrase_consistency \
    --rephrase_num_questions 3 \
    --rephrase_consensus_threshold 2 \
    --rephrase_generation_max_tokens 128 \
    --rephrase_answer_max_tokens 16 \
    --rephrase_review_max_tokens 96
}

extra_args_for() {
  local name="$1"
  case "${name}" in
    rephrase_keep_trace_mixed)
      echo --rephrase_generation_mode mixed --rephrase_arbitration keep_baseline --rephrase_answer_context same --rephrase_trigger always
      ;;
    rephrase_majority2_mixed)
      echo --rephrase_generation_mode mixed --rephrase_arbitration majority_if_consensus --rephrase_answer_context same --rephrase_trigger always
      ;;
    rephrase_review2_mixed)
      echo --rephrase_generation_mode mixed --rephrase_arbitration conservative_review --rephrase_answer_context same --rephrase_trigger always
      ;;
    rephrase_allagree_mixed)
      echo --rephrase_generation_mode mixed --rephrase_arbitration all_agree --rephrase_answer_context same --rephrase_trigger always --rephrase_consensus_threshold 3
      ;;
    rephrase_review2_visual_focus)
      echo --rephrase_generation_mode visual_focus --rephrase_arbitration conservative_review --rephrase_answer_context same --rephrase_trigger always
      ;;
    rephrase_review2_answer_type)
      echo --rephrase_generation_mode answer_type --rephrase_arbitration conservative_review --rephrase_answer_context same --rephrase_trigger always
      ;;
    rephrase_review2_regional)
      echo --rephrase_generation_mode mixed --rephrase_arbitration conservative_review --rephrase_answer_context regional --rephrase_trigger always --use_all_regional_captions --max_regional_captions 25
      ;;
    rephrase_review2_risky_only)
      echo --rephrase_generation_mode mixed --rephrase_arbitration conservative_review --rephrase_answer_context same --rephrase_trigger risky_qtype
      ;;
    *)
      echo "unknown experiment: ${name}" >&2
      exit 2
      ;;
  esac
}

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  local log_dir="${LOG_ROOT}/${RUN_ID}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${CODE}" \
    $(common_args "${name}") \
    $(extra_args_for "${name}") \
    --shard_id "${shard}" \
    --num_shards 3 \
    > "${log_dir}/${name}_gpu${gpu}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  local out="${OUT_ROOT}/${RUN_ID}/${ENGINE}_${name}"
  local log_dir="${LOG_ROOT}/${RUN_ID}"

  CUDA_VISIBLE_DEVICES="" "${PY}" "${CODE}" \
    --merge_only \
    $(common_args "${name}") \
    $(extra_args_for "${name}") \
    --summary_log "${out}/accuracy.log" \
    > "${log_dir}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"

  echo "[rephrase] start ${name} on GPU ${gpu}, RUN_ID=${RUN_ID}"
  run_shard "${name}" "${gpu}" 0 &
  run_shard "${name}" "${gpu}" 1 &
  run_shard "${name}" "${gpu}" 2 &
  wait
  merge_exp "${name}"
  echo "[rephrase] done ${name}"
}

case "${1:-}" in
  rephrase_keep_trace_mixed|rephrase_majority2_mixed|rephrase_review2_mixed|rephrase_allagree_mixed|rephrase_review2_visual_focus|rephrase_review2_answer_type|rephrase_review2_regional|rephrase_review2_risky_only)
    run_exp_3shards "$1" "${2:?gpu id required}"
    ;;
  *)
    echo "usage: RUN_ID=<id> $0 <experiment> <gpu>" >&2
    exit 2
    ;;
esac
