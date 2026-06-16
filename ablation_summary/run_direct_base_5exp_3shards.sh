#!/usr/bin/env bash
set -euo pipefail

PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
CODE=/data2/lizhengxue/WorkSpace/onion/forward_code/onion.py
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa/direct_base_5exp
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_ROOT=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_direct_base_5exp_3shards
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_ROOT}"

common_args() {
  local name="$1"
  local out="${OUT_ROOT}/${RUN_ID}/${ENGINE}_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_direct_base_5exp_${RUN_ID}_${name}"

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
    --ensemble_strategy first
}

extra_args_for() {
  local name="$1"
  case "${name}" in
    direct_baseline_no_cot_rounds1)
      echo --answer_postprocess none --direct_prompt_style default
      ;;
    direct_safe_postprocess)
      echo --answer_postprocess safe_rules --direct_prompt_style default
      ;;
    direct_answer_first_strict)
      echo --answer_postprocess safe_rules --direct_prompt_style answer_first_strict
      ;;
    direct_type_specialist)
      echo --answer_postprocess safe_rules --direct_prompt_style type_specialist
      ;;
    direct_context_gated)
      echo --answer_postprocess safe_rules --direct_prompt_style context_gated --use_ocr_context --use_all_regional_captions --max_regional_captions 25
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
  mkdir -p "${log_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${CODE}" \
    $(common_args "${name}") \
    $(extra_args_for "${name}") \
    --shard_id "${shard}" \
    --num_shards 3 \
    > "${log_dir}/${name}_gpu${gpu}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  local gpu="$2"
  local log_dir="${LOG_ROOT}/${RUN_ID}"
  local out="${OUT_ROOT}/${RUN_ID}/${ENGINE}_${name}"
  mkdir -p "${log_dir}"

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

  echo "[direct-base] start ${name} on GPU ${gpu}, RUN_ID=${RUN_ID}"
  run_shard "${name}" "${gpu}" 0 &
  run_shard "${name}" "${gpu}" 1 &
  run_shard "${name}" "${gpu}" 2 &
  wait
  merge_exp "${name}" "${gpu}"
  echo "[direct-base] done ${name}"
}

case "${1:-}" in
  direct_baseline_no_cot_rounds1|direct_safe_postprocess|direct_answer_first_strict|direct_type_specialist|direct_context_gated)
    run_exp_3shards "$1" "${2:?gpu id required}"
    ;;
  *)
    echo "usage: RUN_ID=<id> $0 {direct_baseline_no_cot_rounds1|direct_safe_postprocess|direct_answer_first_strict|direct_type_specialist|direct_context_gated} <gpu>" >&2
    exit 2
    ;;
esac
