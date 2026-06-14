#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_reflective_followup5_3shards

mkdir -p "${LOG_DIR}"

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  shift 3
  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${BASE}/forward_code/onion.py" \
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
    --chain_of_thoughts \
    --direct_verify_policy conflict_only \
    --reflect_rounds 3 \
    "$@" \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${name}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  shift
  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"

  CUDA_VISIBLE_DEVICES="" "${PY}" "${BASE}/forward_code/onion.py" \
    --merge_only \
    --engine "${ENGINE}" \
    --caption_type vinvl \
    --output_path "${out}" \
    --cache_path "${cache}" \
    --sg_path "${DATA_ROOT}/input_text/scene_graph_text" \
    --n_shot 1 \
    --n_ensemble 1 \
    --rounds 1 \
    --context_mode no_round_state \
    --chain_of_thoughts \
    --direct_verify_policy conflict_only \
    --reflect_rounds 3 \
    "$@" \
    --summary_log "${out}/accuracy.log" \
    > "${LOG_DIR}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"
  shift 2

  run_shard "${name}" "${gpu}" 0 "$@" &
  run_shard "${name}" "${gpu}" 1 "$@" &
  run_shard "${name}" "${gpu}" 2 "$@" &
  wait
  merge_exp "${name}" "$@"
}

case "${1:-}" in
  adaptive)
    run_exp_3shards reflective_adaptive_highrisk_lowconf_caption_r3_3shards_gpu3 3 \
      --cot_style adaptive_reflective_answer_first \
      --reflect_trigger_mode high_risk_or_low_confidence
    ;;
  keep_revise)
    run_exp_3shards reflective_keep_revise_caption_r3_3shards_gpu4 4 \
      --cot_style reflective_answer_first \
      --reflect_review_format keep_revise
    ;;
  visible_only)
    run_exp_3shards reflective_visible_only_caption_r3_3shards_gpu5 5 \
      --cot_style reflective_answer_first \
      --reflect_evidence_mode visible_only
    ;;
  review_empty)
    run_exp_3shards reflective_review_empty_context_caption_r3_3shards_gpu6 6 \
      --cot_style reflective_answer_first \
      --reflect_review_context empty
    ;;
  initial_ensemble3)
    run_exp_3shards reflective_initial_ensemble3_keep_revise_caption_r3_3shards_gpu7 7 \
      --cot_style reflective_answer_first \
      --reflect_initial_ensemble 3 \
      --reflect_review_format keep_revise
    ;;
  *)
    echo "usage: $0 {adaptive|keep_revise|visible_only|review_empty|initial_ensemble3}" >&2
    exit 2
    ;;
esac
