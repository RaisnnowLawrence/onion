#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_reflective_answer_first_4exp_3shards

mkdir -p "${LOG_DIR}"

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  local cot_style="$4"
  local context_mode="$5"
  local reflect_rounds="$6"
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
    --context_mode "${context_mode}" \
    --chain_of_thoughts \
    --cot_style "${cot_style}" \
    --reflect_rounds "${reflect_rounds}" \
    --direct_verify_policy conflict_only \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${name}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  local cot_style="$2"
  local context_mode="$3"
  local reflect_rounds="$4"
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
    --context_mode "${context_mode}" \
    --chain_of_thoughts \
    --cot_style "${cot_style}" \
    --reflect_rounds "${reflect_rounds}" \
    --direct_verify_policy conflict_only \
    --summary_log "${out}/accuracy.log" \
    > "${LOG_DIR}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"
  local cot_style="$3"
  local context_mode="$4"
  local reflect_rounds="$5"

  run_shard "${name}" "${gpu}" 0 "${cot_style}" "${context_mode}" "${reflect_rounds}" &
  run_shard "${name}" "${gpu}" 1 "${cot_style}" "${context_mode}" "${reflect_rounds}" &
  run_shard "${name}" "${gpu}" 2 "${cot_style}" "${context_mode}" "${reflect_rounds}" &
  wait
  merge_exp "${name}" "${cot_style}" "${context_mode}" "${reflect_rounds}"
}

case "${1:-}" in
  reflective_caption_r3)
    run_exp_3shards reflective_answer_first_caption_r3_rounds1_3shards_gpu4 4 reflective_answer_first no_round_state 3
    ;;
  reflective_no_caption_r3)
    run_exp_3shards reflective_answer_first_no_caption_r3_rounds1_3shards_gpu5 5 reflective_answer_first empty 3
    ;;
  answer_first_locked_no_caption)
    run_exp_3shards answer_first_locked_no_caption_rounds1_3shards_gpu6 6 answer_first_locked empty 3
    ;;
  reflective_caption_r5)
    run_exp_3shards reflective_answer_first_caption_r5_rounds1_3shards_gpu7 7 reflective_answer_first no_round_state 5
    ;;
  *)
    echo "usage: $0 {reflective_caption_r3|reflective_no_caption_r3|answer_first_locked_no_caption|reflective_caption_r5}" >&2
    exit 2
    ;;
esac
