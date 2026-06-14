#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_pure_cot_direct_4exp_3shards

mkdir -p "${LOG_DIR}"

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  local cot_style="$4"
  local context_mode="$5"
  local chain_flag="$6"
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
    ${chain_flag} \
    --cot_style "${cot_style}" \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${name}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  local cot_style="$2"
  local context_mode="$3"
  local chain_flag="$4"
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
    ${chain_flag} \
    --cot_style "${cot_style}" \
    --summary_log "${out}/accuracy.log" \
    > "${LOG_DIR}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"
  local cot_style="$3"
  local context_mode="$4"
  local chain_flag="$5"

  run_shard "${name}" "${gpu}" 0 "${cot_style}" "${context_mode}" "${chain_flag}" &
  run_shard "${name}" "${gpu}" 1 "${cot_style}" "${context_mode}" "${chain_flag}" &
  run_shard "${name}" "${gpu}" 2 "${cot_style}" "${context_mode}" "${chain_flag}" &
  wait
  merge_exp "${name}" "${cot_style}" "${context_mode}" "${chain_flag}"
}

case "${1:-}" in
  answer_first_locked)
    run_exp_3shards pure_cot_answer_first_locked_rounds1_3shards_gpu4 4 answer_first_locked no_round_state "--chain_of_thoughts"
    ;;
  visual_facts)
    run_exp_3shards pure_cot_visual_facts_rounds1_3shards_gpu5 5 visual_facts no_round_state "--chain_of_thoughts"
    ;;
  visual_facts_no_caption)
    run_exp_3shards pure_cot_visual_facts_no_caption_rounds1_3shards_gpu6 6 visual_facts empty "--chain_of_thoughts"
    ;;
  direct_image_question)
    run_exp_3shards direct_image_question_only_rounds1_3shards_gpu7 7 step_by_step empty ""
    ;;
  *)
    echo "usage: $0 {answer_first_locked|visual_facts|visual_facts_no_caption|direct_image_question}" >&2
    exit 2
    ;;
esac
