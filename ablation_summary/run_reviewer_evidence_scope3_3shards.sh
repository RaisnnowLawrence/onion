#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_reviewer_evidence_scope3_3shards

mkdir -p "${LOG_DIR}"

run_shard() {
  local name="$1"
  local gpu="$2"
  local scope="$3"
  local shard="$4"
  local extra_args="$5"
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
    --chain_of_thoughts \
    --cot_style reviewer_evidence \
    --direct_verify_policy conflict_only \
    --context_mode no_round_state \
    --reviewer_evidence_scope "${scope}" \
    ${extra_args} \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${name}_shard${shard}.log" 2>&1
}

merge_exp() {
  local name="$1"
  local scope="$2"
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
    --chain_of_thoughts \
    --cot_style reviewer_evidence \
    --direct_verify_policy conflict_only \
    --context_mode no_round_state \
    --reviewer_evidence_scope "${scope}" \
    --summary_log "${out}/accuracy.log" \
    > "${LOG_DIR}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"
  local scope="$3"
  local extra_args="$4"

  run_shard "${name}" "${gpu}" "${scope}" 0 "${extra_args}" &
  run_shard "${name}" "${gpu}" "${scope}" 1 "${extra_args}" &
  run_shard "${name}" "${gpu}" "${scope}" 2 "${extra_args}" &
  wait
  merge_exp "${name}" "${scope}"
}

case "${1:-}" in
  caption_only)
    run_exp_3shards reviewer_evidence_caption_only_rounds1_3shards_gpu4 4 caption_only ""
    ;;
  no_objects)
    run_exp_3shards reviewer_evidence_no_objects_rounds1_3shards_gpu5 5 no_objects "--use_caption_enhance --use_knowledge_enhance"
    ;;
  caption_object)
    run_exp_3shards reviewer_evidence_caption_object_rounds1_3shards_gpu6 6 caption_object ""
    ;;
  *)
    echo "usage: $0 {caption_only|no_objects|caption_object}" >&2
    exit 2
    ;;
esac
