#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PY=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
DATA_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
LOG_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_reviewer_evidence_selective_3shards

GPU="${1:-4}"
NAME=reviewer_evidence_selective_rounds1_3shards_gpu${GPU}
OUT="${OUT_ROOT}/${ENGINE}_forward2_${NAME}"
CACHE="${DATA_ROOT}/image_cache_onion/cache_${NAME}"

mkdir -p "${LOG_DIR}"

run_shard() {
  local shard="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${BASE}/forward_code/onion.py" \
    --cache_path "${CACHE}" \
    --output_path "${OUT}" \
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
    --reviewer_evidence_scope selective \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${NAME}_shard${shard}.log" 2>&1
}

run_shard 0 &
run_shard 1 &
run_shard 2 &
wait

CUDA_VISIBLE_DEVICES="" "${PY}" "${BASE}/forward_code/onion.py" \
  --merge_only \
  --engine "${ENGINE}" \
  --caption_type vinvl \
  --output_path "${OUT}" \
  --cache_path "${CACHE}" \
  --sg_path "${DATA_ROOT}/input_text/scene_graph_text" \
  --n_shot 1 \
  --n_ensemble 1 \
  --rounds 1 \
  --chain_of_thoughts \
  --cot_style reviewer_evidence \
  --direct_verify_policy conflict_only \
  --context_mode no_round_state \
  --reviewer_evidence_scope selective \
  --summary_log "${OUT}/accuracy.log" \
  > "${LOG_DIR}/${NAME}_merge.log" 2>&1

echo "selective reviewer evidence finished: ${OUT}"
