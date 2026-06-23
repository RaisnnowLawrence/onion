#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-aokvqa_dyfo_evidence_val_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
ENGINE="qwen3-VL-4B"
DATA_ROOT="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data"
COCO14_ROOT="/data2/lizhengxue/datasets/coco14"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/aokvqa/dyfo_evidence_val/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_aokvqa_dyfo_evidence_val/${RUN_ID}"
EXP_NAME="dyfo_evidence_direct"
OUT_DIR="${OUT_ROOT}/${ENGINE}_${EXP_NAME}"
CACHE_DIR="${DATA_ROOT}/image_cache_onion/cache_${RUN_ID}_${EXP_NAME}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}" "${LOGDIR}"

COMMON_ARGS=(
  forward_code/onion.py
  --dataset_name aokvqa
  --split_name val
  --engine "${ENGINE}"
  --raw_image_dir "${COCO14_ROOT}"
  --cache_path "${CACHE_DIR}"
  --output_path "${OUT_DIR}"
  --caption_type vinvl
  --n_shot 1
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --sg_path "${DATA_ROOT}/input_text/scene_graph_text"
  --train_sim_metric answer
  --train_sim_file "${DATA_ROOT}/input_text/scene_graph_text/train_object_select_answer.pk"
  --context_mode no_round_state
  --ensemble_strategy first
  --answer_postprocess safe_rules
  --direct_prompt_style default
  --use_image_enhance
  --mcts_action_mode dyfo_evidence
  --use_dyfo_visual_evidence
  --dyfo_trigger_mode visual_detail
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
  --num_shards 6
)

run_shard() {
  local shard_id="$1"
  local gpu_id="$2"
  local log_file="${LOGDIR}/${EXP_NAME}_shard${shard_id}_gpu${gpu_id}.log"
  echo "[$(date '+%F %T')] start shard=${shard_id} gpu=${gpu_id} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${PYTHON}" "${COMMON_ARGS[@]}" \
    --shard_id "${shard_id}" \
    > "${log_file}" 2>&1
}

merge_exp() {
  local log_file="${LOGDIR}/${EXP_NAME}_merge.log"
  echo "[$(date '+%F %T')] merge ${EXP_NAME} -> ${OUT_DIR}/accuracy.log"
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "${PYTHON}" "${COMMON_ARGS[@]}" \
    --merge_only \
    --summary_log "${OUT_DIR}/accuracy.log" \
    > "${log_file}" 2>&1
}

run_shard 0 2 &
run_shard 1 2 &
run_shard 2 2 &
run_shard 3 3 &
run_shard 4 3 &
run_shard 5 3 &
wait
merge_exp

cat <<EOF
RUN_ID=${RUN_ID}
OUT_DIR=${OUT_DIR}
LOGDIR=${LOGDIR}
accuracy=${OUT_DIR}/accuracy.log
EOF
