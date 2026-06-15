#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
PROFILE_DIR=${PROFILE_DIR:-${REPORT_DIR}/strategy_rag_profiles}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_multi_strategy_router_val}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-400}
PROFILE_PATH=${PROFILE_PATH:-${PROFILE_DIR}/multi_combined_train_n${MAX_TRAIN_SAMPLES}.jsonl}

mkdir -p "${LOG_DIR}"

EXP_NAME="multi_strategy_router_train${MAX_TRAIN_SAMPLES}_val_3shards"
OUT="${OUT_ROOT}/${ENGINE}_forward2_${EXP_NAME}"
CACHE="${DATA_ROOT}/image_cache_onion/cache_${EXP_NAME}"
mkdir -p "${OUT}" "${CACHE}"

common_args=(
  --cache_path "${CACHE}"
  --output_path "${OUT}"
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
  --chain_of_thoughts
  --cot_style multi_strategy_router
  --strategy_profile_path "${PROFILE_PATH}"
  --strategy_retrieval_metric imagequestion
  --strategy_topk 20
  --strategy_min_neighbors 5
  --multi_strategy_names direct,reflective_r3,answer_first_no_caption,marker_mcts
  --multi_strategy_default direct
  --multi_strategy_margin 0.08
  --use_image_enhance
  --mcts_n_simulations 5
  --mcts_trigger_mode count_color_object_only
  --mcts_action_mode marker_only
  --mcts_filter_objects
)

run_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib_multi_strategy \
  "${PY}" "${REPO}/forward_code/onion.py" \
    "${common_args[@]}" \
    --shard_id "${shard}" --num_shards 3 \
    > "${LOG_DIR}/${EXP_NAME}_gpu${gpu}_shard${shard}.log" 2>&1
}

echo "[multi-strategy] launch ${EXP_NAME} with profile ${PROFILE_PATH}"
run_shard 0 0 &
run_shard 1 1 &
run_shard 2 2 &
wait

CUDA_VISIBLE_DEVICES="" "${PY}" "${REPO}/forward_code/onion.py" \
  --merge_only \
  "${common_args[@]}" \
  --summary_log "${OUT}/accuracy.log" \
  > "${LOG_DIR}/${EXP_NAME}_merge.log" 2>&1

echo "[multi-strategy] merged ${OUT}/accuracy.log"

