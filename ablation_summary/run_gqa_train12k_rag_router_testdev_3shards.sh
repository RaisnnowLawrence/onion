#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data2/lizhengxue/WorkSpace/onion}
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
DATASET_ROOT=${DATASET_ROOT:-/data2/lizhengxue/datasets}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/gqa}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_gqa_train12k_rag_router_testdev}
PROFILE_PATH=${PROFILE_PATH:-${REPO}/ablation_summary/router_data/gqa_train12k_router_20260708_135230/gqa_train12k_router_training_table.jsonl}
GQA_QUESTION_FILE=${GQA_QUESTION_FILE:-/data2/lizhengxue/datasets/gqa/testdev_balanced_questions.json}
RUN_ID=${RUN_ID:-gqa_testdev_train12k_rag_router_pure_vs_dyfo_$(date +%Y%m%d_%H%M%S)}
NUM_SHARDS=${NUM_SHARDS:-3}
GPUS=(${GPUS:-0 2 3})
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-}

mkdir -p "${LOG_DIR}"
OUT="${OUT_ROOT}/${RUN_ID}"
CACHE="${OUT}/cache"
mkdir -p "${OUT}" "${CACHE}"

common_args=(
  "${REPO}/forward_code/onion.py"
  --dataset_name gqa
  --split_name testdev
  --gqa_question_file "${GQA_QUESTION_FILE}"
  --engine "${ENGINE}"
  --caption_type vinvl
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --dataset_root "${DATASET_ROOT}"
  --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations
  --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
  --train_sim_metric answer
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
  --output_path "${OUT}"
  --cache_path "${CACHE}"
  --context_mode empty
  --remove_caption
  --use_image_enhance
  --mcts_action_mode dyfo_evidence
  --use_dyfo_visual_evidence
  --dyfo_decision_mode weighted_vote
  --dyfo_trigger_mode visual_detail
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
  --chain_of_thoughts
  --cot_style rag_strategy_router
  --strategy_profile_path "${PROFILE_PATH}"
  --strategy_direct_name pure
  --strategy_cot_name dyfo
  --strategy_cot_runtime dyfo_evidence
  --strategy_router_default pure
  --strategy_router_mode conservative_risk
  --strategy_retrieval_metric imagequestion
  --strategy_topk 20
  --strategy_min_neighbors 5
  --strategy_margin 0.005
  --strategy_min_net_gain 0.06
  --strategy_max_damage_rate 0.18
  --strategy_direct_hard_threshold 0.0
  --strategy_direct_safe_threshold 0.3333333333333333
  --num_shards "${NUM_SHARDS}"
)

if [[ -n "${MAX_SAMPLES_PER_SHARD}" ]]; then
  common_args+=(--max_samples_per_shard "${MAX_SAMPLES_PER_SHARD}")
fi

run_shard() {
  local shard="$1"
  local gpu="${GPUS[$shard]}"
  local log="${LOG_DIR}/${RUN_ID}_gpu${gpu}_shard${shard}.log"
  echo "[router-gqa] start shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"     "${PY}" "${common_args[@]}" --shard_id "${shard}" > "${log}" 2>&1 &
}

echo "[router-gqa] RUN_ID=${RUN_ID}"
echo "[router-gqa] PROFILE_PATH=${PROFILE_PATH}"
echo "[router-gqa] GQA_QUESTION_FILE=${GQA_QUESTION_FILE}"
echo "[router-gqa] OUT=${OUT}"
echo "[router-gqa] GPUS=${GPUS[*]}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "${shard}"
done
wait

merge_log="${LOG_DIR}/${RUN_ID}_merge.log"
CUDA_VISIBLE_DEVICES="" "${PY}" "${common_args[@]}" --merge_only --summary_log "${OUT}/summary.log" > "${merge_log}" 2>&1

echo "[router-gqa] merged ${OUT}/summary.log"
