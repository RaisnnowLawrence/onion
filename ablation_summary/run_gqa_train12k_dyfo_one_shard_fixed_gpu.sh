#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <run_id> <shard_id> <gpu_id>" >&2
  exit 2
fi

RUN_ID="$1"
SHARD_ID="$2"
GPU_ID="$3"

REPO="/data2/lizhengxue/WorkSpace/onion"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
DATASET_ROOT="/data2/lizhengxue/datasets"
SUBSET_JSON="${GQA_QUESTION_FILE:-${REPO}/ablation_summary/router_data/gqa_train_balanced_seed42_12000_questions.json}"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/gqa"
LOG_ROOT="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_textvqa_gqa_queue"
EXP_OUT="${OUT_ROOT}/${RUN_ID}"
NUM_SHARDS="${NUM_SHARDS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-120}"
DYFO_DECISION_MODE="${DYFO_DECISION_MODE:-weighted_vote}"

mkdir -p "$LOG_ROOT" "$EXP_OUT"

gpu_free_mb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print $1 + 0; exit}'
}

while true; do
  free_mb="$(gpu_free_mb)"
  if [[ "$free_mb" -ge "$MIN_FREE_MB" ]]; then
    break
  fi
  echo "[gpu] gpu=${GPU_ID} free_mb=${free_mb} min_free_mb=${MIN_FREE_MB}; sleep=${GPU_WAIT_SECONDS}s"
  sleep "$GPU_WAIT_SECONDS"
done

COMMON_ARGS=(
  "$REPO/forward_code/onion.py"
  --dataset_name gqa
  --split_name train
  --engine qwen3-VL-4B
  --caption_type vinvl
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --dataset_root "$DATASET_ROOT"
  --coco_annotation_path /data2/lizhengxue/datasets/coco14/annotations
  --aokvqa_context_path /data2/lizhengxue/datasets/aokvqa
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
  --train_sim_metric answer
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
  --output_path "$EXP_OUT"
  --cache_path "${EXP_OUT}/cache_shard${SHARD_ID}"
  --num_shards "$NUM_SHARDS"
  --gqa_question_file "$SUBSET_JSON"
  --context_mode no_round_state
  --use_image_enhance
  --mcts_action_mode dyfo_evidence
  --use_dyfo_visual_evidence
  --dyfo_decision_mode "$DYFO_DECISION_MODE"
  --dyfo_trigger_mode visual_detail
  --dyfo_n_simulations 6
  --dyfo_max_depth 3
  --dyfo_area_reward compact
)

LOG_FILE="${LOG_ROOT}/${RUN_ID}_gpu${GPU_ID}_shard${SHARD_ID}.log"
echo "[run] RUN_ID=${RUN_ID}"
echo "[run] GPU_ID=${GPU_ID} SHARD_ID=${SHARD_ID}/${NUM_SHARDS}"
echo "[run] EXP_OUT=${EXP_OUT}"
echo "[run] GQA_QUESTION_FILE=${SUBSET_JSON}"
echo "[run] DYFO_DECISION_MODE=${DYFO_DECISION_MODE}"
echo "[run] LOG_FILE=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" "${COMMON_ARGS[@]}" --shard_id "$SHARD_ID" > "$LOG_FILE" 2>&1
