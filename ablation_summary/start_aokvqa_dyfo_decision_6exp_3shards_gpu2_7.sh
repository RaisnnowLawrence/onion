#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-aokvqa_dyfo_decision_6exp_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
ENGINE="qwen3-VL-4B"
FEATURE_ROOT="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data"
AOKVQA_ROOT="/data2/lizhengxue/datasets/aokvqa"
COCO17_ANNO="/data2/lizhengxue/datasets/coco17/annotations"
COCO14_ROOT="/data2/lizhengxue/datasets/coco14"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/aokvqa/dyfo_decision_val/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_aokvqa_dyfo_decision_val/${RUN_ID}"

mkdir -p "${OUT_ROOT}" "${LOGDIR}"
cd "${REPO_DIR}"

BASE_ARGS=(
  forward_code/onion.py
  --dataset_name aokvqa
  --split_name val
  --engine "${ENGINE}"
  --raw_image_dir "${COCO14_ROOT}"
  --coco_path "${AOKVQA_ROOT}"
  --coco_annotation_path "${COCO17_ANNO}"
  --caption_type vinvl
  --n_shot 1
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --sg_path "${FEATURE_ROOT}/input_text/scene_graph_text"
  --train_sim_metric answer
  --train_sim_file "${FEATURE_ROOT}/input_text/scene_graph_text/train_object_select_answer.pk"
  --context_mode no_round_state
  --ensemble_strategy first
  --answer_postprocess safe_rules
  --direct_prompt_style default
  --num_shards 3
)

extra_args_for_exp() {
  case "$1" in
    direct_baseline)
      echo ""
      ;;
    dyfo_evidence_inject)
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode evidence_inject --dyfo_trigger_mode visual_detail --dyfo_n_simulations 6 --dyfo_max_depth 3 --dyfo_area_reward compact"
      ;;
    dyfo_weighted_vote)
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode weighted_vote --dyfo_trigger_mode visual_detail --dyfo_n_simulations 6 --dyfo_max_depth 3 --dyfo_area_reward compact"
      ;;
    dyfo_best_focus_answer)
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode best_focus_answer --dyfo_trigger_mode visual_detail --dyfo_n_simulations 6 --dyfo_max_depth 3 --dyfo_area_reward compact"
      ;;
    dyfo_weighted_vote_more_search)
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode weighted_vote --dyfo_trigger_mode visual_detail --dyfo_n_simulations 10 --dyfo_max_depth 4 --dyfo_area_reward compact"
      ;;
    dyfo_weighted_vote_visual_update)
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode weighted_vote --dyfo_trigger_mode visual_detail --dyfo_n_simulations 6 --dyfo_max_depth 3 --dyfo_area_reward compact --dyfo_text_focus_use_image"
      ;;
    *)
      echo "unknown experiment: $1" >&2
      return 1
      ;;
  esac
}

run_shard() {
  local exp_name="$1"
  local gpu_id="$2"
  local shard_id="$3"
  local out_dir="${OUT_ROOT}/${ENGINE}_${exp_name}"
  local cache_dir="${FEATURE_ROOT}/image_cache_onion/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
  local log_file="${LOGDIR}/${exp_name}_shard${shard_id}_gpu${gpu_id}.log"
  local extra
  extra="$(extra_args_for_exp "${exp_name}")"

  mkdir -p "${out_dir}" "${cache_dir}"
  echo "[$(date '+%F %T')] start exp=${exp_name} shard=${shard_id}/3 gpu=${gpu_id} log=${log_file}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --shard_id "${shard_id}" \
    ${extra} \
    > "${log_file}" 2>&1
}

merge_exp() {
  local exp_name="$1"
  local gpu_id="$2"
  local out_dir="${OUT_ROOT}/${ENGINE}_${exp_name}"
  local cache_dir="${FEATURE_ROOT}/image_cache_onion/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
  local log_file="${LOGDIR}/${exp_name}_merge.log"
  local extra
  extra="$(extra_args_for_exp "${exp_name}")"

  echo "[$(date '+%F %T')] merge exp=${exp_name} -> ${out_dir}/accuracy.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --merge_only \
    --summary_log "${out_dir}/accuracy.log" \
    ${extra} \
    > "${log_file}" 2>&1
}

run_exp() {
  local exp_name="$1"
  local gpu_id="$2"
  for shard_id in 0 1 2; do
    run_shard "${exp_name}" "${gpu_id}" "${shard_id}" &
  done
  wait
  merge_exp "${exp_name}" "${gpu_id}"
}

run_exp direct_baseline 2 &
run_exp dyfo_evidence_inject 3 &
run_exp dyfo_weighted_vote 4 &
run_exp dyfo_best_focus_answer 5 &
run_exp dyfo_weighted_vote_more_search 6 &
run_exp dyfo_weighted_vote_visual_update 7 &
wait

cat <<EOF
RUN_ID=${RUN_ID}
OUT_ROOT=${OUT_ROOT}
LOGDIR=${LOGDIR}
EOF
