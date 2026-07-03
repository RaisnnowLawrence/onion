#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-hallusionbench_dyfo_4exp_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
ENGINE="qwen3-VL-4B"
HALLUSION_ROOT="/data2/lizhengxue/datasets/hallusionbench"
COCO17_ANNO="/data2/lizhengxue/datasets/coco17/annotations"
FEATURE_ROOT="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/hallusionbench/dyfo_4exp/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_hallusionbench_dyfo_4exp/${RUN_ID}"

mkdir -p "${OUT_ROOT}" "${LOGDIR}"
cd "${REPO_DIR}"

BASE_ARGS=(
  forward_code/onion.py
  --dataset_name hallusionbench
  --split_name all
  --engine "${ENGINE}"
  --raw_image_dir "${HALLUSION_ROOT}"
  --coco_path "${HALLUSION_ROOT}"
  --coco_annotation_path "${COCO17_ANNO}"
  --caption_type vinvl
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --iterative_strategy caption
  --sg_path "${FEATURE_ROOT}/input_text/scene_graph_text"
  --similarity_metric random
  --train_sim_metric answer
  --context_mode no_round_state
  --ensemble_strategy first
  --answer_postprocess safe_rules
  --direct_prompt_style default
  --num_shards 1
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
      echo "--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence --dyfo_decision_mode best_focus_answer --dyfo_trigger_mode visual_detail --dyfo_n_simulations 6 --dyfo_max_depth 3 --dyfo_area_reward compact --dyfo_use_focus_image_as_answer"
      ;;
    *)
      echo "unknown experiment: $1" >&2
      return 1
      ;;
  esac
}

run_exp() {
  local exp_name="$1"
  local gpu_id="$2"
  local out_dir="${OUT_ROOT}/${ENGINE}_${exp_name}"
  local cache_dir="${FEATURE_ROOT}/image_cache_onion/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
  local log_file="${LOGDIR}/${exp_name}_gpu${gpu_id}.log"
  local merge_log="${LOGDIR}/${exp_name}_merge.log"
  local extra
  extra="$(extra_args_for_exp "${exp_name}")"

  mkdir -p "${out_dir}" "${cache_dir}"
  echo "[$(date '+%F %T')] start exp=${exp_name} shard=0/1 gpu=${gpu_id} log=${log_file}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --shard_id 0 \
    ${extra} \
    > "${log_file}" 2>&1

  echo "[$(date '+%F %T')] merge exp=${exp_name} -> ${out_dir}/accuracy.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --merge_only \
    --summary_log "${out_dir}/accuracy.log" \
    ${extra} \
    > "${merge_log}" 2>&1
}

run_exp direct_baseline 4 &
run_exp dyfo_evidence_inject 4 &
run_exp dyfo_weighted_vote 5 &
run_exp dyfo_best_focus_answer 5 &
wait

cat <<EOF
RUN_ID=${RUN_ID}
OUT_ROOT=${OUT_ROOT}
LOGDIR=${LOGDIR}
EOF
