#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-mme_dyfo_decision_6exp_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
PARQUET_PYTHON="/data2/lizhengxue/WorkSpace/onion/.venv_parquet/bin/python"
ENGINE="qwen3-VL-4B"
MME_ROOT="/data2/lizhengxue/datasets/mme"
COCO17_ANNO="/data2/lizhengxue/datasets/coco17/annotations"
FEATURE_ROOT="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data"
PREPARED_ROOT="/data2/lizhengxue/WorkSpace/onion_output/mme/prepared"
MME_MANIFEST="${PREPARED_ROOT}/test_manifest.jsonl"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/mme/dyfo_decision_test/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_mme_dyfo_decision_test/${RUN_ID}"
SHARD_LAUNCH_DELAY="${SHARD_LAUNCH_DELAY:-90}"

mkdir -p "${OUT_ROOT}" "${LOGDIR}" "${PREPARED_ROOT}"
cd "${REPO_DIR}"

if [[ ! -s "${MME_MANIFEST}" ]]; then
  echo "[$(date '+%F %T')] preparing MME manifest -> ${MME_MANIFEST}"
  "${PARQUET_PYTHON}" forward_code/prepare_mme_manifest.py \
    --mme_root "${MME_ROOT}" \
    --output_dir "${PREPARED_ROOT}" \
    --split_name test
fi

BASE_ARGS=(
  forward_code/onion.py
  --dataset_name mme
  --split_name test
  --engine "${ENGINE}"
  --raw_image_dir "${PREPARED_ROOT}"
  --coco_path "${MME_ROOT}"
  --mme_manifest_file "${MME_MANIFEST}"
  --coco_annotation_path "${COCO17_ANNO}"
  --caption_type vinvl_tag
  --valcaption_file "${FEATURE_ROOT}/input_text/vinvl_caption/VinVL_base_val2014.tsv"
  --tag_path "${FEATURE_ROOT}/input_text/coco_caption_pred_tags"
  --sg_path "${FEATURE_ROOT}/input_text/scene_graph_text"
  --concept_caption_path scene_graph_coco17_caption
  --similarity_path "${FEATURE_ROOT}/coco_clip_new"
  --similarity_metric imagequestion
  --train_sim_metric answer
  --context_mode no_round_state
  --ensemble_strategy first
  --answer_postprocess safe_rules
  --direct_prompt_style default
  --n_shot 0
  --n_ensemble 1
  --rounds 1
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
  local cache_dir="/data2/lizhengxue/WorkSpace/onion_output/mme/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
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
  local cache_dir="/data2/lizhengxue/WorkSpace/onion_output/mme/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
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
    if [[ "${shard_id}" != "2" ]]; then
      sleep "${SHARD_LAUNCH_DELAY}"
    fi
  done
  wait
  merge_exp "${exp_name}" "${gpu_id}"
}

run_exp direct_baseline 0 &
run_exp dyfo_evidence_inject 1 &
run_exp dyfo_weighted_vote 2 &
run_exp dyfo_best_focus_answer 3 &
run_exp dyfo_weighted_vote_more_search 4 &
run_exp dyfo_weighted_vote_visual_update 5 &
wait

cat <<EOF
RUN_ID=${RUN_ID}
OUT_ROOT=${OUT_ROOT}
LOGDIR=${LOGDIR}
MME_MANIFEST=${MME_MANIFEST}
EOF
