#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_knowledge4_x_dyfo4_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
ENGINE="qwen3-VL-4B"
OKVQA_ROOT="/data2/lizhengxue/datasets/okvqa"
COCO14_ROOT="/data2/lizhengxue/datasets/coco14"
COCO17_ANNO="/data2/lizhengxue/datasets/coco17/annotations"
DATASET_ROOT="/data2/lizhengxue/datasets"
FEATURE_ROOT="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data"
OUT_ROOT="/data2/lizhengxue/WorkSpace/onion_output/okvqa/knowledge4_x_dyfo4/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_knowledge4_x_dyfo4/${RUN_ID}"
WAVE_LAUNCH_DELAY="${WAVE_LAUNCH_DELAY:-90}"

mkdir -p "${OUT_ROOT}" "${LOGDIR}"
cd "${REPO_DIR}"

BASE_ARGS=(
  forward_code/onion.py
  --dataset_name okvqa
  --split_name val
  --engine "${ENGINE}"
  --raw_image_dir "${COCO14_ROOT}"
  --coco_path "${OKVQA_ROOT}"
  --coco_annotation_path "${COCO17_ANNO}"
  --caption_type vinvl_tag
  --valcaption_file "${FEATURE_ROOT}/input_text/vinvl_caption/VinVL_base_val2014.tsv"
  --tag_path "${FEATURE_ROOT}/input_text/coco_caption_pred_tags"
  --sg_path "${FEATURE_ROOT}/input_text/scene_graph_text"
  --concept_caption_path scene_graph_coco17_caption
  --similarity_path "${FEATURE_ROOT}/coco_clip_new"
  --similarity_metric imagequestion
  --train_sim_metric answer
  --train_sim_file "${FEATURE_ROOT}/input_text/scene_graph_text/train_object_select_okvqa.pk"
  --context_mode no_round_state
  --ensemble_strategy first
  --answer_postprocess safe_rules
  --direct_prompt_style default
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --num_shards 1
  --use_knowledge_enhance
  --knowledge_notes_mode raw_retrieved
  --knowledge_enhance_trigger always
  --knowledge_dataset_root "${DATASET_ROOT}"
  --knowledge_top_k 5
  --knowledge_per_source_top_k 5
  --knowledge_source_max_records 50000
  --knowledge_source_scan_limit 500000
)

extra_args_for_visual() {
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
      echo "unknown visual experiment: $1" >&2
      return 1
      ;;
  esac
}

run_exp() {
  local knowledge_source="$1"
  local visual_name="$2"
  local gpu_id="$3"
  local exp_name="${visual_name}__kb_${knowledge_source}"
  local out_dir="${OUT_ROOT}/${ENGINE}_${exp_name}"
  local cache_dir="${FEATURE_ROOT}/image_cache_onion/cache_${RUN_ID}_${exp_name}_gpu${gpu_id}"
  local log_file="${LOGDIR}/${exp_name}_gpu${gpu_id}.log"
  local merge_log="${LOGDIR}/${exp_name}_merge.log"
  local visual_extra
  visual_extra="$(extra_args_for_visual "${visual_name}")"

  mkdir -p "${out_dir}" "${cache_dir}"
  echo "[$(date '+%F %T')] start exp=${exp_name} shard=0/1 gpu=${gpu_id} log=${log_file}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --knowledge_sources "${knowledge_source}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --shard_id 0 \
    ${visual_extra} \
    > "${log_file}" 2>&1

  echo "[$(date '+%F %T')] merge exp=${exp_name} -> ${out_dir}/accuracy.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "${PYTHON}" "${BASE_ARGS[@]}" \
    --knowledge_sources "${knowledge_source}" \
    --cache_path "${cache_dir}" \
    --output_path "${out_dir}" \
    --merge_only \
    --summary_log "${out_dir}/accuracy.log" \
    ${visual_extra} \
    > "${merge_log}" 2>&1
}

run_wave() {
  local knowledge_source="$1"
  echo "[$(date '+%F %T')] ===== start wave knowledge_source=${knowledge_source} ====="
  run_exp "${knowledge_source}" direct_baseline 4 &
  sleep "${WAVE_LAUNCH_DELAY}"
  run_exp "${knowledge_source}" dyfo_evidence_inject 4 &
  run_exp "${knowledge_source}" dyfo_weighted_vote 5 &
  sleep "${WAVE_LAUNCH_DELAY}"
  run_exp "${knowledge_source}" dyfo_best_focus_answer 5 &
  wait
  echo "[$(date '+%F %T')] ===== finish wave knowledge_source=${knowledge_source} ====="
}

for knowledge_source in gs112k wikidata_kat wiki21m conceptnet; do
  run_wave "${knowledge_source}"
done

cat <<EOF
RUN_ID=${RUN_ID}
OUT_ROOT=${OUT_ROOT}
LOGDIR=${LOGDIR}
EOF
