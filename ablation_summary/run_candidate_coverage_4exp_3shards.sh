#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_candidate_coverage_4exp_3shards}
NUM_SHARDS=${NUM_SHARDS:-3}
SHARD_PARALLEL=${SHARD_PARALLEL:-1}

mkdir -p "${LOG_DIR}"

common_args=(
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
  --cot_style candidate_judge
)

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  shift 3

  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  mkdir -p "${out}" "${cache}"

  echo "[candidate_coverage] start ${name} shard ${shard} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    "$@" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    > "${LOG_DIR}/${name}_gpu${gpu}_shard${shard}.log" 2>&1
  echo "[candidate_coverage] finish ${name} shard ${shard} status=0"
}

merge_exp() {
  local name="$1"
  shift

  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  mkdir -p "${out}" "${cache}"

  CUDA_VISIBLE_DEVICES="" "${PY}" "${REPO}/forward_code/onion.py" \
    --merge_only \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    "$@" \
    --summary_log "${out}/accuracy.log" \
    > "${LOG_DIR}/${name}_merge.log" 2>&1
}

run_exp_3shards() {
  local name="$1"
  local gpu="$2"
  shift 2

  echo "[candidate_coverage] launch ${name} on GPU ${gpu}"
  if [[ "${SHARD_PARALLEL}" == "1" ]]; then
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      run_shard "${name}" "${gpu}" "${shard}" "$@" &
    done
    wait
  else
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      run_shard "${name}" "${gpu}" "${shard}" "$@"
    done
  fi
  echo "[candidate_coverage] merge ${name}"
  merge_exp "${name}" "$@"
}

run_named_exp() {
  local exp="$1"
  local gpu="$2"

  case "${exp}" in
    coverage_scan)
      run_exp_3shards candidate_coverage_scan_rerun3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_include_coverage_candidate \
        --candidate_judge_include_caption_candidate \
        --candidate_judge_route_evidence \
        --use_all_regional_captions \
        --max_regional_captions 12 \
        --use_caption_enhance \
        --use_knowledge_enhance \
        --use_ocr_context
      ;;
    count_specialist)
      run_exp_3shards candidate_count_specialist_rerun3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_include_count_candidate \
        --candidate_judge_include_coverage_candidate \
        --candidate_judge_always_judge \
        --candidate_judge_route_evidence \
        --use_all_regional_captions \
        --max_regional_captions 12 \
        --use_image_enhance \
        --candidate_judge_use_enhanced_image \
        --mcts_n_simulations 5 \
        --mcts_trigger_mode count_color_object_only \
        --mcts_action_mode marker_only \
        --mcts_filter_objects
      ;;
    ocr_specialist)
      run_exp_3shards candidate_ocr_specialist_rerun3shards_gpu${gpu} "${gpu}" \
        --caption_type vinvl_ocr \
        --candidate_judge_include_ocr_candidate \
        --candidate_judge_include_coverage_candidate \
        --candidate_judge_always_judge \
        --candidate_judge_route_evidence \
        --use_ocr_context \
        --use_all_regional_captions \
        --max_regional_captions 8
      ;;
    diverse_pool)
      run_exp_3shards candidate_diverse_pool_rerun3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_include_caption_candidate \
        --candidate_judge_include_count_candidate \
        --candidate_judge_include_ocr_candidate \
        --candidate_judge_include_coverage_candidate \
        --candidate_judge_include_contrast_candidate \
        --candidate_judge_always_judge \
        --candidate_judge_allow_new_answer \
        --candidate_judge_route_evidence \
        --use_all_regional_captions \
        --max_regional_captions 12 \
        --use_caption_enhance \
        --use_knowledge_enhance \
        --use_ocr_context
      ;;
    *)
      echo "unknown experiment: ${exp}" >&2
      exit 2
      ;;
  esac
}

case "${1:-}" in
  coverage_scan|count_specialist|ocr_specialist|diverse_pool)
    run_named_exp "$1" "${2:-0}"
    ;;
  all)
    run_named_exp coverage_scan 0 &
    run_named_exp count_specialist 1 &
    run_named_exp ocr_specialist 2 &
    run_named_exp diverse_pool 3 &
    wait
    ;;
  *)
    cat >&2 <<'EOF'
usage:
  run_candidate_coverage_4exp_3shards.sh all
  run_candidate_coverage_4exp_3shards.sh EXP GPU

EXP:
  coverage_scan
  count_specialist
  ocr_specialist
  diverse_pool
EOF
    exit 2
    ;;
esac
