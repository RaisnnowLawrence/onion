#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_candidate_judge_8ablations_3shards}
NUM_SHARDS=${NUM_SHARDS:-3}

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

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    "$@" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    > "${LOG_DIR}/${name}_gpu${gpu}_shard${shard}.log" 2>&1
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

  echo "[candidate_judge] launch ${name} on GPU ${gpu}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_shard "${name}" "${gpu}" "${shard}" "$@" &
  done
  wait
  echo "[candidate_judge] merge ${name}"
  merge_exp "${name}" "$@"
}

run_named_exp() {
  local exp="$1"
  local gpu="$2"

  case "${exp}" in
    core)
      run_exp_3shards candidate_judge_core_3shards_gpu${gpu} "${gpu}"
      ;;
    always_judge)
      run_exp_3shards candidate_judge_always_judge_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_always_judge
      ;;
    caption_candidate)
      run_exp_3shards candidate_judge_caption_candidate_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_include_caption_candidate
      ;;
    strict_consensus3)
      run_exp_3shards candidate_judge_strict_consensus3_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_consensus_votes 3
      ;;
    routed_caption_knowledge)
      run_exp_3shards candidate_judge_routed_caption_knowledge_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_route_evidence \
        --use_caption_enhance \
        --use_knowledge_enhance
      ;;
    regions_ocr)
      run_exp_3shards candidate_judge_regions_ocr_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_route_evidence \
        --use_all_regional_captions \
        --max_regional_captions 8 \
        --use_ocr_context
      ;;
    marker_mcts)
      run_exp_3shards candidate_judge_marker_mcts_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_route_evidence \
        --candidate_judge_use_enhanced_image \
        --use_image_enhance \
        --mcts_n_simulations 5 \
        --mcts_trigger_mode count_color_object_only \
        --mcts_action_mode marker_only \
        --mcts_filter_objects
      ;;
    allow_new_answer)
      run_exp_3shards candidate_judge_allow_new_answer_3shards_gpu${gpu} "${gpu}" \
        --candidate_judge_include_caption_candidate \
        --candidate_judge_route_evidence \
        --candidate_judge_allow_new_answer \
        --use_caption_enhance \
        --use_knowledge_enhance
      ;;
    *)
      echo "unknown experiment: ${exp}" >&2
      exit 2
      ;;
  esac
}

case "${1:-}" in
  core|always_judge|caption_candidate|strict_consensus3|routed_caption_knowledge|regions_ocr|marker_mcts|allow_new_answer)
    run_named_exp "$1" "${2:-0}"
    ;;
  all)
    run_named_exp core 0 &
    run_named_exp always_judge 1 &
    run_named_exp caption_candidate 2 &
    run_named_exp strict_consensus3 3 &
    run_named_exp routed_caption_knowledge 4 &
    run_named_exp regions_ocr 5 &
    run_named_exp marker_mcts 6 &
    run_named_exp allow_new_answer 7 &
    wait
    ;;
  *)
    cat >&2 <<'EOF'
usage:
  run_candidate_judge_8ablations_3shards.sh all
  run_candidate_judge_8ablations_3shards.sh EXP GPU

EXP:
  core
  always_judge
  caption_candidate
  strict_consensus3
  routed_caption_knowledge
  regions_ocr
  marker_mcts
  allow_new_answer
EOF
    exit 2
    ;;
esac
