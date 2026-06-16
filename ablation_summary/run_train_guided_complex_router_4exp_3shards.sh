#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data2/lizhengxue/WorkSpace/onion}
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_ROOT=${LOG_ROOT:-${REPORT_DIR}/logs_train_guided_complex_router_4exp_3shards}
PROFILE_ROOT=${PROFILE_ROOT:-${REPORT_DIR}/train_guided_complex_profiles}
NUM_SHARDS=${NUM_SHARDS:-3}
TRAIN_PROFILE_TOTAL=${TRAIN_PROFILE_TOTAL:-10000}
TRAIN_PROFILE_MAX_PER_SHARD=${TRAIN_PROFILE_MAX_PER_SHARD:-$(( (TRAIN_PROFILE_TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))}
PROFILE_RUN_ID=${PROFILE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}

mkdir -p "${LOG_ROOT}" "${PROFILE_ROOT}"

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
  --use_all_regional_captions
  --max_regional_captions 10
)

router_args_for_exp() {
  local exp="$1"
  case "${exp}" in
    direct_failure)
      echo "--strategy_router_mode direct_failure --strategy_topk 20 --strategy_min_direct_hard_rate 0.55"
      ;;
    direct_vs_complex)
      echo "--strategy_router_mode direct_vs_complex --strategy_topk 20 --strategy_min_complex_win_rate 0.20 --strategy_margin 0.06"
      ;;
    qtype_conditional)
      echo "--strategy_router_mode qtype_conditional --strategy_topk 12 --strategy_min_neighbors 4 --strategy_min_complex_win_rate 0.18 --strategy_margin 0.04"
      ;;
    conservative_risk)
      echo "--strategy_router_mode conservative_risk --strategy_topk 20 --strategy_margin 0.02 --strategy_min_net_gain 0.06 --strategy_max_damage_rate 0.18"
      ;;
    *)
      echo "unknown experiment: ${exp}" >&2
      exit 2
      ;;
  esac
}

needs_complex_profile() {
  local exp="$1"
  case "${exp}" in
    direct_failure) return 1 ;;
    direct_vs_complex|qtype_conditional|conservative_risk) return 0 ;;
    *) echo "unknown experiment: ${exp}" >&2; exit 2 ;;
  esac
}

run_profile_shard() {
  local exp="$1"
  local strategy="$2"
  local gpu="$3"
  local shard="$4"
  shift 4

  local exp_profile_dir="${PROFILE_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local exp_log_dir="${LOG_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local name="train_guided_${exp}_${strategy}_train10k_shard${shard}_gpu${gpu}_${PROFILE_RUN_ID}"
  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  local profile="${exp_profile_dir}/${strategy}_train_shard${shard}.jsonl"
  mkdir -p "${out}" "${cache}" "${exp_profile_dir}" "${exp_log_dir}"
  rm -f "${profile}"

  echo "[profile:${exp}] start ${strategy} shard ${shard} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --split_name train \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    "$@" \
    --strategy_name "${strategy}" \
    --strategy_profile_output "${profile}" \
    --max_samples_per_shard "${TRAIN_PROFILE_MAX_PER_SHARD}" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    > "${exp_log_dir}/${name}.log" 2>&1
  echo "[profile:${exp}] finish ${strategy} shard ${shard}"
}

build_exp_profile() {
  local exp="$1"
  local gpu="$2"

  local exp_profile_dir="${PROFILE_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local exp_log_dir="${LOG_ROOT}/${exp}_${PROFILE_RUN_ID}"
  mkdir -p "${exp_profile_dir}" "${exp_log_dir}"
  rm -f "${exp_profile_dir}"/*.jsonl

  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_profile_shard "${exp}" direct "${gpu}" "${shard}" &
  done
  wait
  cat "${exp_profile_dir}"/direct_train_shard*.jsonl > "${exp_profile_dir}/direct_train_profile.jsonl"

  if needs_complex_profile "${exp}"; then
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      run_profile_shard "${exp}" complex_decompose "${gpu}" "${shard}" \
        --chain_of_thoughts \
        --cot_style complex_decompose \
        --decompose_complexity_mode adaptive \
        --decompose_verify &
    done
    wait
    cat "${exp_profile_dir}"/complex_decompose_train_shard*.jsonl > "${exp_profile_dir}/complex_decompose_train_profile.jsonl"
    "${PY}" "${REPO}/ablation_summary/build_strategy_rag_profile.py" \
      --profile "${exp_profile_dir}/direct_train_profile.jsonl" \
      --profile "${exp_profile_dir}/complex_decompose_train_profile.jsonl" \
      --output "${exp_profile_dir}/combined_profile.jsonl" \
      > "${exp_log_dir}/build_profile.log" 2>&1
  else
    "${PY}" "${REPO}/ablation_summary/build_strategy_rag_profile.py" \
      --profile "${exp_profile_dir}/direct_train_profile.jsonl" \
      --output "${exp_profile_dir}/combined_profile.jsonl" \
      > "${exp_log_dir}/build_profile.log" 2>&1
  fi
}

run_router_shard() {
  local exp="$1"
  local gpu="$2"
  local shard="$3"
  shift 3

  local exp_profile_dir="${PROFILE_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local exp_log_dir="${LOG_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local name="train_guided_${exp}_train10k_router_val_3shards_gpu${gpu}"
  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  mkdir -p "${out}" "${cache}" "${exp_log_dir}"

  echo "[router:${exp}] start shard ${shard} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --split_name val \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    --chain_of_thoughts \
    --cot_style rag_strategy_router \
    --strategy_profile_path "${exp_profile_dir}/combined_profile.jsonl" \
    --strategy_direct_name direct \
    --strategy_cot_name complex_decompose \
    --strategy_cot_runtime complex_decompose \
    --decompose_complexity_mode adaptive \
    --decompose_verify \
    "$@" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    > "${exp_log_dir}/${name}_shard${shard}.log" 2>&1
  echo "[router:${exp}] finish shard ${shard}"
}

merge_router_exp() {
  local exp="$1"
  local gpu="$2"
  shift 2

  local exp_profile_dir="${PROFILE_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local exp_log_dir="${LOG_ROOT}/${exp}_${PROFILE_RUN_ID}"
  local name="train_guided_${exp}_train10k_router_val_3shards_gpu${gpu}"
  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  mkdir -p "${out}" "${cache}" "${exp_log_dir}"

  CUDA_VISIBLE_DEVICES="" "${PY}" "${REPO}/forward_code/onion.py" \
    --merge_only \
    --split_name val \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    --chain_of_thoughts \
    --cot_style rag_strategy_router \
    --strategy_profile_path "${exp_profile_dir}/combined_profile.jsonl" \
    --strategy_direct_name direct \
    --strategy_cot_name complex_decompose \
    --strategy_cot_runtime complex_decompose \
    --decompose_complexity_mode adaptive \
    --decompose_verify \
    "$@" \
    --summary_log "${out}/accuracy.log" \
    > "${exp_log_dir}/${name}_merge.log" 2>&1
}

run_exp() {
  local exp="$1"
  local gpu="$2"
  shift 2

  local router_args
  router_args=$(router_args_for_exp "${exp}")

  echo "[exp:${exp}] build train RAG profile on GPU ${gpu}, run_id=${PROFILE_RUN_ID}"
  build_exp_profile "${exp}" "${gpu}"

  echo "[exp:${exp}] run val router on GPU ${gpu}"
  # shellcheck disable=SC2086
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_router_shard "${exp}" "${gpu}" "${shard}" ${router_args} "$@" &
  done
  wait
  # shellcheck disable=SC2086
  merge_router_exp "${exp}" "${gpu}" ${router_args} "$@"
}

run_all() {
  run_exp direct_failure 0 &
  run_exp direct_vs_complex 1 &
  run_exp qtype_conditional 2 &
  run_exp conservative_risk 3 &
  wait
}

case "${1:-}" in
  all)
    run_all
    ;;
  direct_failure|direct_vs_complex|qtype_conditional|conservative_risk)
    run_exp "$1" "${2:-0}"
    ;;
  *)
    cat >&2 <<'EOF'
usage:
  run_train_guided_complex_router_4exp_3shards.sh all
  run_train_guided_complex_router_4exp_3shards.sh EXP GPU

EXP:
  direct_failure
  direct_vs_complex
  qtype_conditional
  conservative_risk

ENV:
  TRAIN_PROFILE_TOTAL=10000
  NUM_SHARDS=3
  PROFILE_RUN_ID=<optional stable id>
EOF
    exit 2
    ;;
esac
