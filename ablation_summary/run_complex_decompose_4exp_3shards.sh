#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/lizhengxue/WorkSpace/onion
PY=${PY:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=${ENGINE:-qwen3-VL-4B}
OUT_ROOT=${OUT_ROOT:-/data2/lizhengxue/WorkSpace/onion_output/aokvqa}
DATA_ROOT=${DATA_ROOT:-/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_complex_decompose_4exp_3shards}
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
  --use_all_regional_captions
  --max_regional_captions 10
  --chain_of_thoughts
  --cot_style complex_decompose
)

run_shard() {
  local name="$1"
  local gpu="$2"
  local shard="$3"
  shift 3

  local out="${OUT_ROOT}/${ENGINE}_forward2_${name}"
  local cache="${DATA_ROOT}/image_cache_onion/cache_${name}"
  mkdir -p "${out}" "${cache}"

  echo "[complex_decompose] start ${name} shard ${shard} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" "${REPO}/forward_code/onion.py" \
    --cache_path "${cache}" \
    --output_path "${out}" \
    "${common_args[@]}" \
    "$@" \
    --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
    > "${LOG_DIR}/${name}_gpu${gpu}_shard${shard}.log" 2>&1
  echo "[complex_decompose] finish ${name} shard ${shard} status=0"
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

  echo "[complex_decompose] launch ${name} on GPU ${gpu}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_shard "${name}" "${gpu}" "${shard}" "$@" &
  done
  wait
  echo "[complex_decompose] merge ${name}"
  merge_exp "${name}" "$@"
}

run_named_exp() {
  local exp="$1"
  local gpu="$2"

  case "${exp}" in
    always)
      run_exp_3shards complex_decompose_always_3shards_gpu${gpu} "${gpu}" \
        --decompose_complexity_mode always
      ;;
    adaptive)
      run_exp_3shards complex_decompose_adaptive_3shards_gpu${gpu} "${gpu}" \
        --decompose_complexity_mode adaptive
      ;;
    adaptive_verify)
      run_exp_3shards complex_decompose_adaptive_verify_3shards_gpu${gpu} "${gpu}" \
        --decompose_complexity_mode adaptive \
        --decompose_verify
      ;;
    conservative_verify)
      run_exp_3shards complex_decompose_conservative_verify_3shards_gpu${gpu} "${gpu}" \
        --decompose_complexity_mode conservative \
        --decompose_verify
      ;;
    *)
      echo "unknown experiment: ${exp}" >&2
      exit 2
      ;;
  esac
}

case "${1:-}" in
  always|adaptive|adaptive_verify|conservative_verify)
    run_named_exp "$1" "${2:-0}"
    ;;
  all)
    run_named_exp always 0 &
    run_named_exp adaptive 1 &
    run_named_exp adaptive_verify 2 &
    run_named_exp conservative_verify 3 &
    wait
    ;;
  *)
    cat >&2 <<'EOF'
usage:
  run_complex_decompose_4exp_3shards.sh all
  run_complex_decompose_4exp_3shards.sh EXP GPU

EXP:
  always
  adaptive
  adaptive_verify
  conservative_verify
EOF
    exit 2
    ;;
esac
