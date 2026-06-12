#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_ROOT="${REPORT_DIR}/logs_cot_rescue_rounds35_4shards"
MASTER_LOG="${REPORT_DIR}/cot_rescue_rounds35_4shards_master.log"

SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
CACHE_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion
NUM_SHARDS=4
TOTAL_SAMPLES=1145

mkdir -p "${LOG_ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

sample_count() {
    local output_path="$1"
    find "${output_path}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

launch_shard() {
    local exp_name="$1"
    local rounds="$2"
    local shard="$3"
    local gpu="$4"

    local output_path="${OUT_ROOT}/${ENGINE}_forward2_${exp_name}"
    local cache_path="${CACHE_ROOT}/cache_forward4b_${exp_name}"
    local log_dir="${LOG_ROOT}/${exp_name}"
    local log_path="${log_dir}/${exp_name}_gpu${gpu}_shard${shard}.log"
    mkdir -p "${output_path}" "${cache_path}" "${log_dir}"

    echo "[$(timestamp)] launch ${exp_name} shard ${shard}/${NUM_SHARDS} on GPU ${gpu}" | tee -a "${MASTER_LOG}"
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --cache_path "${cache_path}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --n_shot 1 \
        --n_ensemble 5 \
        --rounds "${rounds}" \
        --iterative_strategy caption \
        --engine "${ENGINE}" \
        --sg_path "${SG_PATH}" \
        --train_sim_metric answer \
        --train_sim_file "${TRAIN_SIM_FILE}" \
        --tag_path "${TAG_PATH}" \
        --with_clip_verify \
        --chain_of_thoughts \
        --cot_style compact \
        --shard_id "${shard}" --num_shards "${NUM_SHARDS}" \
        > "${log_path}" 2>&1 < /dev/null &
    echo "$!" >> "${LOG_ROOT}/${exp_name}.pids"
}

merge_exp() {
    local exp_name="$1"
    local output_path="${OUT_ROOT}/${ENGINE}_forward2_${exp_name}"
    local summary_log="${output_path}/accuracy.log"
    local count
    count="$(sample_count "${output_path}")"
    if [[ "${count}" -lt "${TOTAL_SAMPLES}" ]]; then
        echo "[$(timestamp)] skip merge ${exp_name}: ${count}/${TOTAL_SAMPLES} samples" | tee -a "${MASTER_LOG}"
        return 1
    fi
    if [[ -f "${summary_log}" ]]; then
        echo "[$(timestamp)] already merged ${exp_name}: ${summary_log}" | tee -a "${MASTER_LOG}"
        return 0
    fi

    echo "[$(timestamp)] merge ${exp_name}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        --chain_of_thoughts \
        --cot_style compact \
        --summary_log "${summary_log}" \
        >> "${LOG_ROOT}/${exp_name}_merge.log" 2>&1
}

main() {
    echo "[$(timestamp)] CoT rescue rounds3/5 4-shard relaunch started" | tee -a "${MASTER_LOG}"

    launch_shard cot_rescue_compact_rounds3_4shards 3 0 2
    launch_shard cot_rescue_compact_rounds3_4shards 3 1 3
    launch_shard cot_rescue_compact_rounds3_4shards 3 2 4
    launch_shard cot_rescue_compact_rounds3_4shards 3 3 5

    launch_shard cot_rescue_compact_rounds5_4shards 5 0 6
    launch_shard cot_rescue_compact_rounds5_4shards 5 1 6
    launch_shard cot_rescue_compact_rounds5_4shards 5 2 7
    launch_shard cot_rescue_compact_rounds5_4shards 5 3 2

    echo "[$(timestamp)] all rounds3/5 4-shard jobs launched" | tee -a "${MASTER_LOG}"
    wait

    merge_exp cot_rescue_compact_rounds3_4shards || true
    merge_exp cot_rescue_compact_rounds5_4shards || true
    echo "[$(timestamp)] CoT rescue rounds3/5 4-shard relaunch finished" | tee -a "${MASTER_LOG}"
}

main "$@"
