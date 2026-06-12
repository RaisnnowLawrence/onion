#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_ROOT="${REPORT_DIR}/logs_direct_verify_parallel"
MASTER_LOG="${REPORT_DIR}/direct_verify_parallel_master.log"

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
    local shard="$2"
    local gpu="$3"
    local mcts_action="$4"

    local output_path="${OUT_ROOT}/${ENGINE}_forward2_${exp_name}"
    local cache_path="${CACHE_ROOT}/cache_forward4b_${exp_name}"
    local log_dir="${LOG_ROOT}/${exp_name}"
    local log_path="${log_dir}/${exp_name}_gpu${gpu}_shard${shard}.log"
    mkdir -p "${output_path}" "${cache_path}" "${log_dir}"

    local cmd=(
        env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
        "${PYTHON_BIN}" "${BASE}/forward_code/onion.py"
        --cache_path "${cache_path}"
        --output_path "${output_path}"
        --caption_type vinvl
        --n_shot 1
        --n_ensemble 5
        --rounds 1
        --iterative_strategy caption
        --engine "${ENGINE}"
        --sg_path "${SG_PATH}"
        --train_sim_metric answer
        --train_sim_file "${TRAIN_SIM_FILE}"
        --tag_path "${TAG_PATH}"
        --with_clip_verify
        --chain_of_thoughts
        --cot_style direct_verify
        --shard_id "${shard}" --num_shards "${NUM_SHARDS}"
    )

    if [[ "${mcts_action}" != "none" ]]; then
        cmd+=(
            --use_image_enhance
            --mcts_n_simulations 10
            --mcts_trigger_mode count_color_object_only
            --mcts_action_mode "${mcts_action}"
            --mcts_filter_objects
        )
    fi

    echo "[$(timestamp)] launch ${exp_name} shard ${shard}/${NUM_SHARDS} on GPU ${gpu}" | tee -a "${MASTER_LOG}"
    setsid "${cmd[@]}" > "${log_path}" 2>&1 < /dev/null &
    echo "$!" >> "${LOG_ROOT}/${exp_name}.pids"
}

merge_exp() {
    local exp_name="$1"
    local mcts_action="$2"
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

    local cmd=(
        env CUDA_VISIBLE_DEVICES=""
        "${PYTHON_BIN}" "${BASE}/forward_code/onion.py"
        --engine "${ENGINE}"
        --output_path "${output_path}"
        --caption_type vinvl
        --sg_path "${SG_PATH}"
        --similarity_metric imagequestion
        --merge_only
        --chain_of_thoughts
        --cot_style direct_verify
        --summary_log "${summary_log}"
    )

    if [[ "${mcts_action}" != "none" ]]; then
        cmd+=(
            --use_image_enhance
            --mcts_n_simulations 10
            --mcts_trigger_mode count_color_object_only
            --mcts_action_mode "${mcts_action}"
            --mcts_filter_objects
        )
    fi

    echo "[$(timestamp)] merge ${exp_name}" | tee -a "${MASTER_LOG}"
    "${cmd[@]}" >> "${LOG_ROOT}/${exp_name}_merge.log" 2>&1
}

main() {
    echo "[$(timestamp)] direct_verify ablations started" | tee -a "${MASTER_LOG}"

    launch_shard cot_rescue_direct_verify_rounds1_4shards 0 0 none
    launch_shard cot_rescue_direct_verify_rounds1_4shards 1 0 none
    launch_shard cot_rescue_direct_verify_rounds1_4shards 2 0 none
    launch_shard cot_rescue_direct_verify_rounds1_4shards 3 0 none

    launch_shard cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards 0 1 marker_only
    launch_shard cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards 1 1 marker_only
    launch_shard cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards 2 1 marker_only
    launch_shard cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards 3 1 marker_only

    launch_shard cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards 0 2 outline_only
    launch_shard cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards 1 2 outline_only
    launch_shard cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards 2 2 outline_only
    launch_shard cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards 3 2 outline_only

    echo "[$(timestamp)] all direct_verify shards launched" | tee -a "${MASTER_LOG}"
    wait

    merge_exp cot_rescue_direct_verify_rounds1_4shards none || true
    merge_exp cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards marker_only || true
    merge_exp cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards outline_only || true
    echo "[$(timestamp)] direct_verify ablations finished" | tee -a "${MASTER_LOG}"
}

main "$@"
