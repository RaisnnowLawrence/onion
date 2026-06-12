#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_ROOT="${REPORT_DIR}/logs_cot_rescue_parallel"
MASTER_LOG="${REPORT_DIR}/cot_rescue_parallel_master.log"

SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
CACHE_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion
NUM_SHARDS=2
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
    local cot_style="$2"
    local rounds="$3"
    local shard="$4"
    local gpu="$5"
    local mcts_action="$6"
    local trigger_mode="$7"
    local n_sim="$8"

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
        --rounds "${rounds}"
        --iterative_strategy caption
        --engine "${ENGINE}"
        --sg_path "${SG_PATH}"
        --train_sim_metric answer
        --train_sim_file "${TRAIN_SIM_FILE}"
        --tag_path "${TAG_PATH}"
        --with_clip_verify
        --chain_of_thoughts
        --cot_style "${cot_style}"
        --shard_id "${shard}" --num_shards "${NUM_SHARDS}"
    )

    if [[ "${mcts_action}" != "none" ]]; then
        cmd+=(
            --use_image_enhance
            --mcts_n_simulations "${n_sim}"
            --mcts_trigger_mode "${trigger_mode}"
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
    local cot_style="$2"
    local mcts_action="$3"
    local trigger_mode="$4"
    local n_sim="$5"

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
        --cot_style "${cot_style}"
        --summary_log "${summary_log}"
    )

    if [[ "${mcts_action}" != "none" ]]; then
        cmd+=(
            --use_image_enhance
            --mcts_n_simulations "${n_sim}"
            --mcts_trigger_mode "${trigger_mode}"
            --mcts_action_mode "${mcts_action}"
            --mcts_filter_objects
        )
    fi

    echo "[$(timestamp)] merge ${exp_name}" | tee -a "${MASTER_LOG}"
    "${cmd[@]}" >> "${LOG_ROOT}/${exp_name}_merge.log" 2>&1
}

main() {
    echo "[$(timestamp)] CoT rescue ablations started" | tee -a "${MASTER_LOG}"

    launch_shard cot_rescue_compact_rounds1_2shards compact 1 0 0 none none 0
    launch_shard cot_rescue_compact_rounds1_2shards compact 1 1 0 none none 0

    launch_shard cot_rescue_answer_first_rounds1_2shards answer_first 1 0 1 none none 0
    launch_shard cot_rescue_answer_first_rounds1_2shards answer_first 1 1 1 none none 0

    launch_shard cot_rescue_compact_rounds3_2shards compact 3 0 2 none none 0
    launch_shard cot_rescue_compact_rounds3_2shards compact 3 1 2 none none 0

    launch_shard cot_rescue_compact_rounds5_2shards compact 5 0 6 none none 0
    launch_shard cot_rescue_compact_rounds5_2shards compact 5 1 6 none none 0

    launch_shard cot_rescue_compact_outline_mcts_rounds1_n10_2shards compact 1 0 7 outline_only count_color_object_only 10
    launch_shard cot_rescue_compact_outline_mcts_rounds1_n10_2shards compact 1 1 7 outline_only count_color_object_only 10

    launch_shard cot_rescue_compact_marker_mcts_rounds1_n10_2shards compact 1 0 0 marker_only count_color_object_only 10
    launch_shard cot_rescue_compact_marker_mcts_rounds1_n10_2shards compact 1 1 1 marker_only count_color_object_only 10

    echo "[$(timestamp)] all CoT rescue shards launched" | tee -a "${MASTER_LOG}"
    wait

    merge_exp cot_rescue_compact_rounds1_2shards compact none none 0 || true
    merge_exp cot_rescue_answer_first_rounds1_2shards answer_first none none 0 || true
    merge_exp cot_rescue_compact_rounds3_2shards compact none none 0 || true
    merge_exp cot_rescue_compact_rounds5_2shards compact none none 0 || true
    merge_exp cot_rescue_compact_outline_mcts_rounds1_n10_2shards compact outline_only count_color_object_only 10 || true
    merge_exp cot_rescue_compact_marker_mcts_rounds1_n10_2shards compact marker_only count_color_object_only 10 || true

    echo "[$(timestamp)] CoT rescue ablations finished" | tee -a "${MASTER_LOG}"
}

main "$@"
