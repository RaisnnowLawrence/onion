#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_ROOT="${REPORT_DIR}/logs_marker_mcts_parallel"
MASTER_LOG="${REPORT_DIR}/marker_mcts_parallel_master.log"

SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
CACHE_ROOT=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion

mkdir -p "${LOG_ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

launch_shard() {
    local exp_name="$1"
    local n_sim="$2"
    local trigger_mode="$3"
    local action_mode="$4"
    local shard="$5"
    local num_shards="$6"
    local gpu="$7"

    local output_path="${OUT_ROOT}/${ENGINE}_forward2_${exp_name}"
    local cache_path="${CACHE_ROOT}/cache_forward4b_${exp_name}"
    local log_dir="${LOG_ROOT}/${exp_name}"
    local log_path="${log_dir}/${exp_name}_gpu${gpu}_shard${shard}.log"
    mkdir -p "${output_path}" "${cache_path}" "${log_dir}"

    echo "[$(timestamp)] launch ${exp_name} shard ${shard}/${num_shards} on GPU ${gpu}" | tee -a "${MASTER_LOG}"
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --cache_path "${cache_path}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --n_shot 1 \
        --n_ensemble 5 \
        --rounds 1 \
        --iterative_strategy caption \
        --engine "${ENGINE}" \
        --sg_path "${SG_PATH}" \
        --train_sim_metric answer \
        --train_sim_file "${TRAIN_SIM_FILE}" \
        --tag_path "${TAG_PATH}" \
        --with_clip_verify \
        --use_image_enhance \
        --mcts_n_simulations "${n_sim}" \
        --mcts_trigger_mode "${trigger_mode}" \
        --mcts_action_mode "${action_mode}" \
        --mcts_filter_objects \
        --shard_id "${shard}" --num_shards "${num_shards}" \
        > "${log_path}" 2>&1 < /dev/null &
    echo "$!" >> "${LOG_ROOT}/${exp_name}.pids"
}

main() {
    echo "[$(timestamp)] marker MCTS parallel ablations started" | tee -a "${MASTER_LOG}"

    # 1. Key comparison: marker-only narrow trigger, n=10, 4 shards.
    launch_shard mcts_marker_narrow_no_cot_rounds1_n10_4shards 10 count_color_object_only marker_only 0 4 0
    launch_shard mcts_marker_narrow_no_cot_rounds1_n10_4shards 10 count_color_object_only marker_only 1 4 0
    launch_shard mcts_marker_narrow_no_cot_rounds1_n10_4shards 10 count_color_object_only marker_only 2 4 1
    launch_shard mcts_marker_narrow_no_cot_rounds1_n10_4shards 10 count_color_object_only marker_only 3 4 2

    # 2. Low-search comparison.
    launch_shard mcts_marker_narrow_no_cot_rounds1_n5_2shards 5 count_color_object_only marker_only 0 2 0
    launch_shard mcts_marker_narrow_no_cot_rounds1_n5_2shards 5 count_color_object_only marker_only 1 2 1

    # 3. High-search comparison.
    launch_shard mcts_marker_narrow_no_cot_rounds1_n20_2shards 20 count_color_object_only marker_only 0 2 1
    launch_shard mcts_marker_narrow_no_cot_rounds1_n20_2shards 20 count_color_object_only marker_only 1 2 2

    # 5. Wider trigger with marker-only.
    launch_shard mcts_marker_visual_no_cot_rounds1_n10_2shards 10 visual_detail_only marker_only 0 2 2
    launch_shard mcts_marker_visual_no_cot_rounds1_n10_2shards 10 visual_detail_only marker_only 1 2 6

    # 6. Minimal-search marker: tests whether marker itself is enough.
    launch_shard mcts_marker_narrow_no_cot_rounds1_n1_2shards 1 count_color_object_only marker_only 0 2 6
    launch_shard mcts_marker_narrow_no_cot_rounds1_n1_2shards 1 count_color_object_only marker_only 1 2 6

    echo "[$(timestamp)] all marker MCTS shards launched" | tee -a "${MASTER_LOG}"
}

main "$@"
