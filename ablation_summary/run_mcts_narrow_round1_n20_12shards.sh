#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
EXP_NAME=mcts_narrow_no_cot_rounds1_n20_12shards
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
OUTPUT_PATH="${OUT_ROOT}/${ENGINE}_forward2_${EXP_NAME}"
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs_mcts_narrow_round1_n20_12shards"
MASTER_LOG="${REPORT_DIR}/mcts_narrow_round1_n20_12shards_master.log"

CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b_mcts_narrow_round1_n20_12shards
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags

mkdir -p "${LOG_DIR}" "${CACHE_PATH}" "${OUTPUT_PATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

sample_count() {
    find "${OUTPUT_PATH}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

launch_shard() {
    local shard="$1"
    local gpu="$2"
    local log_path="${LOG_DIR}/${EXP_NAME}_gpu${gpu}_shard${shard}.log"

    echo "[$(timestamp)] launch ${EXP_NAME} shard ${shard}/12 on GPU ${gpu}" | tee -a "${MASTER_LOG}"
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --cache_path "${CACHE_PATH}" \
        --output_path "${OUTPUT_PATH}" \
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
        --mcts_n_simulations 20 \
        --mcts_trigger_mode count_color_object_only \
        --mcts_action_mode outline_only \
        --mcts_filter_objects \
        --shard_id "${shard}" --num_shards 12 \
        > "${log_path}" 2>&1 < /dev/null &
    RUN_PID="$!"
}

merge_if_complete() {
    local summary_log="${OUTPUT_PATH}/accuracy.log"
    local count
    count="$(sample_count)"
    if [[ "${count}" -lt 1145 ]]; then
        echo "[$(timestamp)] skip merge: ${count}/1145 samples" | tee -a "${MASTER_LOG}"
        return 1
    fi
    if [[ -f "${summary_log}" ]]; then
        echo "[$(timestamp)] already merged: ${summary_log}" | tee -a "${MASTER_LOG}"
        return 0
    fi

    echo "[$(timestamp)] merge ${EXP_NAME}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${OUTPUT_PATH}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        --use_image_enhance \
        --mcts_n_simulations 20 \
        --mcts_trigger_mode count_color_object_only \
        --mcts_action_mode outline_only \
        --mcts_filter_objects \
        --summary_log "${summary_log}" \
        >> "${LOG_DIR}/${EXP_NAME}_merge.log" 2>&1
}

main() {
    echo "[$(timestamp)] MCTS narrow round1 n20 12-shard experiment started" | tee -a "${MASTER_LOG}"
    echo "[$(timestamp)] output: ${OUTPUT_PATH}" | tee -a "${MASTER_LOG}"

    local pid0 pid1 pid2 pid3 pid4 pid5 pid6 pid7 pid8 pid9 pid10 pid11
    launch_shard 0 0; pid0="${RUN_PID}"
    launch_shard 1 0; pid1="${RUN_PID}"
    launch_shard 2 0; pid2="${RUN_PID}"
    launch_shard 3 1; pid3="${RUN_PID}"
    launch_shard 4 1; pid4="${RUN_PID}"
    launch_shard 5 1; pid5="${RUN_PID}"
    launch_shard 6 2; pid6="${RUN_PID}"
    launch_shard 7 2; pid7="${RUN_PID}"
    launch_shard 8 2; pid8="${RUN_PID}"
    launch_shard 9 6; pid9="${RUN_PID}"
    launch_shard 10 6; pid10="${RUN_PID}"
    launch_shard 11 6; pid11="${RUN_PID}"
    echo "[$(timestamp)] pids: shard0=${pid0}, shard1=${pid1}, shard2=${pid2}, shard3=${pid3}, shard4=${pid4}, shard5=${pid5}, shard6=${pid6}, shard7=${pid7}, shard8=${pid8}, shard9=${pid9}, shard10=${pid10}, shard11=${pid11}" | tee -a "${MASTER_LOG}"

    wait "${pid0}" || true
    wait "${pid1}" || true
    wait "${pid2}" || true
    wait "${pid3}" || true
    wait "${pid4}" || true
    wait "${pid5}" || true
    wait "${pid6}" || true
    wait "${pid7}" || true
    wait "${pid8}" || true
    wait "${pid9}" || true
    wait "${pid10}" || true
    wait "${pid11}" || true

    merge_if_complete || true
    echo "[$(timestamp)] MCTS narrow round1 n20 12-shard experiment finished with $(sample_count)/1145 samples" | tee -a "${MASTER_LOG}"
}

main "$@"
