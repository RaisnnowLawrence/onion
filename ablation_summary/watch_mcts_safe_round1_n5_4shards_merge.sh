#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
EXP_NAME=mcts_safe_no_cot_rounds1_n5_4shards
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
OUTPUT_PATH="${OUT_ROOT}/${ENGINE}_forward2_${EXP_NAME}"
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs_mcts_safe_round1_n5_4shards"
WATCH_LOG="${REPORT_DIR}/mcts_safe_round1_n5_4shards_merge_watch.log"
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

sample_count() {
    find "${OUTPUT_PATH}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

running_pids() {
    pgrep -f "forward_code/onion.py.*${OUTPUT_PATH}" || true
}

merge_if_complete() {
    local count
    count="$(sample_count)"
    if [[ -f "${OUTPUT_PATH}/accuracy.log" ]]; then
        echo "[$(timestamp)] already merged: ${OUTPUT_PATH}/accuracy.log" | tee -a "${WATCH_LOG}"
        return 0
    fi
    if [[ "${count}" -lt 1145 ]]; then
        echo "[$(timestamp)] waiting: ${count}/1145 samples" | tee -a "${WATCH_LOG}"
        return 1
    fi

    echo "[$(timestamp)] merge ${EXP_NAME}" | tee -a "${WATCH_LOG}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${OUTPUT_PATH}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        --use_image_enhance \
        --mcts_n_simulations 5 \
        --mcts_trigger_mode visual_detail_only \
        --mcts_action_mode outline_only \
        --mcts_filter_objects \
        --summary_log "${OUTPUT_PATH}/accuracy.log" \
        >> "${LOG_DIR}/${EXP_NAME}_merge.log" 2>&1
}

echo "[$(timestamp)] MCTS 4-shard merge watcher started" | tee -a "${WATCH_LOG}"
while true; do
    if merge_if_complete; then
        break
    fi
    if [[ -z "$(running_pids)" ]]; then
        echo "[$(timestamp)] no running MCTS 4-shard processes; current samples=$(sample_count)" | tee -a "${WATCH_LOG}"
    fi
    sleep 600
done
echo "[$(timestamp)] MCTS 4-shard merge watcher finished" | tee -a "${WATCH_LOG}"
