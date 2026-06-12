#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_ROOT="${REPORT_DIR}/logs_marker_mcts_parallel"
WATCH_LOG="${REPORT_DIR}/marker_mcts_parallel_merge_watch.log"
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text

EXPERIMENTS=(
  "mcts_marker_narrow_no_cot_rounds1_n10_4shards:10:count_color_object_only:marker_only"
  "mcts_marker_narrow_no_cot_rounds1_n5_2shards:5:count_color_object_only:marker_only"
  "mcts_marker_narrow_no_cot_rounds1_n20_2shards:20:count_color_object_only:marker_only"
  "mcts_marker_visual_no_cot_rounds1_n10_2shards:10:visual_detail_only:marker_only"
  "mcts_marker_narrow_no_cot_rounds1_n1_2shards:1:count_color_object_only:marker_only"
)

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

output_path() {
    echo "${OUT_ROOT}/${ENGINE}_forward2_$1"
}

sample_count() {
    find "$(output_path "$1")/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

running_pids() {
    local exp_name="$1"
    pgrep -f "forward_code/onion.py.*$(output_path "${exp_name}")" || true
}

merge_one() {
    local exp_name="$1"
    local n_sim="$2"
    local trigger_mode="$3"
    local action_mode="$4"
    local out
    out="$(output_path "${exp_name}")"
    local summary_log="${out}/accuracy.log"
    local count
    count="$(sample_count "${exp_name}")"

    if [[ -f "${summary_log}" ]]; then
        echo "[$(timestamp)] already merged ${exp_name}: $(cat "${summary_log}")" | tee -a "${WATCH_LOG}"
        return 0
    fi
    if [[ "${count}" -lt 1145 ]]; then
        echo "[$(timestamp)] waiting ${exp_name}: ${count}/1145" | tee -a "${WATCH_LOG}"
        return 1
    fi

    echo "[$(timestamp)] merge ${exp_name}" | tee -a "${WATCH_LOG}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${out}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        --use_image_enhance \
        --mcts_n_simulations "${n_sim}" \
        --mcts_trigger_mode "${trigger_mode}" \
        --mcts_action_mode "${action_mode}" \
        --mcts_filter_objects \
        --summary_log "${summary_log}" \
        >> "${LOG_ROOT}/${exp_name}/${exp_name}_merge.log" 2>&1
}

make_marker_diagnostic() {
    local marker_out
    local outline_out
    local report_path
    marker_out="$(output_path mcts_marker_narrow_no_cot_rounds1_n10_4shards)"
    outline_out="$(output_path mcts_narrow_no_cot_rounds1_n10_6shards)"
    report_path="${REPORT_DIR}/marker_vs_outline_n10_diagnostic.md"
    if [[ -f "${report_path}" ]]; then
        return 0
    fi
    if [[ ! -f "${marker_out}/accuracy.log" || ! -d "${outline_out}/prompt_samples" ]]; then
        return 1
    fi

    "${PYTHON_BIN}" "${REPORT_DIR}/compare_two_runs.py" \
        --left-name "outline_n10" \
        --left-dir "${outline_out}/prompt_samples" \
        --right-name "marker_n10" \
        --right-dir "${marker_out}/prompt_samples" \
        --out-csv "${REPORT_DIR}/marker_vs_outline_n10_diagnostic.csv" \
        --out-md "${report_path}" \
        >> "${LOG_ROOT}/marker_vs_outline_diagnostic.log" 2>&1
    echo "[$(timestamp)] marker vs outline diagnostic written: ${report_path}" | tee -a "${WATCH_LOG}"
}

echo "[$(timestamp)] marker MCTS merge watcher started" | tee -a "${WATCH_LOG}"
while true; do
    done_count=0
    for item in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name n_sim trigger_mode action_mode <<< "${item}"
        if merge_one "${exp_name}" "${n_sim}" "${trigger_mode}" "${action_mode}"; then
            done_count=$((done_count + 1))
        fi
    done

    make_marker_diagnostic || true

    if [[ "${done_count}" -eq "${#EXPERIMENTS[@]}" && -f "${REPORT_DIR}/marker_vs_outline_n10_diagnostic.md" ]]; then
        break
    fi

    active=""
    for item in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name _ _ _ <<< "${item}"
        active+=$(running_pids "${exp_name}")
    done
    if [[ -z "${active}" ]]; then
        echo "[$(timestamp)] no running marker MCTS processes; watcher still waiting for merge/diagnostic conditions" | tee -a "${WATCH_LOG}"
    fi
    sleep 600
done
echo "[$(timestamp)] marker MCTS merge watcher finished" | tee -a "${WATCH_LOG}"
