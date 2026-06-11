#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs_followup2"
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
WATCH_LOG="${REPORT_DIR}/followup2_manual_merge_watch.log"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

output_path_for() {
    echo "${OUT_ROOT}/${ENGINE}_forward2_$1"
}

sample_count() {
    local exp_name="$1"
    find "$(output_path_for "${exp_name}")/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

accuracy_log() {
    local exp_name="$1"
    echo "$(output_path_for "${exp_name}")/accuracy.log"
}

merge_exp() {
    local exp_name="$1"
    local extra_args="$2"
    local output_path
    output_path="$(output_path_for "${exp_name}")"
    local summary_log
    summary_log="$(accuracy_log "${exp_name}")"

    if [[ -f "${summary_log}" ]]; then
        return 0
    fi
    if [[ "$(sample_count "${exp_name}")" -lt 1145 ]]; then
        return 1
    fi

    echo "[$(timestamp)] merge ${exp_name}" | tee -a "${WATCH_LOG}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        --chain_of_thoughts \
        ${extra_args} \
        --summary_log "${summary_log}" \
        >> "${LOG_DIR}/${exp_name}_manual_merge.log" 2>&1
}

all_done() {
    local exp_name
    for exp_name in context_no_round_state qwen_caption_local qwen_caption_short answer_strict_final; do
        if [[ ! -f "$(accuracy_log "${exp_name}")" ]]; then
            return 1
        fi
    done
    return 0
}

echo "[$(timestamp)] manual merge watcher started" | tee -a "${WATCH_LOG}"
while ! all_done; do
    merge_exp context_no_round_state "--context_mode no_round_state" || true
    merge_exp qwen_caption_local "--use_qwen_blip2_caption --qwen_caption_mode local" || true
    merge_exp qwen_caption_short "--use_qwen_blip2_caption --qwen_caption_max_tokens 48 --qwen_caption_final_max_chars 220" || true
    merge_exp answer_strict_final "--answer_extraction_strategy strict_final" || true
    sleep 600
done
echo "[$(timestamp)] manual merge watcher finished" | tee -a "${WATCH_LOG}"
