#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs"
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5}
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
export PYTHONUNBUFFERED=1

CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags

mkdir -p "${LOG_DIR}" "${BASE}/logs" "${CACHE_PATH}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

sample_count() {
    local exp_name="$1"
    find "${OUT_ROOT}/${ENGINE}_forward_${exp_name}/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

accuracy_log() {
    local exp_name="$1"
    echo "${OUT_ROOT}/${ENGINE}_forward_${exp_name}/accuracy.log"
}

is_merged() {
    local exp_name="$1"
    [[ "$(sample_count "${exp_name}")" -ge 1145 && -f "$(accuracy_log "${exp_name}")" ]]
}

cleanup_exp_processes() {
    local exp_name="$1"
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"
    local pids
    pids="$(pgrep -f "forward_code/onion.py.*${output_path}" || true)"
    if [[ -n "${pids}" ]]; then
        echo "[$(timestamp)] cleanup ${exp_name}: kill ${pids}" | tee -a "${REPORT_DIR}/followup_master.log"
        kill ${pids} 2>/dev/null || true
        sleep 5
        pids="$(pgrep -f "forward_code/onion.py.*${output_path}" || true)"
        if [[ -n "${pids}" ]]; then
            echo "[$(timestamp)] cleanup ${exp_name}: force kill ${pids}" | tee -a "${REPORT_DIR}/followup_master.log"
            kill -9 ${pids} 2>/dev/null || true
        fi
    fi
}

merge_exp() {
    local exp_name="$1"
    local with_cot="$2"
    local extra_args="$3"
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"
    local summary_log
    summary_log="$(accuracy_log "${exp_name}")"

    if [[ "$(sample_count "${exp_name}")" -lt 1145 ]]; then
        echo "[$(timestamp)] ${exp_name}: skip merge, only $(sample_count "${exp_name}") samples" | tee -a "${REPORT_DIR}/followup_master.log"
        return 0
    fi
    if [[ -f "${summary_log}" ]]; then
        echo "[$(timestamp)] ${exp_name}: already merged" | tee -a "${REPORT_DIR}/followup_master.log"
        return 0
    fi

    local cot_arg=()
    if [[ "${with_cot}" == "cot" ]]; then
        cot_arg=(--chain_of_thoughts)
    fi

    echo "[$(timestamp)] ${exp_name}: merge" | tee -a "${REPORT_DIR}/followup_master.log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        "${cot_arg[@]}" \
        --summary_log "${summary_log}" \
        ${extra_args} \
        >> "${LOG_DIR}/${exp_name}_merge.log" 2>&1
}

wait_current_experiment() {
    local exp_name="$1"
    local with_cot="$2"
    local extra_args="$3"
    while [[ "$(sample_count "${exp_name}")" -lt 1145 ]]; do
        echo "[$(timestamp)] waiting ${exp_name}: $(sample_count "${exp_name}")/1145" | tee -a "${REPORT_DIR}/followup_master.log"
        sleep 300
    done
    merge_exp "${exp_name}" "${with_cot}" "${extra_args}"
    cleanup_exp_processes "${exp_name}"
}

run_exp_same_gpu() {
    local exp_name="$1"
    local gpu="$2"
    local with_cot="$3"
    local extra_args="$4"
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"

    if is_merged "${exp_name}"; then
        echo "[$(timestamp)] ${exp_name}: already done" | tee -a "${REPORT_DIR}/followup_master.log"
        return 0
    fi

    local cot_arg=""
    if [[ "${with_cot}" == "cot" ]]; then
        cot_arg="--chain_of_thoughts"
    fi

    echo "[$(timestamp)] launch ${exp_name} on GPU ${gpu}: ${extra_args} ${cot_arg}" | tee -a "${REPORT_DIR}/followup_master.log"
    for shard in 0 1; do
        local log_path="${LOG_DIR}/${exp_name}_samegpu_shard${shard}.log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
            --cache_path "${CACHE_PATH}" \
            --output_path "${output_path}" \
            --caption_type vinvl \
            --n_shot 1 \
            --n_ensemble 5 \
            --rounds 5 \
            --iterative_strategy caption \
            --engine "${ENGINE}" \
            --sg_path "${SG_PATH}" \
            --train_sim_metric answer \
            --train_sim_file "${TRAIN_SIM_FILE}" \
            --tag_path "${TAG_PATH}" \
            --with_clip_verify \
            ${cot_arg} \
            ${extra_args} \
            --shard_id "${shard}" --num_shards 2 \
            >> "${log_path}" 2>&1 &
    done
    wait
    merge_exp "${exp_name}" "${with_cot}" "${extra_args}"
    cleanup_exp_processes "${exp_name}"
    "${PYTHON_BIN}" "${REPORT_DIR}/collect_ablation_report.py" >> "${REPORT_DIR}/followup_master.log" 2>&1
    echo "[$(timestamp)] done ${exp_name}" | tee -a "${REPORT_DIR}/followup_master.log"
}

run_followups() {
    local experiments=(
        "remove_caption|cot|--remove_caption"
        "rounds1|cot|--rounds 1"
        "rounds3|cot|--rounds 3"
        "ensemble1|cot|--n_ensemble 1"
        "ensemble3|cot|--n_ensemble 3"
        "nshot0|cot|--n_shot 0"
        "nshot4|cot|--n_shot 4"
        "sim_question|cot|--similarity_metric question"
        "caption_vinvl_tag|cot|--caption_type vinvl_tag"
        "caption_vinvl_sg|cot|--caption_type vinvl_sg"
    )

    local active=0
    local idx=0
    for item in "${experiments[@]}"; do
        IFS='|' read -r exp_name with_cot extra_args <<< "${item}"
        local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        run_exp_same_gpu "${exp_name}" "${gpu}" "${with_cot}" "${extra_args}" &
        active=$((active + 1))
        idx=$((idx + 1))
        if [[ "${active}" -ge "${#GPUS[@]}" ]]; then
            wait -n
            active=$((active - 1))
        fi
    done
    wait
}

main() {
    echo "[$(timestamp)] follow-up controller started" | tee -a "${REPORT_DIR}/followup_master.log"
    "${PYTHON_BIN}" "${REPORT_DIR}/collect_ablation_report.py" >> "${REPORT_DIR}/followup_master.log" 2>&1

    wait_current_experiment baseline cot ""
    wait_current_experiment qwen_caption cot "--use_qwen_blip2_caption"
    wait_current_experiment qwen_thought cot "--use_qwen_blip2_thought_verify"
    wait_current_experiment no_cot no_cot ""

    "${PYTHON_BIN}" "${REPORT_DIR}/collect_ablation_report.py" >> "${REPORT_DIR}/followup_master.log" 2>&1
    run_followups
    "${PYTHON_BIN}" "${REPORT_DIR}/collect_ablation_report.py" >> "${REPORT_DIR}/followup_master.log" 2>&1
    echo "[$(timestamp)] follow-up controller finished" | tee -a "${REPORT_DIR}/followup_master.log"
}

main "$@"
