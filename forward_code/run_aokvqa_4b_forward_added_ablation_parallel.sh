#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
ENGINE=qwen3-VL-4B
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
EXP_GPU_LIST=${EXP_GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a EXP_GPUS <<< "${EXP_GPU_LIST}"
export PYTHONUNBUFFERED=1

CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa

mkdir -p "${BASE}/logs" "${CACHE_PATH}"

common_args=(
    --cache_path "${CACHE_PATH}"
    --caption_type vinvl
    --n_shot 1
    --n_ensemble 5
    --rounds 5
    --iterative_strategy caption
    --engine "${ENGINE}"
    --sg_path "${SG_PATH}"
    --train_sim_metric answer
    --train_sim_file "${TRAIN_SIM_FILE}"
    --with_clip_verify
    --chain_of_thoughts
    --shard_id 0
    --num_shards 1
)

merge_exp() {
    local exp_name="$1"
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only --chain_of_thoughts \
        --summary_log "${output_path}/accuracy.log"
}

run_exp() {
    local exp_name="$1"
    local gpu_id="$2"
    shift 2
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"
    local log_path="${BASE}/logs/aokvqa_${ENGINE}_forward_${exp_name}_parallel.log"

    {
        echo "=============================="
        echo "Start ${exp_name}"
        echo "GPU: ${gpu_id}"
        echo "Output: ${output_path}"
        echo "Extra args: $*"
        echo "=============================="
    } >> "${log_path}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        "${common_args[@]}" \
        --output_path "${output_path}" \
        "$@" \
        >> "${log_path}" 2>&1

    merge_exp "${exp_name}" >> "${log_path}" 2>&1
    echo "Done ${exp_name}" >> "${log_path}"
}

experiments=(
    "baseline::"
    "ocr::--use_ocr_context"
    "clip_thought::--use_clip_thought_verify"
    "qwen_caption::--use_qwen_blip2_caption"
    "qwen_thought::--use_qwen_blip2_thought_verify"
    "all_regions::--use_all_regional_captions"
    "ensemble_norm::--ensemble_strategy normalized_majority"
    "all_added::--use_ocr_context --use_clip_thought_verify --use_qwen_blip2_caption --use_qwen_blip2_thought_verify --use_all_regional_captions --ensemble_strategy normalized_majority"
)

for idx in "${!experiments[@]}"; do
    item="${experiments[$idx]}"
    exp_name="${item%%::*}"
    extra="${item#*::}"
    gpu_id="${EXP_GPUS[$((idx % ${#EXP_GPUS[@]}))]}"
    # shellcheck disable=SC2086
    run_exp "${exp_name}" "${gpu_id}" ${extra} &
    echo "Launched ${exp_name} on GPU ${gpu_id}"
done

wait

summary_path="${OUT_ROOT}/${ENGINE}_forward_added_parallel_summary.log"
: > "${summary_path}"
for item in "${experiments[@]}"; do
    exp_name="${item%%::*}"
    logfile="${OUT_ROOT}/${ENGINE}_forward_${exp_name}/accuracy.log"
    echo -n "${exp_name}: " | tee -a "${summary_path}"
    cat "${logfile}" 2>/dev/null | tee -a "${summary_path}" || echo "未生成" | tee -a "${summary_path}"
done

echo "全部并行新增模块消融实验完成"
echo "Summary: ${summary_path}"
