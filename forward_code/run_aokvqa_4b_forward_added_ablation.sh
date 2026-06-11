#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
ENGINE=qwen3-VL-4B
GPU_LIST=${GPU_LIST:-6,7}
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_SHARDS=${NUM_SHARDS:-${#GPU_IDS[@]}}
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
export PYTHONUNBUFFERED=1

CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa

mkdir -p "${BASE}/logs" "${CACHE_PATH}"

run_exp() {
    local exp_name="$1"
    shift
    local output_path="${OUT_ROOT}/${ENGINE}_forward_${exp_name}"
    local common_args=(
        --cache_path "${CACHE_PATH}"
        --output_path "${output_path}"
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
    )

    echo "=============================="
echo "Start ${exp_name}"
    echo "Output: ${output_path}"
    echo "Extra args: $*"
    echo "GPU_LIST: ${GPU_LIST}; NUM_SHARDS: ${NUM_SHARDS}"
    echo "=============================="

    if [[ "${NUM_SHARDS}" == "1" ]]; then
        local gpu_id="${GPU_IDS[0]}"
        echo "Shard 0 -> GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
            "${common_args[@]}" "$@" \
            --shard_id 0 --num_shards 1 \
            >> "${BASE}/logs/aokvqa_${ENGINE}_forward_${exp_name}_shard0.log" 2>&1
    else
        for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
            local gpu_id="${GPU_IDS[$((shard_id % ${#GPU_IDS[@]}))]}"
            echo "Shard ${shard_id} -> GPU ${gpu_id}"
            CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
                "${common_args[@]}" "$@" \
                --shard_id "${shard_id}" --num_shards "${NUM_SHARDS}" \
                >> "${BASE}/logs/aokvqa_${ENGINE}_forward_${exp_name}_shard${shard_id}.log" 2>&1 &
        done
        wait
    fi

    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only --chain_of_thoughts \
        --summary_log "${output_path}/accuracy.log"

    echo "Done ${exp_name}"
}

run_exp baseline
run_exp ocr --use_ocr_context
run_exp clip_thought --use_clip_thought_verify
run_exp qwen_caption --use_qwen_blip2_caption
run_exp qwen_thought --use_qwen_blip2_thought_verify
run_exp all_regions --use_all_regional_captions
run_exp ensemble_norm --ensemble_strategy normalized_majority
run_exp all_added \
    --use_ocr_context \
    --use_clip_thought_verify \
    --use_qwen_blip2_caption \
    --use_qwen_blip2_thought_verify \
    --use_all_regional_captions \
    --ensemble_strategy normalized_majority

echo "全部新增模块消融实验完成"
echo "=============================="
echo "准确率汇总"
echo "=============================="
for exp in baseline ocr clip_thought qwen_caption qwen_thought all_regions ensemble_norm all_added; do
    logfile="${OUT_ROOT}/${ENGINE}_forward_${exp}/accuracy.log"
    echo -n "${exp}: "
    cat "${logfile}" 2>/dev/null || echo "未生成"
done
