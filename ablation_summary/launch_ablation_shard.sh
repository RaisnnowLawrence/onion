#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 EXP_NAME GPU SHARD" >&2
    exit 2
fi

EXP_NAME="$1"
GPU="$2"
SHARD="$3"

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=/data2/lizhengxue/anaconda3/envs/sam/bin/python
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs"
CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags

mkdir -p "${LOG_DIR}" "${CACHE_PATH}"

extra_args=()
case "${EXP_NAME}" in
    remove_caption)
        extra_args=(--remove_caption)
        ;;
    rounds1)
        extra_args=(--rounds 1)
        ;;
    rounds3)
        extra_args=(--rounds 3)
        ;;
    nshot0)
        extra_args=(--n_shot 0)
        ;;
    nshot4)
        extra_args=(--n_shot 4)
        ;;
    sim_question)
        extra_args=(--similarity_metric question)
        ;;
    caption_vinvl_tag)
        extra_args=(--caption_type vinvl_tag)
        ;;
    caption_vinvl_sg)
        extra_args=(--caption_type vinvl_sg)
        ;;
    *)
        echo "unknown experiment: ${EXP_NAME}" >&2
        exit 2
        ;;
esac

output_path="${OUT_ROOT}/${ENGINE}_forward_${EXP_NAME}"
log_path="${LOG_DIR}/${EXP_NAME}_gpu${GPU}_shard${SHARD}_manual.log"

existing="$(pgrep -f "forward_code/onion.py.*${output_path}.*--shard_id ${SHARD} " || true)"
if [[ -n "${existing}" ]]; then
    echo "${EXP_NAME} shard ${SHARD} already running: ${existing}"
    exit 0
fi

setsid env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
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
    --chain_of_thoughts \
    "${extra_args[@]}" \
    --shard_id "${SHARD}" --num_shards 2 \
    > "${log_path}" 2>&1 < /dev/null &

echo "$!"
