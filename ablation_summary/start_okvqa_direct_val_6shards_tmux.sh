#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_direct_val_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
OUT="/data2/lizhengxue/WorkSpace/onion_output/okvqa/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa/${RUN_ID}"
CACHE="/data2/lizhengxue/WorkSpace/onion_output/okvqa/cache_${RUN_ID}"

mkdir -p "${OUT}" "${LOGDIR}" "${CACHE}"

BASE_ARGS=(
  forward_code/onion.py
  --dataset_name okvqa
  --split_name val
  --engine qwen3-VL-4B
  --coco_path /data2/lizhengxue/datasets/okvqa
  --raw_image_dir /data2/lizhengxue/datasets/okvqa
  --output_path "${OUT}"
  --cache_path "${CACHE}"
  --caption_type vinvl_tag
  --valcaption_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/vinvl_caption/VinVL_base_val2014.tsv
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
  --concept_caption_path scene_graph_coco17_caption
  --similarity_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_clip_new
  --similarity_metric imagequestion
  --train_sim_metric answer
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_okvqa.pk
  --n_shot 0
  --n_ensemble 1
  --rounds 1
  --num_shards 6
)

launch_one() {
  local shard_id="$1"
  local gpu_id="$2"
  local session="okvqa_${RUN_ID}_s${shard_id}"
  local log_file="${LOGDIR}/shard_${shard_id}_gpu${gpu_id}.log"
  tmux new-session -d -s "${session}" \
    "cd '${REPO_DIR}' && CUDA_VISIBLE_DEVICES=${gpu_id} '${PYTHON}' ${BASE_ARGS[*]} --shard_id ${shard_id} > '${log_file}' 2>&1"
  echo "started session=${session} gpu=${gpu_id} shard=${shard_id} log=${log_file}"
}

launch_one 0 4
launch_one 1 4
launch_one 2 4
launch_one 3 5
launch_one 4 5
launch_one 5 5

echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"
echo "LOGDIR=${LOGDIR}"

