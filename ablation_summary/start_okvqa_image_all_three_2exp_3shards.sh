#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-okvqa_image_all_three_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/data2/lizhengxue/WorkSpace/onion"
PYTHON="/data2/lizhengxue/anaconda3/envs/sam/bin/python"
ROOT_OUT="/data2/lizhengxue/WorkSpace/onion_output/okvqa/three_enhance_ablation/${RUN_ID}"
LOGDIR="/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_okvqa_three_enhance/${RUN_ID}"

mkdir -p "${ROOT_OUT}" "${LOGDIR}"

COMMON_ARGS=(
  forward_code/onion.py
  --dataset_name okvqa
  --split_name val
  --engine qwen3-VL-4B
  --coco_path /data2/lizhengxue/datasets/okvqa
  --raw_image_dir /data2/lizhengxue/datasets/coco14
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
  --num_shards 3
  --mcts_n_simulations 5
)

launch_one() {
  local exp_name="$1"
  local gpu_id="$2"
  local shard_id="$3"
  shift 3
  local extra_args=("$@")
  local out_dir="${ROOT_OUT}/qwen3-VL-4B_${exp_name}"
  local cache_dir="${ROOT_OUT}/cache_${exp_name}"
  local session="okvqa_${RUN_ID}_${exp_name}_s${shard_id}"
  local log_file="${LOGDIR}/${exp_name}_shard_${shard_id}_gpu${gpu_id}.log"
  mkdir -p "${out_dir}" "${cache_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${REPO_DIR}' && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${gpu_id} '${PYTHON}' ${COMMON_ARGS[*]} --output_path '${out_dir}' --cache_path '${cache_dir}' ${extra_args[*]} --shard_id ${shard_id} > '${log_file}' 2>&1"
  echo "started exp=${exp_name} session=${session} gpu=${gpu_id} shard=${shard_id} log=${log_file}"
}

for shard_id in 0 1 2; do
  launch_one image_only 4 "${shard_id}" --use_image_enhance
  launch_one all_three 7 "${shard_id}" --use_image_enhance --use_caption_enhance --use_knowledge_enhance
  if [[ "${shard_id}" != "2" ]]; then
    echo "waiting 120s before launching next shard wave..."
    sleep 120
  fi
done

cat <<EOF
RUN_ID=${RUN_ID}
ROOT_OUT=${ROOT_OUT}
LOGDIR=${LOGDIR}
EOF
