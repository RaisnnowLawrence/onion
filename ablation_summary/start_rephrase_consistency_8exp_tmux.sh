#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-rephrase_$(date +%Y%m%d_%H%M%S)}"
SCRIPT=/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_rephrase_consistency_8exp_3shards.sh

start_one() {
  local session="$1"
  local exp="$2"
  local gpu="$3"
  tmux new-session -d -s "${session}" "cd /data2/lizhengxue/WorkSpace/onion && RUN_ID=${RUN_ID} ${SCRIPT} ${exp} ${gpu}"
  echo "started ${session}: ${exp} on GPU ${gpu}"
}

start_one "rephrase_${RUN_ID}_g0" rephrase_keep_trace_mixed 0
start_one "rephrase_${RUN_ID}_g1" rephrase_majority2_mixed 1
start_one "rephrase_${RUN_ID}_g2" rephrase_review2_mixed 2
start_one "rephrase_${RUN_ID}_g3" rephrase_allagree_mixed 3
start_one "rephrase_${RUN_ID}_g4" rephrase_review2_visual_focus 4
start_one "rephrase_${RUN_ID}_g5" rephrase_review2_answer_type 5
start_one "rephrase_${RUN_ID}_g6" rephrase_review2_regional 6
start_one "rephrase_${RUN_ID}_g7" rephrase_review2_risky_only 7

echo "RUN_ID=${RUN_ID}"
echo "logs: /data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_rephrase_consistency_8exp_3shards/${RUN_ID}"
echo "outputs: /data2/lizhengxue/WorkSpace/onion_output/aokvqa/rephrase_consistency_8exp/${RUN_ID}"
