#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-directbase_$(date +%Y%m%d_%H%M%S)}"
SCRIPT=/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_direct_base_5exp_3shards.sh

start_one() {
  local session="$1"
  local exp="$2"
  local gpu="$3"
  tmux new-session -d -s "${session}" "cd /data2/lizhengxue/WorkSpace/onion && RUN_ID=${RUN_ID} ${SCRIPT} ${exp} ${gpu}"
  echo "started ${session}: ${exp} on GPU ${gpu}"
}

start_one "directbase_${RUN_ID}_g7" direct_baseline_no_cot_rounds1 7
start_one "directbase_${RUN_ID}_g5" direct_safe_postprocess 5
start_one "directbase_${RUN_ID}_g4" direct_answer_first_strict 4
start_one "directbase_${RUN_ID}_g0" direct_type_specialist 0
start_one "directbase_${RUN_ID}_g1" direct_context_gated 1

echo "RUN_ID=${RUN_ID}"
echo "GPU2 left as spare."
echo "logs: /data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_direct_base_5exp_3shards/${RUN_ID}"
echo "outputs: /data2/lizhengxue/WorkSpace/onion_output/aokvqa/direct_base_5exp/${RUN_ID}"
