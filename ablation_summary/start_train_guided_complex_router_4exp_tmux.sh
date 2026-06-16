#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/data2/lizhengxue/WorkSpace/onion}
OUT=${OUT:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
RUN="${BASE}/ablation_summary/run_train_guided_complex_router_4exp_3shards.sh"
RUN_ID=${PROFILE_RUN_ID:-trainrag10k_$(date +%Y%m%d_%H%M%S)}

mkdir -p "${OUT}"

for session in \
  trainrag_direct_failure \
  trainrag_direct_vs_complex \
  trainrag_qtype_conditional \
  trainrag_conservative_risk
do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

tmux new-session -d -s trainrag_direct_failure \
  "PROFILE_RUN_ID=${RUN_ID} TRAIN_PROFILE_TOTAL=10000 NUM_SHARDS=3 ${RUN} direct_failure 0 2>&1 | tee ${OUT}/trainrag_direct_failure_${RUN_ID}.master.out"

tmux new-session -d -s trainrag_direct_vs_complex \
  "PROFILE_RUN_ID=${RUN_ID} TRAIN_PROFILE_TOTAL=10000 NUM_SHARDS=3 ${RUN} direct_vs_complex 1 2>&1 | tee ${OUT}/trainrag_direct_vs_complex_${RUN_ID}.master.out"

tmux new-session -d -s trainrag_qtype_conditional \
  "PROFILE_RUN_ID=${RUN_ID} TRAIN_PROFILE_TOTAL=10000 NUM_SHARDS=3 ${RUN} qtype_conditional 2 2>&1 | tee ${OUT}/trainrag_qtype_conditional_${RUN_ID}.master.out"

tmux new-session -d -s trainrag_conservative_risk \
  "PROFILE_RUN_ID=${RUN_ID} TRAIN_PROFILE_TOTAL=10000 NUM_SHARDS=3 ${RUN} conservative_risk 3 2>&1 | tee ${OUT}/trainrag_conservative_risk_${RUN_ID}.master.out"

echo "run_id=${RUN_ID}"
tmux ls | grep trainrag_
