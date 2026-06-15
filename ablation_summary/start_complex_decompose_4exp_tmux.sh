#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/onion
OUT=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
RUN="${BASE}/ablation_summary/run_complex_decompose_4exp_3shards.sh"

mkdir -p "${OUT}"

for session in \
  complex_decomp_gpu0 \
  complex_adapt_gpu1 \
  complex_adapt_verify_gpu2 \
  complex_conservative_gpu3
do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

tmux new-session -d -s complex_decomp_gpu0 \
  "NUM_SHARDS=3 ${RUN} always 0 2>&1 | tee ${OUT}/complex_decompose_always_gpu0.master.out"

tmux new-session -d -s complex_adapt_gpu1 \
  "NUM_SHARDS=3 ${RUN} adaptive 1 2>&1 | tee ${OUT}/complex_decompose_adaptive_gpu1.master.out"

tmux new-session -d -s complex_adapt_verify_gpu2 \
  "NUM_SHARDS=3 ${RUN} adaptive_verify 2 2>&1 | tee ${OUT}/complex_decompose_adaptive_verify_gpu2.master.out"

tmux new-session -d -s complex_conservative_gpu3 \
  "NUM_SHARDS=3 ${RUN} conservative_verify 3 2>&1 | tee ${OUT}/complex_decompose_conservative_verify_gpu3.master.out"

tmux ls
