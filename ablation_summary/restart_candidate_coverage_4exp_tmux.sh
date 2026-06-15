#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/onion
OUT=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
RUN="${BASE}/ablation_summary/run_candidate_coverage_4exp_3shards.sh"

mkdir -p "${OUT}"

for session in \
  cand_cov_gpu0 \
  cand_count_gpu1 \
  cand_ocr_gpu2 \
  cand_diverse_gpu3 \
  candidate_cov_smoke
do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

tmux new-session -d -s cand_cov_gpu0 \
  "NUM_SHARDS=3 SHARD_PARALLEL=1 ${RUN} coverage_scan 0 2>&1 | tee ${OUT}/candidate_coverage_scan_gpu0.rerun3shards.master.out"

tmux new-session -d -s cand_count_gpu1 \
  "NUM_SHARDS=3 SHARD_PARALLEL=1 ${RUN} count_specialist 1 2>&1 | tee ${OUT}/candidate_count_specialist_gpu1.rerun3shards.master.out"

tmux new-session -d -s cand_ocr_gpu2 \
  "NUM_SHARDS=3 SHARD_PARALLEL=1 ${RUN} ocr_specialist 2 2>&1 | tee ${OUT}/candidate_ocr_specialist_gpu2.rerun3shards.master.out"

tmux new-session -d -s cand_diverse_gpu3 \
  "NUM_SHARDS=3 SHARD_PARALLEL=1 ${RUN} diverse_pool 3 2>&1 | tee ${OUT}/candidate_diverse_pool_gpu3.rerun3shards.master.out"

tmux ls
