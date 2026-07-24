#!/usr/bin/env bash
set -euo pipefail

TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
BATCH_ID="${BATCH_ID:-gqa_testdev_dyfo_dual_vs_single_$(date +%Y%m%d_%H%M%S)}"
DUAL_RUN_ID="${DUAL_RUN_ID:-gqa_testdev_dyfo_dual_expert_node_concat_full_9shards_${BATCH_ID}}"
SINGLE_RUN_ID="${SINGLE_RUN_ID:-gqa_testdev_dyfo_single_expert_node_concat_full_9shards_${BATCH_ID}}"
TS_SOCKET="${TS_SOCKET:-/tmp/ts_onion_${BATCH_ID}.sock}"
CONTROLLER="/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_gqa_testdev_dyfo_wait_three_gpus_9shards.sh"

export TS_SOCKET
"$TS_BIN" -S 1

dual_job=$(
  "$TS_BIN" -L dual_expert_full env \
    CANDIDATE_GPUS="0 1 2 3 4 5 6 7" \
    MIN_FREE_MB=46080 GPU_WAIT_SECONDS=60 SHARD_LAUNCH_DELAY=180 \
    DYFO_DUAL_VISUAL_EXPERTS=1 \
    DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal \
    bash "$CONTROLLER" "$DUAL_RUN_ID"
)

single_job=$(
  "$TS_BIN" -D "$dual_job" -L single_expert_full env \
    CANDIDATE_GPUS="0 1 2 3 4 5 6 7" \
    MIN_FREE_MB=46080 GPU_WAIT_SECONDS=60 SHARD_LAUNCH_DELAY=180 \
    DYFO_DUAL_VISUAL_EXPERTS=0 \
    DYFO_NODE_ANSWER_IMAGE_MODE=concat_horizontal \
    bash "$CONTROLLER" "$SINGLE_RUN_ID"
)

echo "[queue] BATCH_ID=${BATCH_ID}"
echo "[queue] TS_SOCKET=${TS_SOCKET}"
echo "[queue] dual_job=${dual_job} run_id=${DUAL_RUN_ID}"
echo "[queue] single_job=${single_job} run_id=${SINGLE_RUN_ID} depends_on=${dual_job}"
echo "[queue] requirement=three GPUs each with >=46080 MiB free; three shards per GPU"
"$TS_BIN" -l
