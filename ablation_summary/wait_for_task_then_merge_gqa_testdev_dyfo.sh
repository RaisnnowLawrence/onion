#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <task_id_to_wait_for> <run_id>" >&2
  exit 2
fi

WAIT_JOB_ID="$1"
RUN_ID="$2"
TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
MERGE_SCRIPT="/data2/lizhengxue/WorkSpace/onion/ablation_summary/merge_gqa_testdev_dyfo_focus_image_run.sh"

echo "[merge barrier] waiting for task ${WAIT_JOB_ID}"
"$TS_BIN" -w "$WAIT_JOB_ID" >/dev/null
echo "[merge barrier] task ${WAIT_JOB_ID} finished; starting merge for ${RUN_ID}"
exec bash "$MERGE_SCRIPT" "$RUN_ID"
