#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <wait_job_1> <wait_job_2> <run_id>" >&2
  exit 2
fi

WAIT_JOB_1="$1"
WAIT_JOB_2="$2"
RUN_ID="$3"
TS_BIN="${TS_BIN:-/data2/lizhengxue/.local/bin/ts}"
SCRIPT_DIR="/data2/lizhengxue/WorkSpace/onion/ablation_summary"
RUN_DIR="/data2/lizhengxue/WorkSpace/onion_output/gqa/${RUN_ID}"
PY="/data2/lizhengxue/anaconda3/envs/sam/bin/python"

echo "[audit finish] waiting for jobs ${WAIT_JOB_1} and ${WAIT_JOB_2}"
"$TS_BIN" -w "$WAIT_JOB_1" >/dev/null
"$TS_BIN" -w "$WAIT_JOB_2" >/dev/null

env NUM_SHARDS=9 SPLIT_NAME=testdev \
  DYFO_DECISION_MODE=token_confidence_override \
  DYFO_ANSWER_IMAGE_MODE=concat_horizontal \
  DYFO_NODE_ANSWER_IMAGE_MODE=crop \
  DYFO_TRIGGER_MODE=always DYFO_FORCE_RUN_ALL_SAMPLES=1 \
  DYFO_TEXT_FOCUS_USE_IMAGE=1 DYFO_FOCUS_PADDING=1.2 \
  DYFO_TOKEN_CONFIDENCE_THRESHOLD=0.95 DYFO_TOKEN_CONFIDENCE_MARGIN=0.0 \
  bash "$SCRIPT_DIR/merge_gqa_testdev_dyfo_focus_image_run.sh" "$RUN_ID"

PYTHONPATH="/data2/lizhengxue/WorkSpace/onion/forward_code" \
  "$PY" "$SCRIPT_DIR/analyze_gqa_dyfo_region_audit.py" --run_dir "$RUN_DIR"

echo "[audit finish] report=${RUN_DIR}/region_audit_report/REGION_AUDIT_REPORT.md"
