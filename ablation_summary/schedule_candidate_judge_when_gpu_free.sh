#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data2/lizhengxue/WorkSpace/onion}
LAUNCHER=${LAUNCHER:-${REPO}/ablation_summary/run_candidate_judge_8ablations_3shards.sh}
REPORT_DIR=${REPORT_DIR:-/data2/lizhengxue/WorkSpace/onion_output/ablation_summary}
LOG_DIR=${LOG_DIR:-${REPORT_DIR}/logs_candidate_judge_scheduler}
STATE_FILE=${STATE_FILE:-${LOG_DIR}/launched_experiments.state}
SCHED_LOG=${SCHED_LOG:-${LOG_DIR}/scheduler.log}

MIN_FREE_MIB=${MIN_FREE_MIB:-40000}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
REQUIRED_STABLE_CHECKS=${REQUIRED_STABLE_CHECKS:-10}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
MAX_LAUNCHES=${MAX_LAUNCHES:-8}

mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}"

priority_queue=(
  core
  caption_candidate
  regions_ocr
  strict_consensus3
  routed_caption_knowledge
  always_judge
  marker_mcts
  allow_new_answer
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${SCHED_LOG}"
}

is_gpu_allowed() {
  local gpu="$1"
  local item
  IFS=',' read -ra items <<< "${GPU_LIST}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${gpu}" ]]; then
      return 0
    fi
  done
  return 1
}

already_launched() {
  local exp="$1"
  grep -q "^${exp}[[:space:]]" "${STATE_FILE}"
}

next_experiment() {
  local exp
  for exp in "${priority_queue[@]}"; do
    if ! already_launched "${exp}"; then
      echo "${exp}"
      return 0
    fi
  done
  return 1
}

launch_experiment() {
  local exp="$1"
  local gpu="$2"
  local launch_log="${LOG_DIR}/${exp}_gpu${gpu}_launcher.log"

  log "launch ${exp} on GPU ${gpu}; log=${launch_log}"
  nohup "${LAUNCHER}" "${exp}" "${gpu}" > "${launch_log}" 2>&1 < /dev/null &
  local pid="$!"
  LAST_LAUNCH_PID="${pid}"
  printf "%s\t%s\t%s\t%s\n" "${exp}" "${gpu}" "${pid}" "$(timestamp)" >> "${STATE_FILE}"
}

gpu_free_memory_snapshot() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
}

declare -A stable_counts
declare -A running_pid_by_gpu

log "scheduler started: MIN_FREE_MIB=${MIN_FREE_MIB}, CHECK_INTERVAL=${CHECK_INTERVAL}, REQUIRED_STABLE_CHECKS=${REQUIRED_STABLE_CHECKS}, GPU_LIST=${GPU_LIST}, MAX_LAUNCHES=${MAX_LAUNCHES}"
log "state file: ${STATE_FILE}"

launch_count="$(wc -l < "${STATE_FILE}")"

while [[ "${launch_count}" -lt "${MAX_LAUNCHES}" ]]; do
  snapshot="$(gpu_free_memory_snapshot)"

  while IFS=',' read -r gpu free_mib; do
    gpu="$(echo "${gpu}" | xargs)"
    free_mib="$(echo "${free_mib}" | xargs)"

    if ! is_gpu_allowed "${gpu}"; then
      continue
    fi

    if [[ -n "${running_pid_by_gpu[${gpu}]:-}" ]]; then
      if kill -0 "${running_pid_by_gpu[${gpu}]}" 2>/dev/null; then
        log "gpu=${gpu} has scheduler-launched job pid=${running_pid_by_gpu[${gpu}]}; skip"
        continue
      fi
      log "gpu=${gpu} previous scheduler-launched job pid=${running_pid_by_gpu[${gpu}]} finished; resume monitoring"
      unset "running_pid_by_gpu[${gpu}]"
      stable_counts["${gpu}"]=0
    fi

    if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
      stable_counts["${gpu}"]=$(( ${stable_counts["${gpu}"]:-0} + 1 ))
    else
      stable_counts["${gpu}"]=0
    fi

    log "gpu=${gpu} free_mib=${free_mib} stable=${stable_counts["${gpu}"]}/${REQUIRED_STABLE_CHECKS}"

    if [[ "${stable_counts["${gpu}"]}" -ge "${REQUIRED_STABLE_CHECKS}" ]]; then
      if exp="$(next_experiment)"; then
        launch_experiment "${exp}" "${gpu}"
        running_pid_by_gpu["${gpu}"]="${LAST_LAUNCH_PID}"
        stable_counts["${gpu}"]=0
        launch_count=$((launch_count + 1))
      else
        log "all experiments already launched"
        exit 0
      fi
    fi
  done <<< "${snapshot}"

  if [[ "${launch_count}" -ge "${MAX_LAUNCHES}" ]]; then
    break
  fi

  sleep "${CHECK_INTERVAL}"
done

log "scheduler finished after launching ${launch_count} experiment entries"
