#!/usr/bin/env bash
set -euo pipefail

BASE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
PYTHON_BIN=${PYTHON_BIN:-/data2/lizhengxue/anaconda3/envs/sam/bin/python}
ENGINE=qwen3-VL-4B
OUT_ROOT=/data2/lizhengxue/WorkSpace/onion_output/aokvqa
REPORT_DIR=/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
LOG_DIR="${REPORT_DIR}/logs_followup2"
GPU_LIST=${GPU_LIST:-0,1,2}
GPU_SLOTS=${GPU_SLOTS:-2}
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
export PYTHONUNBUFFERED=1

CACHE_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/image_cache_onion/cache_forward4b
SG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
TRAIN_SIM_FILE=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk
TAG_PATH=/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
META_FILE="${REPORT_DIR}/followup2_experiments.tsv"
MASTER_LOG="${REPORT_DIR}/followup2_master.log"

mkdir -p "${LOG_DIR}" "${CACHE_PATH}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

output_path_for() {
    echo "${OUT_ROOT}/${ENGINE}_forward2_$1"
}

sample_count() {
    local exp_name="$1"
    find "$(output_path_for "${exp_name}")/prompt_samples" -name 'sample_*.json' 2>/dev/null | wc -l
}

accuracy_log() {
    local exp_name="$1"
    echo "$(output_path_for "${exp_name}")/accuracy.log"
}

running_pids_for() {
    local exp_name="$1"
    local output_path
    output_path="$(output_path_for "${exp_name}")"
    pgrep -f "forward_code/onion.py.*${output_path}" || true
}

write_meta() {
    cat > "${META_FILE}" <<'META'
id	name	description	params
C1	clean_rounds1	clean rerun of rounds=1 CoT	--rounds 1 --chain_of_thoughts
NC1	no_cot_rounds1	no-CoT with one round	--rounds 1
NC3	no_cot_rounds3	no-CoT with three rounds	--rounds 3
NE1	no_cot_ensemble1	no-CoT with one ensemble sample	--n_ensemble 1
NE3	no_cot_ensemble3	no-CoT with three ensemble samples	--n_ensemble 3
CTX0	context_empty	empty brief context	--context_mode empty --chain_of_thoughts
CTXC	context_caption_only	only original caption in brief context	--context_mode caption_only --chain_of_thoughts
CTXO	context_objects_only	only selected object names in brief context	--context_mode objects_only --chain_of_thoughts
CTXN	context_no_round_state	no previous round/thought state in brief context	--context_mode no_round_state --chain_of_thoughts
QG	qwen_caption_global	Qwen global caption only	--use_qwen_blip2_caption --qwen_caption_mode global --chain_of_thoughts
QL	qwen_caption_local	Qwen local caption only	--use_qwen_blip2_caption --qwen_caption_mode local --chain_of_thoughts
QS	qwen_caption_short	short Qwen global+local captions	--use_qwen_blip2_caption --qwen_caption_max_tokens 48 --qwen_caption_final_max_chars 220 --chain_of_thoughts
QNF	qwen_caption_no_final	query Qwen captions but do not inject into final prompt	--use_qwen_blip2_caption --qwen_caption_no_final_context --chain_of_thoughts
AS	answer_strict_final	strict Final Answer/Answer extraction	--answer_extraction_strategy strict_final --chain_of_thoughts
AL	answer_last_line	last non-empty line extraction	--answer_extraction_strategy last_line --chain_of_thoughts
AR	answer_raw	raw CoT response for voting/eval	--answer_extraction_strategy raw --chain_of_thoughts
META
}

run_report() {
    "${PYTHON_BIN}" - "${META_FILE}" "${REPORT_DIR}/followup2_results.csv" "${REPORT_DIR}/followup2_report.md" <<'PY'
import csv
import sys
from pathlib import Path

meta_file, csv_out, md_out = map(Path, sys.argv[1:])
out_root = Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa")
engine = "qwen3-VL-4B"
rows = []
with meta_file.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        out_dir = out_root / f"{engine}_forward2_{row['name']}"
        sample_dir = out_dir / "prompt_samples"
        samples = len(list(sample_dir.glob("sample_*.json"))) if sample_dir.exists() else 0
        acc_log = out_dir / "accuracy.log"
        accuracy = ""
        score = ""
        accuracy_line = ""
        status = "not_started"
        if samples:
            status = "running_or_partial"
        if acc_log.exists():
            status = "merged"
            for line in acc_log.read_text(errors="ignore").splitlines():
                if "准确率" in line:
                    accuracy_line = line.strip()
                    try:
                        part = line.split("准确率:", 1)[1].strip()
                        accuracy = part.split("%", 1)[0].strip()
                        score = part.split("(", 1)[1].split(")", 1)[0]
                    except Exception:
                        pass
                    break
        rows.append({
            "id": row["id"],
            "name": row["name"],
            "status": status,
            "samples": samples,
            "accuracy": accuracy,
            "score": score,
            "description": row["description"],
            "params": row["params"],
            "output_dir": str(out_dir),
            "accuracy_line": accuracy_line,
        })

with csv_out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = ["# VisualCoT Follow-up 2 Ablation Report", "", "| ID | Experiment | Status | Samples | Accuracy | Score | Params |", "|---|---|---:|---:|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r['id']} | {r['name']} | {r['status']} | {r['samples']} | {r['accuracy']} | {r['score']} | `{r['params']}` |")
md_out.write_text("\n".join(lines) + "\n")
print(csv_out)
print(md_out)
PY
}

launch_shard() {
    local exp_name="$1"
    local gpu="$2"
    local shard="$3"
    local cot="$4"
    local extra_args="$5"
    local output_path
    output_path="$(output_path_for "${exp_name}")"
    local log_path="${LOG_DIR}/${exp_name}_gpu${gpu}_shard${shard}.log"

    local cot_arg=()
    if [[ "${cot}" == "cot" ]]; then
        cot_arg=(--chain_of_thoughts)
    fi

    # shellcheck disable=SC2086
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --cache_path "${CACHE_PATH}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --n_shot 1 \
        --n_ensemble 5 \
        --rounds 5 \
        --iterative_strategy caption \
        --engine "${ENGINE}" \
        --sg_path "${SG_PATH}" \
        --train_sim_metric answer \
        --train_sim_file "${TRAIN_SIM_FILE}" \
        --tag_path "${TAG_PATH}" \
        --with_clip_verify \
        "${cot_arg[@]}" \
        ${extra_args} \
        --shard_id "${shard}" --num_shards 2 \
        > "${log_path}" 2>&1 < /dev/null &
    RUN_PID="$!"
}

merge_exp() {
    local exp_name="$1"
    local cot="$2"
    local extra_args="$3"
    local output_path
    output_path="$(output_path_for "${exp_name}")"
    local summary_log
    summary_log="$(accuracy_log "${exp_name}")"

    if [[ "$(sample_count "${exp_name}")" -lt 1145 ]]; then
        echo "[$(timestamp)] ${exp_name}: skip merge, only $(sample_count "${exp_name}") samples" | tee -a "${MASTER_LOG}"
        return 1
    fi
    if [[ -f "${summary_log}" ]]; then
        echo "[$(timestamp)] ${exp_name}: already merged" | tee -a "${MASTER_LOG}"
        return 0
    fi

    local cot_arg=()
    if [[ "${cot}" == "cot" ]]; then
        cot_arg=(--chain_of_thoughts)
    fi

    echo "[$(timestamp)] ${exp_name}: merge" | tee -a "${MASTER_LOG}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${BASE}/forward_code/onion.py" \
        --engine "${ENGINE}" \
        --output_path "${output_path}" \
        --caption_type vinvl \
        --sg_path "${SG_PATH}" \
        --similarity_metric imagequestion \
        --merge_only \
        "${cot_arg[@]}" \
        ${extra_args} \
        --summary_log "${summary_log}" \
        >> "${LOG_DIR}/${exp_name}_merge.log" 2>&1
}

run_exp_same_gpu() {
    local exp_name="$1"
    local gpu="$2"
    local cot="$3"
    local extra_args="$4"

    if [[ -f "$(accuracy_log "${exp_name}")" ]]; then
        echo "[$(timestamp)] ${exp_name}: already done" | tee -a "${MASTER_LOG}"
        return 0
    fi

    local running_pids
    running_pids="$(running_pids_for "${exp_name}")"
    if [[ -n "${running_pids}" ]]; then
        echo "[$(timestamp)] ${exp_name}: already running, pids ${running_pids//$'\n'/, }" | tee -a "${MASTER_LOG}"
        while [[ -n "$(running_pids_for "${exp_name}")" ]]; do
            sleep 300
        done
        merge_exp "${exp_name}" "${cot}" "${extra_args}" || true
        run_report >> "${MASTER_LOG}" 2>&1
        echo "[$(timestamp)] done ${exp_name}: $(sample_count "${exp_name}") samples" | tee -a "${MASTER_LOG}"
        return 0
    fi

    echo "[$(timestamp)] launch ${exp_name} on GPU ${gpu}: ${extra_args}" | tee -a "${MASTER_LOG}"
    local pid0 pid1
    launch_shard "${exp_name}" "${gpu}" 0 "${cot}" "${extra_args}"
    pid0="${RUN_PID}"
    launch_shard "${exp_name}" "${gpu}" 1 "${cot}" "${extra_args}"
    pid1="${RUN_PID}"
    echo "[$(timestamp)] ${exp_name}: pids ${pid0}, ${pid1}" | tee -a "${MASTER_LOG}"
    wait "${pid0}" || true
    wait "${pid1}" || true
    merge_exp "${exp_name}" "${cot}" "${extra_args}" || true
    run_report >> "${MASTER_LOG}" 2>&1
    echo "[$(timestamp)] done ${exp_name}: $(sample_count "${exp_name}") samples" | tee -a "${MASTER_LOG}"
}

main() {
    write_meta
    run_report >> "${MASTER_LOG}" 2>&1
    echo "[$(timestamp)] followup2 controller started; GPUs=${GPU_LIST}; GPU_SLOTS=${GPU_SLOTS}" | tee -a "${MASTER_LOG}"

    local experiments=(
        "clean_rounds1|cot|--rounds 1"
        "no_cot_rounds1|no_cot|--rounds 1"
        "no_cot_rounds3|no_cot|--rounds 3"
        "no_cot_ensemble1|no_cot|--n_ensemble 1"
        "no_cot_ensemble3|no_cot|--n_ensemble 3"
        "context_empty|cot|--context_mode empty"
        "context_caption_only|cot|--context_mode caption_only"
        "context_objects_only|cot|--context_mode objects_only"
        "context_no_round_state|cot|--context_mode no_round_state"
        "qwen_caption_global|cot|--use_qwen_blip2_caption --qwen_caption_mode global"
        "qwen_caption_local|cot|--use_qwen_blip2_caption --qwen_caption_mode local"
        "qwen_caption_short|cot|--use_qwen_blip2_caption --qwen_caption_max_tokens 48 --qwen_caption_final_max_chars 220"
        "qwen_caption_no_final|cot|--use_qwen_blip2_caption --qwen_caption_no_final_context"
        "answer_strict_final|cot|--answer_extraction_strategy strict_final"
        "answer_last_line|cot|--answer_extraction_strategy last_line"
        "answer_raw|cot|--answer_extraction_strategy raw"
    )

    local slot_gpus=()
    local slot
    for ((slot = 0; slot < GPU_SLOTS; slot++)); do
        local gpu_for_slot
        for gpu_for_slot in "${GPUS[@]}"; do
            slot_gpus+=("${gpu_for_slot}")
        done
    done

    local active=0
    local idx=0
    for item in "${experiments[@]}"; do
        IFS='|' read -r exp_name cot extra_args <<< "${item}"
        local gpu="${slot_gpus[$((idx % ${#slot_gpus[@]}))]}"
        run_exp_same_gpu "${exp_name}" "${gpu}" "${cot}" "${extra_args}" &
        active=$((active + 1))
        idx=$((idx + 1))
        if [[ "${active}" -ge "${#slot_gpus[@]}" ]]; then
            wait -n
            active=$((active - 1))
        fi
    done
    wait
    run_report >> "${MASTER_LOG}" 2>&1
    echo "[$(timestamp)] followup2 controller finished" | tee -a "${MASTER_LOG}"
}

main "$@"
