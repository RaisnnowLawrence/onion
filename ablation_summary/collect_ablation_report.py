#!/usr/bin/env python3
import csv
import glob
import os
import re
from datetime import datetime


OUT_ROOT = "/data2/lizhengxue/WorkSpace/onion_output/aokvqa"
REPORT_DIR = "/data2/lizhengxue/WorkSpace/onion_output/ablation_summary"
ENGINE = "qwen3-VL-4B"


EXPERIMENTS = [
    {
        "id": "0",
        "name": "baseline",
        "group": "current",
        "description": "baseline: CoT, rounds=5, n_ensemble=5, n_shot=1, caption_type=vinvl",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A1",
        "name": "remove_caption",
        "group": "followup",
        "description": "remove brief caption from final answer context",
        "params": "--remove_caption --caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A2",
        "name": "no_cot",
        "group": "current",
        "description": "disable chain-of-thought prompting",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5",
    },
    {
        "id": "A3.1",
        "name": "rounds1",
        "group": "followup",
        "description": "single interactive round",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 1 --chain_of_thoughts",
    },
    {
        "id": "A3.2",
        "name": "rounds3",
        "group": "followup",
        "description": "three interactive rounds",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 3 --chain_of_thoughts",
    },
    {
        "id": "A4.1",
        "name": "ensemble1",
        "group": "followup",
        "description": "one self-consistency sample",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 1 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A4.2",
        "name": "ensemble3",
        "group": "followup",
        "description": "three self-consistency samples",
        "params": "--caption_type vinvl --n_shot 1 --n_ensemble 3 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A5.1",
        "name": "nshot0",
        "group": "followup",
        "description": "zero-shot prompt",
        "params": "--caption_type vinvl --n_shot 0 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A5.2",
        "name": "nshot4",
        "group": "followup",
        "description": "four-shot prompt",
        "params": "--caption_type vinvl --n_shot 4 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A6",
        "name": "sim_question",
        "group": "followup",
        "description": "question-only retrieval instead of image+question retrieval",
        "params": "--similarity_metric question --caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A7.1",
        "name": "caption_vinvl_tag",
        "group": "followup",
        "description": "VinVL caption plus predicted tags",
        "params": "--caption_type vinvl_tag --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "A7.2",
        "name": "caption_vinvl_sg",
        "group": "followup",
        "description": "VinVL scene-graph caption setting",
        "params": "--caption_type vinvl_sg --n_shot 1 --n_ensemble 5 --rounds 5 --chain_of_thoughts",
    },
    {
        "id": "B1",
        "name": "ocr",
        "group": "current-added-modules",
        "description": "OCR context switch",
        "params": "--use_ocr_context",
    },
    {
        "id": "B2",
        "name": "clip_thought",
        "group": "current-added-modules",
        "description": "CLIP thought verification",
        "params": "--use_clip_thought_verify",
    },
    {
        "id": "B3",
        "name": "qwen_caption",
        "group": "current-added-modules",
        "description": "Qwen-VL global/local caption helper",
        "params": "--use_qwen_blip2_caption",
    },
    {
        "id": "B4",
        "name": "qwen_thought",
        "group": "current-added-modules",
        "description": "Qwen-VL thought verifier",
        "params": "--use_qwen_blip2_thought_verify",
    },
    {
        "id": "B5",
        "name": "all_regions",
        "group": "current-added-modules",
        "description": "inject all regional captions",
        "params": "--use_all_regional_captions",
    },
    {
        "id": "B6",
        "name": "ensemble_norm",
        "group": "current-added-modules",
        "description": "normalized majority voting",
        "params": "--ensemble_strategy normalized_majority",
    },
    {
        "id": "B7",
        "name": "all_added",
        "group": "current-added-modules",
        "description": "all added modules except image enhancement",
        "params": "--use_ocr_context --use_clip_thought_verify --use_qwen_blip2_caption --use_qwen_blip2_thought_verify --use_all_regional_captions --ensemble_strategy normalized_majority",
    },
]


def output_dir(name):
    return os.path.join(OUT_ROOT, f"{ENGINE}_forward_{name}")


def sample_count(name):
    return len(glob.glob(os.path.join(output_dir(name), "prompt_samples", "sample_*.json")))


def accuracy_line(name):
    path = os.path.join(output_dir(name), "accuracy.log")
    if not os.path.isfile(path):
        return "", "", ""
    lines = [line.strip() for line in open(path) if line.strip()]
    if not lines:
        return "", "", ""
    line = lines[-1]
    match = re.search(r"([0-9.]+)%\s+\((\d+)/(\d+)\)", line)
    if not match:
        return line, "", ""
    return line, match.group(1), f"{match.group(2)}/{match.group(3)}"


def status(name):
    count = sample_count(name)
    if count >= 1145 and os.path.isfile(os.path.join(output_dir(name), "accuracy.log")):
        return "merged"
    if count >= 1145:
        return "needs_merge"
    if count > 0:
        return "running_or_partial"
    return "not_started"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = []
    for exp in EXPERIMENTS:
        line, acc, score = accuracy_line(exp["name"])
        rows.append({
            **exp,
            "samples": sample_count(exp["name"]),
            "status": status(exp["name"]),
            "accuracy": acc,
            "score": score,
            "accuracy_line": line,
            "output_dir": output_dir(exp["name"]),
        })

    csv_path = os.path.join(REPORT_DIR, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "id", "name", "group", "status", "samples", "accuracy", "score",
            "description", "params", "output_dir", "accuracy_line",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(REPORT_DIR, "ablation_report.md")
    with open(md_path, "w") as f:
        f.write("# VisualCoT A-OKVQA Ablation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| ID | Experiment | Status | Samples | Accuracy | Score | Params | Output |\n")
        f.write("|---|---|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['id']} | {row['name']} | {row['status']} | {row['samples']} | "
                f"{row['accuracy']} | {row['score']} | `{row['params']}` | `{row['output_dir']}` |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- A1-A7 are the no-code ablations discussed after the added-module ablation run.\n")
        f.write("- A8 image enhancement is intentionally excluded from the follow-up run.\n")
        f.write("- B-series entries are the added-module ablations already run in this session.\n")

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
