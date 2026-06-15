#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict


DEFAULT_OUT_ROOT = "/data2/lizhengxue/WorkSpace/onion_output/aokvqa"
DEFAULT_REPORT_DIR = "/data2/lizhengxue/WorkSpace/onion/ablation_summary"

METHODS = [
    (
        "no_cot",
        "qwen3-VL-4B_forward2_no_cot_rounds1",
    ),
    (
        "reflective_r3",
        "qwen3-VL-4B_forward2_reflective_answer_first_caption_r3_rounds1_3shards_gpu4",
    ),
    (
        "answer_first_no_caption",
        "qwen3-VL-4B_forward2_answer_first_locked_no_caption_rounds1_3shards_gpu6",
    ),
    (
        "reflective_empty_review",
        "qwen3-VL-4B_forward2_reflective_review_empty_context_caption_r3_3shards_gpu6",
    ),
    (
        "candidate_marker_mcts",
        "qwen3-VL-4B_forward2_candidate_judge_marker_mcts_3shards_gpu3",
    ),
    (
        "rag_protected_n400",
        "qwen3-VL-4B_forward2_strategy_rag_protected_reflective_train400_val_3shards",
    ),
]


def classify_question(question):
    text = str(question).lower()
    text_cues = ("text", "word", "letter", "sign", "read", "says", "written", "logo", "number")
    visual_detail_cues = (
        "how many", "count", "what color", "which color", "color", "where", "which side",
        "left", "right", "behind", "front", "next to", "wearing", "holding", "doing",
        "mouth", "hand", "what is in", "what are in", "what is on"
    )
    knowledge_cues = (
        "why", "used for", "use for", "purpose", "probably", "most likely", "event",
        "sport", "game", "season", "weather", "celebrated", "celebrating"
    )
    category_cues = (
        "what kind", "what type", "which animal", "what animal", "what food", "what object",
        "what item", "what device", "what appliance", "made of"
    )
    if any(cue in text for cue in text_cues):
        return "text_ocr"
    if any(cue in text for cue in visual_detail_cues):
        return "visual_detail"
    if any(cue in text for cue in knowledge_cues):
        return "knowledge"
    if any(cue in text for cue in category_cues):
        return "category"
    return "general"


def extract_question(prompt):
    match = re.search(r"Question:\s*(.+?)(?:\n===|\nAnswer:|\Z)", str(prompt), flags=re.S)
    if not match:
        return ""
    line = match.group(1).strip().splitlines()[0].strip()
    return line


def extract_route(entry):
    text_parts = []
    if len(entry) > 4 and isinstance(entry[4], list):
        text_parts.extend(str(x) for x in entry[4])
    if len(entry) > 5 and isinstance(entry[5], list):
        text_parts.extend(str(x) for x in entry[5])
    text = "\n".join(text_parts)
    route = ""
    reason = ""
    stats = {}
    match = re.search(r"RAG Strategy Router:\s*([^\n]+)", text)
    if match:
        route = match.group(1).strip()
    match = re.search(
        r"Route Stats:\s*direct_avg=([0-9.]+)\s+cot_avg=([0-9.]+)\s+"
        r"rescue_rate=([0-9.]+)\s+damage_rate=([0-9.]+)\s+reason=([^\n]+)",
        text,
    )
    if match:
        stats = {
            "direct_avg": float(match.group(1)),
            "cot_avg": float(match.group(2)),
            "rescue_rate": float(match.group(3)),
            "damage_rate": float(match.group(4)),
        }
        reason = match.group(5).strip()
    return route, reason, stats


def load_method(out_root, dirname):
    prompt_dir = os.path.join(out_root, dirname, "prompt_samples")
    rows = {}
    for path in glob.glob(os.path.join(prompt_dir, "sample_*.json")):
        try:
            entry = json.load(open(path))
        except Exception:
            continue
        if not isinstance(entry, list) or len(entry) < 4:
            continue
        key = entry[0]
        question = extract_question(entry[2] if len(entry) > 2 else "")
        route, reason, route_stats = extract_route(entry)
        rows[key] = {
            "key": key,
            "answer": entry[1],
            "score": float(entry[3]),
            "question": question,
            "question_type": classify_question(question),
            "route": route,
            "route_reason": reason,
            "route_stats": route_stats,
            "path": path,
        }
    return rows


def pct(score, total):
    return 100.0 * score / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--base", default="no_cot")
    args = parser.parse_args()

    methods = {}
    for name, dirname in METHODS:
        rows = load_method(args.out_root, dirname)
        methods[name] = rows
        print(f"{name}: loaded {len(rows)} samples")

    common_keys = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if not common_keys:
        raise SystemExit("No common samples found")

    base_rows = methods[args.base]
    summary_rows = []
    for name, rows in methods.items():
        score_sum = sum(rows[k]["score"] for k in common_keys)
        base_score_sum = sum(base_rows[k]["score"] for k in common_keys)
        rescued = sum(1 for k in common_keys if base_rows[k]["score"] <= 0 and rows[k]["score"] > 0)
        damaged = sum(1 for k in common_keys if base_rows[k]["score"] > 0 and rows[k]["score"] <= 0)
        improved = sum(1 for k in common_keys if rows[k]["score"] > base_rows[k]["score"])
        worsened = sum(1 for k in common_keys if rows[k]["score"] < base_rows[k]["score"])
        delta_score = score_sum - base_score_sum
        summary_rows.append({
            "method": name,
            "samples": len(common_keys),
            "score_sum": score_sum,
            "accuracy": pct(score_sum, len(common_keys)),
            "delta_score_vs_base": delta_score,
            "delta_acc_vs_base": pct(delta_score, len(common_keys)),
            "rescued_base_zero": rescued,
            "damaged_base_positive": damaged,
            "improved_score": improved,
            "worsened_score": worsened,
        })

    oracle_score = sum(max(methods[name][k]["score"] for name in methods) for k in common_keys)
    all_wrong = sum(1 for k in common_keys if max(methods[name][k]["score"] for name in methods) <= 0)
    oracle_by_set = []
    ordered_names = [name for name, _ in METHODS]
    for upto in range(1, len(ordered_names) + 1):
        names = ordered_names[:upto]
        score = sum(max(methods[name][k]["score"] for name in names) for k in common_keys)
        oracle_by_set.append((("+".join(names)), score, pct(score, len(common_keys))))

    qtype_rows = []
    for name, rows in methods.items():
        by_type = defaultdict(list)
        for k in common_keys:
            by_type[rows[k]["question_type"]].append(rows[k]["score"])
        for qtype, scores in sorted(by_type.items()):
            qtype_rows.append({
                "method": name,
                "question_type": qtype,
                "samples": len(scores),
                "accuracy": pct(sum(scores), len(scores)),
                "score_sum": sum(scores),
            })

    route_rows = []
    rag_name = "rag_protected_n400"
    if rag_name in methods:
        route_counter = Counter()
        reason_counter = Counter()
        route_score = defaultdict(float)
        route_count = Counter()
        for k in common_keys:
            rec = methods[rag_name][k]
            route = rec["route"] or "unknown"
            reason = rec["route_reason"] or "unknown"
            route_counter[route] += 1
            reason_counter[reason] += 1
            route_score[route] += rec["score"]
            route_count[route] += 1
        for route, count in sorted(route_counter.items()):
            route_rows.append({
                "route": route,
                "samples": count,
                "accuracy": pct(route_score[route], count),
            })

    pair_rows = []
    for name in methods:
        if name == args.base:
            continue
        both = sum(1 for k in common_keys if base_rows[k]["score"] > 0 and methods[name][k]["score"] > 0)
        base_only = sum(1 for k in common_keys if base_rows[k]["score"] > 0 and methods[name][k]["score"] <= 0)
        method_only = sum(1 for k in common_keys if base_rows[k]["score"] <= 0 and methods[name][k]["score"] > 0)
        neither = sum(1 for k in common_keys if base_rows[k]["score"] <= 0 and methods[name][k]["score"] <= 0)
        pair_rows.append({
            "method": name,
            "both_positive": both,
            "base_only_positive": base_only,
            "method_only_positive": method_only,
            "neither_positive": neither,
        })

    os.makedirs(args.report_dir, exist_ok=True)
    csv_path = os.path.join(args.report_dir, "strategy_overlap_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    qtype_csv = os.path.join(args.report_dir, "strategy_overlap_by_qtype.csv")
    with open(qtype_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qtype_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qtype_rows)

    md_path = os.path.join(args.report_dir, "strategy_overlap_report.md")
    with open(md_path, "w") as f:
        f.write("# Strategy Overlap Diagnostic\n\n")
        f.write(f"Common samples: `{len(common_keys)}`\n\n")
        f.write("## Method Summary\n\n")
        f.write("| Method | Acc | Score | Delta vs no_cot | Rescued no_cot=0 | Damaged no_cot>0 | Improved | Worsened |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| `{row['method']}` | {row['accuracy']:.2f}% | {row['score_sum']:.1f}/{row['samples']} | "
                f"{row['delta_acc_vs_base']:+.2f}% | {row['rescued_base_zero']} | "
                f"{row['damaged_base_positive']} | {row['improved_score']} | {row['worsened_score']} |\n"
            )

        f.write("\n## Oracle Upper Bound\n\n")
        f.write(f"All-method oracle: `{pct(oracle_score, len(common_keys)):.2f}%` ({oracle_score:.1f}/{len(common_keys)})\n\n")
        f.write(f"All methods wrong/zero score: `{all_wrong}`\n\n")
        f.write("| Method Set | Oracle Acc | Score |\n")
        f.write("|---|---:|---:|\n")
        for names, score, acc in oracle_by_set:
            f.write(f"| `{names}` | {acc:.2f}% | {score:.1f}/{len(common_keys)} |\n")

        f.write("\n## Pairwise vs no_cot\n\n")
        f.write("| Method | Both >0 | no_cot only >0 | Method only >0 | Neither >0 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in pair_rows:
            f.write(
                f"| `{row['method']}` | {row['both_positive']} | {row['base_only_positive']} | "
                f"{row['method_only_positive']} | {row['neither_positive']} |\n"
            )

        if route_rows:
            f.write("\n## RAG Route Distribution\n\n")
            f.write("| Route | Samples | Acc |\n")
            f.write("|---|---:|---:|\n")
            for row in route_rows:
                f.write(f"| `{row['route']}` | {row['samples']} | {row['accuracy']:.2f}% |\n")

        f.write("\n## Question-Type Accuracy\n\n")
        f.write("| Method | Type | Samples | Acc |\n")
        f.write("|---|---|---:|---:|\n")
        for row in qtype_rows:
            f.write(f"| `{row['method']}` | `{row['question_type']}` | {row['samples']} | {row['accuracy']:.2f}% |\n")

    print(md_path)
    print(csv_path)
    print(qtype_csv)


if __name__ == "__main__":
    main()
