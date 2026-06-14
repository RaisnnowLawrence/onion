#!/usr/bin/env python3
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict


ROOT = "/data2/lizhengxue/WorkSpace/onion_output/aokvqa"
OUT_DIR = "/data2/lizhengxue/WorkSpace/onion_output/ablation_summary"
MD_DOCS = "/data2/lizhengxue/WorkSpace/onion_output/md_docs"

METHODS = {
    "r3_caption": "qwen3-VL-4B_forward2_reflective_answer_first_caption_r3_rounds1_3shards_gpu4",
    "answer_first_locked_no_caption": "qwen3-VL-4B_forward2_answer_first_locked_no_caption_rounds1_3shards_gpu6",
    "review_empty_context": "qwen3-VL-4B_forward2_reflective_review_empty_context_caption_r3_3shards_gpu6",
    "no_cot_rounds1": "qwen3-VL-4B_forward2_no_cot_rounds1",
    "r3_no_caption": "qwen3-VL-4B_forward2_reflective_answer_first_no_caption_r3_rounds1_3shards_gpu5",
    "keep_revise": "qwen3-VL-4B_forward2_reflective_keep_revise_caption_r3_3shards_gpu4",
    "visible_only": "qwen3-VL-4B_forward2_reflective_visible_only_caption_r3_3shards_gpu5",
    "reviewer_caption_only": "qwen3-VL-4B_forward2_reviewer_evidence_caption_only_rounds1_3shards_gpu4",
}


def sample_idx(path):
    match = re.search(r"sample_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1


def extract_question(prompt):
    match = re.search(r"Question:\s*(.+?)(?:\n===|\nAnswer:|\Z)", prompt, flags=re.DOTALL)
    if not match:
        return ""
    question = match.group(1).strip()
    question = re.sub(r"\nChoices:.*", "", question, flags=re.DOTALL).strip()
    return question


def extract_context(prompt):
    match = re.search(r"Brief Context:\s*(.*?)\n===The question", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def question_type(question):
    q = question.lower()
    if "how many" in q or "number" in q or re.search(r"\bcount\b", q):
        return "count"
    if any(x in q for x in ("color", "colour")):
        return "color"
    if any(x in q for x in ("text", "word", "letter", "sign", "read", "say", "written")):
        return "text_ocr"
    if any(x in q for x in ("why", "purpose", "used for", "use for", "what is the purpose")):
        return "purpose_reason"
    if any(x in q for x in ("where", "left", "right", "behind", "front", "next to", "near", "located")):
        return "spatial_location"
    if any(x in q for x in ("doing", "holding", "wearing", "riding", "eating", "playing")):
        return "action_state"
    if any(x in q for x in ("what kind", "type of", "what type")):
        return "category_type"
    if q.startswith("what "):
        return "object_what"
    return "other"


def load_method(method, dirname):
    prompt_dir = os.path.join(ROOT, dirname, "prompt_samples")
    rows = {}
    for path in glob.glob(os.path.join(prompt_dir, "sample_*.json")):
        idx = sample_idx(path)
        with open(path, "r") as f:
            item = json.load(f)
        prompt = item[2] if len(item) > 2 else ""
        state = item[-1] if item and isinstance(item[-1], dict) else {}
        rows[idx] = {
            "idx": idx,
            "key": item[0] if len(item) > 0 else "",
            "answer": item[1] if len(item) > 1 else "",
            "score": float(item[3]) if len(item) > 3 else 0.0,
            "question": extract_question(prompt),
            "context": extract_context(prompt),
            "objects": ",".join(state.get("selected_objects", [])),
            "method": method,
        }
    return rows


def main():
    data = {method: load_method(method, dirname) for method, dirname in METHODS.items()}
    base = data["r3_caption"]
    common = sorted(set(base).intersection(*[set(v) for v in data.values()]))

    rows = []
    strict_recoverable = []
    soft_improvable = []
    oracle_total = 0.0
    base_total = 0.0
    method_wins = Counter()
    type_counter = Counter()
    strict_type_counter = Counter()

    for idx in common:
        base_row = base[idx]
        base_score = base_row["score"]
        base_total += base_score
        candidates = []
        for method, method_rows in data.items():
            row = method_rows[idx]
            candidates.append((row["score"], method, row["answer"]))
        best_score, best_method, best_answer = max(candidates, key=lambda x: x[0])
        oracle_total += best_score
        qtype = question_type(base_row["question"])
        if best_score > base_score:
            type_counter[qtype] += 1
            method_wins[best_method] += 1
            row = {
                "idx": idx,
                "key": base_row["key"],
                "question": base_row["question"],
                "question_type": qtype,
                "context": base_row["context"],
                "objects": base_row["objects"],
                "r3_answer": base_row["answer"],
                "r3_score": base_score,
                "best_method": best_method,
                "best_answer": best_answer,
                "best_score": best_score,
                "delta": best_score - base_score,
            }
            for method in METHODS:
                row[f"{method}_answer"] = data[method][idx]["answer"]
                row[f"{method}_score"] = data[method][idx]["score"]
            soft_improvable.append(row)
            if base_score == 0.0 and best_score > 0.0:
                strict_type_counter[qtype] += 1
                strict_recoverable.append(row)
        rows.append({
            "idx": idx,
            "r3_score": base_score,
            "oracle_score": best_score,
            "oracle_method": best_method,
        })

    csv_path = os.path.join(OUT_DIR, "reflective_oracle_r3_error_cases.csv")
    fieldnames = [
        "idx", "key", "question", "question_type", "context", "objects",
        "r3_answer", "r3_score", "best_method", "best_answer", "best_score", "delta",
    ]
    for method in METHODS:
        fieldnames.extend([f"{method}_answer", f"{method}_score"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(soft_improvable)

    strict_csv_path = os.path.join(OUT_DIR, "reflective_oracle_r3_strict_zero_cases.csv")
    with open(strict_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strict_recoverable)

    oracle_acc = oracle_total * 100.0 / len(common)
    base_acc = base_total * 100.0 / len(common)
    gain = oracle_acc - base_acc

    top_examples = sorted(soft_improvable, key=lambda r: (-r["delta"], r["idx"]))[:20]

    md_path = os.path.join(MD_DOCS, "07_reflective_oracle_analysis.md")
    with open(md_path, "w") as f:
        f.write("# Reflective R3 逐题 Oracle 分析\n\n")
        f.write("生成时间：2026-06-14\n\n")
        f.write("目标：找出 `reflective_answer_first_caption_r3` 错、但其他强方法能做对或得分更高的题。\n\n")
        f.write("## 参与对比的方法\n\n")
        f.write("| 方法名 | 实验目录 |\n|---|---|\n")
        for method, dirname in METHODS.items():
            f.write(f"| `{method}` | `{dirname}` |\n")
        f.write("\n## 总体 Oracle 上界\n\n")
        f.write(f"- 对齐样本数：`{len(common)}`\n")
        f.write(f"- r3 caption soft accuracy：`{base_acc:.2f}%`\n")
        f.write(f"- oracle 每题选最高分方法：`{oracle_acc:.2f}%`\n")
        f.write(f"- 理论可提升：`+{gain:.2f}` 个百分点，约 `{oracle_total - base_total:.1f}` 个 soft-correct 样本\n")
        f.write(f"- soft 可改进题数：`{len(soft_improvable)}`\n")
        f.write(f"- 严格 r3=0 且其他方法>0 的题数：`{len(strict_recoverable)}`\n\n")

        f.write("## 哪些方法补到了 r3 的错题\n\n")
        f.write("| 方法 | 成为 best 的题数 |\n|---|---:|\n")
        for method, count in method_wins.most_common():
            f.write(f"| `{method}` | {count} |\n")
        f.write("\n## soft 可改进题型分布\n\n")
        f.write("| 题型 | 题数 |\n|---|---:|\n")
        for qtype, count in type_counter.most_common():
            f.write(f"| `{qtype}` | {count} |\n")
        f.write("\n## 严格可挽救错题题型分布\n\n")
        f.write("| 题型 | 题数 |\n|---|---:|\n")
        for qtype, count in strict_type_counter.most_common():
            f.write(f"| `{qtype}` | {count} |\n")

        f.write("\n## Top 可改进样例\n\n")
        f.write("| idx | type | question | r3 | r3 score | best method | best answer | best score |\n")
        f.write("|---:|---|---|---|---:|---|---|---:|\n")
        for row in top_examples:
            question = row["question"].replace("|", "/")
            f.write(
                f"| {row['idx']} | `{row['question_type']}` | {question} | "
                f"{row['r3_answer']} | {row['r3_score']:.1f} | `{row['best_method']}` | "
                f"{row['best_answer']} | {row['best_score']:.1f} |\n"
            )

        f.write("\n## 初步结论\n\n")
        f.write("1. 如果能学会在少量题上从 r3 caption 切换到其他强方法，理论上足够突破 60%。\n")
        f.write("2. 最有价值的不是新增长 CoT，而是做一个轻量 routing：什么题保持 r3，什么题切换到 no-caption/empty-review/locked。\n")
        f.write("3. 下一步建议人工抽查 `reflective_oracle_r3_error_cases.csv` 中 delta 最大的题，归纳可写成规则的失败模式。\n")
        f.write("4. 严格 r3=0 的可挽救错题更适合做规则，因为它们不是 soft-score 微小差异。\n\n")
        f.write("## 输出文件\n\n")
        f.write(f"- soft 可改进题：`{csv_path}`\n")
        f.write(f"- 严格 r3=0 可挽救题：`{strict_csv_path}`\n")

    print(md_path)
    print(csv_path)
    print(strict_csv_path)
    print(f"common={len(common)} base={base_acc:.2f} oracle={oracle_acc:.2f} gain={gain:.2f}")


if __name__ == "__main__":
    main()
