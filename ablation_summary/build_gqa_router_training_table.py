#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a paired pure-vs-DyFo router training table.")
    parser.add_argument("--pure-dir", required=True, help="Pure experiment output directory.")
    parser.add_argument("--dyfo-dir", required=True, help="DyFo experiment output directory.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--tie-eps", type=float, default=1e-9)
    return parser.parse_args()


def sample_id(path):
    match = re.search(r"sample_(\d+)_", path.name)
    if not match:
        raise ValueError(f"cannot parse sample id from {path}")
    return int(match.group(1))


def extract_question(prompt):
    match = re.search(r"Question:\s*(.*?)\n===", prompt, flags=re.S)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"Question:\s*(.*)", prompt, flags=re.S)
    return " ".join(match.group(1).split()) if match else ""


def question_type(question):
    q = question.lower()
    if re.search(r"\b(left|right|front of|behind|side|between|above|below|near|next to|on top of|under)\b", q):
        return "spatial_relation"
    if re.search(r"\b(is|are|was|were|do|does|did|has|have|can|could)\b", q):
        return "yes_no"
    if "color" in q or "colour" in q:
        return "color"
    if re.search(r"\bhow many\b", q):
        return "count"
    if re.search(r"\bwho\b", q):
        return "who"
    if re.search(r"\bwhere\b", q):
        return "where"
    if re.search(r"\bwhich\b", q):
        return "which"
    if re.search(r"\bwhat\b", q):
        return "what"
    return "other"


def load_prompt_samples(exp_dir):
    root = Path(exp_dir) / "prompt_samples"
    rows = {}
    for path in root.glob("sample_*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sid = sample_id(path)
        state = data[6] if len(data) > 6 and isinstance(data[6], dict) else {}
        key = str(data[0])
        image_id = key.split("<->", 1)[0]
        rows[key] = {
            "sample_id": sid,
            "key": key,
            "image_id": image_id,
            "question_id": key.split("<->", 1)[1] if "<->" in key else "",
            "question": extract_question(data[2]),
            "answer": data[1],
            "score": float(data[3]),
            "pred_answer": state.get("pred_answer", data[1]),
            "pred_candidates": state.get("pred_candidates", []),
            "instruction": state.get("instruction", ""),
            "executed_evidence": state.get("executed_evidence", []),
            "dyfo_visual_evidence": state.get("dyfo_visual_evidence", ""),
            "dyfo_decision_trace": state.get("dyfo_decision_trace"),
        }
    return rows


def label(pure_score, dyfo_score, eps):
    delta = dyfo_score - pure_score
    if delta > eps:
        return "dyfo_better"
    if delta < -eps:
        return "llm_better"
    if pure_score > eps:
        return "same_good"
    return "same_bad"


def main():
    args = parse_args()
    pure = load_prompt_samples(args.pure_dir)
    dyfo = load_prompt_samples(args.dyfo_dir)
    keys = sorted(set(pure) & set(dyfo), key=lambda k: pure[k]["sample_id"])
    if not keys:
        raise ValueError("no overlapping samples between pure and dyfo outputs")

    rows = []
    for key in keys:
        p = pure[key]
        d = dyfo[key]
        qtype = question_type(p["question"])
        rows.append({
            "sample_id": p["sample_id"],
            "key": key,
            "image_id": p["image_id"],
            "question_id": p["question_id"],
            "question": p["question"],
            "question_type": qtype,
            "pure_answer": p["pred_answer"],
            "pure_score": p["score"],
            "dyfo_answer": d["pred_answer"],
            "dyfo_score": d["score"],
            "score_delta_dyfo_minus_pure": d["score"] - p["score"],
            "router_label": label(p["score"], d["score"], args.tie_eps),
            "recommended_route": "dyfo" if d["score"] - p["score"] > args.tie_eps else "pure",
            "dyfo_executed_evidence": ",".join(map(str, d["executed_evidence"])),
            "dyfo_visual_evidence": d["dyfo_visual_evidence"],
            "dyfo_decision_trace": json.dumps(d["dyfo_decision_trace"], ensure_ascii=False),
        })

    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["router_label"] for row in rows)
    qtype_counts = Counter((row["question_type"], row["router_label"]) for row in rows)
    pure_total = sum(row["pure_score"] for row in rows)
    dyfo_total = sum(row["dyfo_score"] for row in rows)

    qtypes = sorted({row["question_type"] for row in rows})
    qtype_lines = []
    for qt in qtypes:
        total = sum(v for (name, _), v in qtype_counts.items() if name == qt)
        qtype_lines.append(
            f"| {qt} | {total} | {qtype_counts[(qt, 'llm_better')]} | "
            f"{qtype_counts[(qt, 'dyfo_better')]} | {qtype_counts[(qt, 'same_good')]} | "
            f"{qtype_counts[(qt, 'same_bad')]} |"
        )

    report = f"""# GQA Train 12k Router Training Table

| Item | Value |
| --- | ---: |
| paired samples | {len(rows)} |
| pure score sum | {pure_total:.2f} |
| dyfo score sum | {dyfo_total:.2f} |
| dyfo - pure | {dyfo_total - pure_total:+.2f} |

## Labels

| Label | Count |
| --- | ---: |
| llm_better | {counts['llm_better']} |
| dyfo_better | {counts['dyfo_better']} |
| same_good | {counts['same_good']} |
| same_bad | {counts['same_bad']} |

## By Question Type

| Type | Total | llm_better | dyfo_better | same_good | same_bad |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(qtype_lines)}

Outputs:

- CSV: `{out_csv}`
- JSONL: `{out_jsonl}`
"""
    out_md.write_text(report, encoding="utf-8")
    print(f"paired={len(rows)}")
    print(f"pure_score={pure_total:.2f}")
    print(f"dyfo_score={dyfo_total:.2f}")
    print(f"wrote={out_csv}")
    print(f"wrote={out_jsonl}")
    print(f"wrote={out_md}")


if __name__ == "__main__":
    main()
