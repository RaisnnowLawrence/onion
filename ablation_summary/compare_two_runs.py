#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two VisualCoT prompt_samples runs.")
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
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
    match = re.search(r"Question:\s*(.*)", prompt)
    return " ".join(match.group(1).split()) if match else ""


def qtype(question):
    q = question.lower()
    patterns = [
        ("where", r"\bwhere\b"),
        ("why", r"\bwhy\b"),
        ("how_many", r"\bhow many\b"),
        ("color", r"\bcolor\b"),
        ("what_type_kind", r"\bwhat (type|kind)\b"),
        ("what_object", r"\bwhat (object|item|animal|food|device|appliance)\b"),
        ("text_sign_logo", r"\b(word|written|sign|logo|brand|number|letter)\b"),
        ("activity_event", r"\b(event|activity|doing|preparing|used for|most likely)\b"),
        ("which", r"\bwhich\b"),
    ]
    for name, pattern in patterns:
        if re.search(pattern, q):
            return name
    return "other"


def load_samples(root):
    root = Path(root)
    rows = {}
    for path in root.glob("sample_*.json"):
        with path.open() as f:
            data = json.load(f)
        sid = sample_id(path)
        state = data[6] if len(data) > 6 and isinstance(data[6], dict) else {}
        rows[sid] = {
            "sample_id": sid,
            "image_key": data[0],
            "pred": data[1],
            "question": extract_question(data[2]),
            "score": float(data[3]),
            "selected_objects": data[5] if len(data) > 5 else [],
            "instruction": state.get("instruction", ""),
            "enhanced": bool(state.get("enhanced_image_path")),
            "executed_evidence": ",".join(state.get("executed_evidence", [])),
        }
    return rows


def bucket(delta):
    if delta > 1e-9:
        return "right_better"
    if delta < -1e-9:
        return "left_better"
    return "same"


def table_rows(rows, limit=15):
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"| {row['sample_id']} | {row['qtype']} | {row['left_score']:.1f} | "
            f"{row['right_score']:.1f} | {row['left_pred']} | {row['right_pred']} | "
            f"{row['right_enhanced']} | {row['question']} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    left = load_samples(args.left_dir)
    right = load_samples(args.right_dir)
    common = sorted(set(left) & set(right))

    details = []
    for sid in common:
        l = left[sid]
        r = right[sid]
        delta = r["score"] - l["score"]
        details.append({
            "sample_id": sid,
            "question": r["question"],
            "qtype": qtype(r["question"]),
            "left_pred": l["pred"],
            "right_pred": r["pred"],
            "left_score": l["score"],
            "right_score": r["score"],
            "delta": delta,
            "bucket": bucket(delta),
            "left_enhanced": l["enhanced"],
            "right_enhanced": r["enhanced"],
            "right_selected_objects": ";".join(map(str, r["selected_objects"])),
        })

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    left_total = sum(r["left_score"] for r in details)
    right_total = sum(r["right_score"] for r in details)
    buckets = Counter(r["bucket"] for r in details)
    by_qtype = defaultdict(lambda: {"n": 0, "delta": 0.0, "right_better": 0, "left_better": 0, "right_enhanced": 0})
    for r in details:
        s = by_qtype[r["qtype"]]
        s["n"] += 1
        s["delta"] += r["delta"]
        s["right_better"] += int(r["bucket"] == "right_better")
        s["left_better"] += int(r["bucket"] == "left_better")
        s["right_enhanced"] += int(r["right_enhanced"])

    qtype_lines = []
    for name, s in sorted(by_qtype.items(), key=lambda kv: kv[1]["delta"]):
        qtype_lines.append(
            f"| {name} | {s['n']} | {s['right_enhanced']} | {s['delta']:+.1f} | "
            f"{s['right_better']} | {s['left_better']} |"
        )

    right_wins = sorted([r for r in details if r["delta"] > 0], key=lambda r: (-r["delta"], r["sample_id"]))
    right_losses = sorted([r for r in details if r["delta"] < 0], key=lambda r: (r["delta"], r["sample_id"]))

    report = f"""# {args.right_name} vs {args.left_name}

## Overall

| System | Score Sum | Accuracy |
| --- | ---: | ---: |
| {args.left_name} | {left_total:.1f}/1145 | {left_total / 1145 * 100:.2f}% |
| {args.right_name} | {right_total:.1f}/1145 | {right_total / 1145 * 100:.2f}% |
| Delta ({args.right_name} - {args.left_name}) | {right_total - left_total:+.1f} | {(right_total - left_total) / 1145 * 100:+.2f} |

## Pairwise Buckets

| Bucket | Count |
| --- | ---: |
| {args.right_name} better | {buckets['right_better']} |
| {args.left_name} better | {buckets['left_better']} |
| same score | {buckets['same']} |

## By Question Type

| Type | Samples | {args.right_name} Enhanced | Delta Sum | {args.right_name} Better | {args.left_name} Better |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(qtype_lines)}

## Strongest {args.right_name} Wins

| ID | Type | {args.left_name} | {args.right_name} | {args.left_name} Pred | {args.right_name} Pred | {args.right_name} Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{table_rows(right_wins)}

## Strongest {args.right_name} Losses

| ID | Type | {args.left_name} | {args.right_name} | {args.left_name} Pred | {args.right_name} Pred | {args.right_name} Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{table_rows(right_losses)}
"""
    out_md.write_text(report)
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
