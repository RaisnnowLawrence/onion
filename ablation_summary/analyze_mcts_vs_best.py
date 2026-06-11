#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BEST_DIR = Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_no_cot_rounds1/prompt_samples")
MCTS_DIR = Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_mcts_safe_no_cot_rounds1_n5_6shards/prompt_samples")
OUT_DIR = Path("/data2/lizhengxue/WorkSpace/onion_output/ablation_summary")
DETAIL_CSV = OUT_DIR / "mcts_vs_best_diagnostic.csv"
REPORT_MD = OUT_DIR / "mcts_vs_best_diagnostic.md"


def sample_id(path):
    match = re.search(r"sample_(\d+)_", path.name)
    if not match:
        raise ValueError(f"cannot parse sample id from {path}")
    return int(match.group(1))


def load_samples(root):
    rows = {}
    for path in root.glob("sample_*.json"):
        with path.open() as f:
            data = json.load(f)
        sid = sample_id(path)
        state = data[6] if len(data) > 6 and isinstance(data[6], dict) else {}
        question = extract_question(data[2])
        rows[sid] = {
            "sample_id": sid,
            "image_key": data[0],
            "pred": data[1],
            "question": question,
            "score": float(data[3]),
            "selected_objects": data[5] if len(data) > 5 else [],
            "instruction": state.get("instruction", ""),
            "enhanced": bool(state.get("enhanced_image_path")),
            "executed_evidence": ",".join(state.get("executed_evidence", [])),
            "evidence_summary": state.get("evidence_summary", ""),
        }
    return rows


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


def bucket(delta):
    if delta > 1e-9:
        return "mcts_better"
    if delta < -1e-9:
        return "best_better"
    return "same"


def short_rows(rows, limit=12):
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"| {row['sample_id']} | {row['qtype']} | {row['best_score']:.1f} | "
            f"{row['mcts_score']:.1f} | {row['best_pred']} | {row['mcts_pred']} | "
            f"{row['mcts_enhanced']} | {row['question']} |"
        )
    return "\n".join(lines)


def main():
    best = load_samples(BEST_DIR)
    mcts = load_samples(MCTS_DIR)
    common = sorted(set(best) & set(mcts))
    if len(common) != 1145:
        print(f"warning: common samples={len(common)}")

    details = []
    for sid in common:
        b = best[sid]
        m = mcts[sid]
        delta = m["score"] - b["score"]
        details.append({
            "sample_id": sid,
            "question": m["question"],
            "qtype": qtype(m["question"]),
            "best_pred": b["pred"],
            "mcts_pred": m["pred"],
            "best_score": b["score"],
            "mcts_score": m["score"],
            "delta": delta,
            "bucket": bucket(delta),
            "mcts_instruction": m["instruction"],
            "mcts_enhanced": m["enhanced"],
            "mcts_selected_objects": ";".join(map(str, m["selected_objects"])),
        })

    total_best = sum(r["best_score"] for r in details)
    total_mcts = sum(r["mcts_score"] for r in details)
    enhanced = [r for r in details if r["mcts_enhanced"]]
    not_enhanced = [r for r in details if not r["mcts_enhanced"]]

    by_bucket = Counter(r["bucket"] for r in details)
    by_qtype = defaultdict(lambda: {"n": 0, "delta": 0.0, "enhanced": 0, "mcts_better": 0, "best_better": 0})
    for r in details:
        s = by_qtype[r["qtype"]]
        s["n"] += 1
        s["delta"] += r["delta"]
        s["enhanced"] += int(r["mcts_enhanced"])
        s["mcts_better"] += int(r["bucket"] == "mcts_better")
        s["best_better"] += int(r["bucket"] == "best_better")

    with DETAIL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    mcts_wins = sorted([r for r in details if r["delta"] > 0], key=lambda r: (-r["delta"], r["sample_id"]))
    mcts_losses = sorted([r for r in details if r["delta"] < 0], key=lambda r: (r["delta"], r["sample_id"]))
    enhanced_losses = [r for r in mcts_losses if r["mcts_enhanced"]]
    enhanced_wins = [r for r in mcts_wins if r["mcts_enhanced"]]

    qtype_lines = []
    for name, s in sorted(by_qtype.items(), key=lambda kv: kv[1]["delta"]):
        qtype_lines.append(
            f"| {name} | {s['n']} | {s['enhanced']} | {s['delta']:+.1f} | "
            f"{s['mcts_better']} | {s['best_better']} |"
        )

    report = f"""# Safe MCTS vs Best no-CoT Diagnostic

## Overall

| System | Score Sum | Accuracy |
| --- | ---: | ---: |
| best no-CoT rounds1 | {total_best:.1f}/1145 | {total_best / 1145 * 100:.2f}% |
| safe MCTS n=5 | {total_mcts:.1f}/1145 | {total_mcts / 1145 * 100:.2f}% |
| Delta | {total_mcts - total_best:+.1f} | {(total_mcts - total_best) / 1145 * 100:+.2f} |

## Pairwise Buckets

| Bucket | Count |
| --- | ---: |
| MCTS better | {by_bucket['mcts_better']} |
| best no-CoT better | {by_bucket['best_better']} |
| same score | {by_bucket['same']} |

## Enhancement Split

| Split | Samples | Delta Sum | Avg Delta |
| --- | ---: | ---: | ---: |
| MCTS actually enhanced | {len(enhanced)} | {sum(r['delta'] for r in enhanced):+.1f} | {sum(r['delta'] for r in enhanced) / max(1, len(enhanced)):+.3f} |
| MCTS not enhanced/skipped | {len(not_enhanced)} | {sum(r['delta'] for r in not_enhanced):+.1f} | {sum(r['delta'] for r in not_enhanced) / max(1, len(not_enhanced)):+.3f} |

## By Question Type

| Type | Samples | Enhanced | Delta Sum | MCTS Better | Best Better |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(qtype_lines)}

## Strongest MCTS Wins

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{short_rows(mcts_wins)}

## Strongest MCTS Losses

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{short_rows(mcts_losses)}

## Enhanced-only Wins

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{short_rows(enhanced_wins)}

## Enhanced-only Losses

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
{short_rows(enhanced_losses)}

## Chinese Takeaway

Safe MCTS 比原始 MCTS 明显更好，但和 best no-CoT 相比仍然净亏。关键不是搜索次数不够，而是增强触发后仍会在一部分问题上改变模型答案，带来额外错误。

最重要的诊断看两处：

- `MCTS actually enhanced` 的净 delta：如果这里为负，说明真正执行 MCTS 的题整体仍在伤害结果。
- `MCTS not enhanced/skipped` 的净 delta：如果这里也有差异，说明即使没有增强，运行随机性、缓存或 prompt 路径也会造成回答波动。

下一步优先改路由和 reward，而不是先盲目提高 `n_simulations`。
"""
    REPORT_MD.write_text(report)
    print(f"wrote {DETAIL_CSV}")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
