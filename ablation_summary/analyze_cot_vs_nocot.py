#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RUNS = {
    "nocot_r1": Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_no_cot_rounds1/prompt_samples"),
    "cot_r1": Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward2_clean_rounds1/prompt_samples"),
    "cot_r5": Path("/data2/lizhengxue/WorkSpace/onion_output/aokvqa/qwen3-VL-4B_forward_baseline/prompt_samples"),
}
OUT_DIR = Path("/data2/lizhengxue/WorkSpace/onion_output/ablation_summary")
OUT_CSV = OUT_DIR / "cot_vs_nocot_diagnostic.csv"
OUT_MD = OUT_DIR / "cot_vs_nocot_diagnostic.md"


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


def answer_len(answer):
    return len(str(answer).split())


def load_samples(root):
    rows = {}
    for path in root.glob("sample_*.json"):
        with path.open() as f:
            data = json.load(f)
        sid = sample_id(path)
        state = data[6] if len(data) > 6 and isinstance(data[6], dict) else {}
        rows[sid] = {
            "sample_id": sid,
            "image_key": data[0],
            "pred": str(data[1]),
            "question": extract_question(data[2]),
            "prompt": data[2],
            "score": float(data[3]),
            "selected_objects": data[5] if len(data) > 5 else [],
            "instruction": state.get("instruction", ""),
            "evidence_summary": state.get("evidence_summary", ""),
            "pred_candidates": state.get("pred_candidates", []),
        }
    return rows


def relation(a, b, eps=1e-9):
    if a > b + eps:
        return "a_better"
    if b > a + eps:
        return "b_better"
    return "same"


def add_group(stats, key, delta, nocot_better, cot_better, n=1):
    s = stats[key]
    s["n"] += n
    s["delta"] += delta
    s["nocot_better"] += int(nocot_better)
    s["cot_better"] += int(cot_better)


def table_rows(rows, cols, limit=12):
    lines = []
    for r in rows[:limit]:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.1f}"
            vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    data = {name: load_samples(path) for name, path in RUNS.items()}
    common = sorted(set.intersection(*(set(rows) for rows in data.values())))
    if len(common) != 1145:
        print(f"warning: common sample count={len(common)}")

    details = []
    for sid in common:
        n = data["nocot_r1"][sid]
        c1 = data["cot_r1"][sid]
        c5 = data["cot_r5"][sid]
        row = {
            "sample_id": sid,
            "question": n["question"],
            "qtype": qtype(n["question"]),
            "nocot_pred": n["pred"],
            "cot_r1_pred": c1["pred"],
            "cot_r5_pred": c5["pred"],
            "nocot_score": n["score"],
            "cot_r1_score": c1["score"],
            "cot_r5_score": c5["score"],
            "delta_cot_r1_minus_nocot": c1["score"] - n["score"],
            "delta_cot_r5_minus_nocot": c5["score"] - n["score"],
            "delta_cot_r5_minus_cot_r1": c5["score"] - c1["score"],
            "nocot_len": answer_len(n["pred"]),
            "cot_r1_len": answer_len(c1["pred"]),
            "cot_r5_len": answer_len(c5["pred"]),
            "cot_r1_instruction": c1["instruction"],
            "cot_r5_instruction": c5["instruction"],
            "cot_r5_selected_objects": ";".join(map(str, c5["selected_objects"])),
        }
        details.append(row)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    totals = {name: sum(rows[sid]["score"] for sid in common) for name, rows in data.items()}
    pair_n_c1 = Counter(relation(r["nocot_score"], r["cot_r1_score"]) for r in details)
    pair_c1_c5 = Counter(relation(r["cot_r1_score"], r["cot_r5_score"]) for r in details)
    pair_n_c5 = Counter(relation(r["nocot_score"], r["cot_r5_score"]) for r in details)

    by_qtype = defaultdict(lambda: {"n": 0, "delta_c1": 0.0, "delta_c5": 0.0, "delta_rounds": 0.0,
                                    "nocot_gt_c1": 0, "c1_gt_nocot": 0, "c1_gt_c5": 0, "c5_gt_c1": 0})
    for r in details:
        s = by_qtype[r["qtype"]]
        s["n"] += 1
        s["delta_c1"] += r["delta_cot_r1_minus_nocot"]
        s["delta_c5"] += r["delta_cot_r5_minus_nocot"]
        s["delta_rounds"] += r["delta_cot_r5_minus_cot_r1"]
        s["nocot_gt_c1"] += int(r["nocot_score"] > r["cot_r1_score"])
        s["c1_gt_nocot"] += int(r["cot_r1_score"] > r["nocot_score"])
        s["c1_gt_c5"] += int(r["cot_r1_score"] > r["cot_r5_score"])
        s["c5_gt_c1"] += int(r["cot_r5_score"] > r["cot_r1_score"])

    qtype_lines = []
    for name, s in sorted(by_qtype.items(), key=lambda kv: kv[1]["delta_c1"]):
        qtype_lines.append(
            f"| {name} | {s['n']} | {s['delta_c1']:+.1f} | {s['delta_c5']:+.1f} | "
            f"{s['delta_rounds']:+.1f} | {s['nocot_gt_c1']} | {s['c1_gt_nocot']} | "
            f"{s['c1_gt_c5']} | {s['c5_gt_c1']} |"
        )

    # Candidate failure sets.
    nocot_wins_c1 = sorted([r for r in details if r["nocot_score"] > r["cot_r1_score"]],
                           key=lambda r: (r["delta_cot_r1_minus_nocot"], r["sample_id"]))
    c1_wins_nocot = sorted([r for r in details if r["cot_r1_score"] > r["nocot_score"]],
                           key=lambda r: (-r["delta_cot_r1_minus_nocot"], r["sample_id"]))
    c1_wins_c5 = sorted([r for r in details if r["cot_r1_score"] > r["cot_r5_score"]],
                        key=lambda r: (r["delta_cot_r5_minus_cot_r1"], r["sample_id"]))
    c5_wins_c1 = sorted([r for r in details if r["cot_r5_score"] > r["cot_r1_score"]],
                        key=lambda r: (-r["delta_cot_r5_minus_cot_r1"], r["sample_id"]))

    avg_lens = {
        "nocot_r1": sum(r["nocot_len"] for r in details) / len(details),
        "cot_r1": sum(r["cot_r1_len"] for r in details) / len(details),
        "cot_r5": sum(r["cot_r5_len"] for r in details) / len(details),
    }
    instr_counts_c1 = Counter(r["cot_r1_instruction"] for r in details)
    instr_counts_c5 = Counter(r["cot_r5_instruction"] for r in details)

    cols = ["sample_id", "qtype", "nocot_score", "cot_r1_score", "cot_r5_score",
            "nocot_pred", "cot_r1_pred", "cot_r5_pred", "question"]
    report = f"""# CoT vs no-CoT Per-sample Diagnostic

## Runs Compared

| Run | Setting | Score Sum | Accuracy |
| --- | --- | ---: | ---: |
| nocot_r1 | no CoT, rounds=1 | {totals['nocot_r1']:.1f}/1145 | {totals['nocot_r1'] / 1145 * 100:.2f}% |
| cot_r1 | CoT, rounds=1 clean rerun | {totals['cot_r1']:.1f}/1145 | {totals['cot_r1'] / 1145 * 100:.2f}% |
| cot_r5 | CoT baseline, rounds=5 | {totals['cot_r5']:.1f}/1145 | {totals['cot_r5'] / 1145 * 100:.2f}% |

## Pairwise Buckets

| Comparison | First Better | Second Better | Same |
| --- | ---: | ---: | ---: |
| nocot_r1 vs cot_r1 | {pair_n_c1['a_better']} | {pair_n_c1['b_better']} | {pair_n_c1['same']} |
| nocot_r1 vs cot_r5 | {pair_n_c5['a_better']} | {pair_n_c5['b_better']} | {pair_n_c5['same']} |
| cot_r1 vs cot_r5 | {pair_c1_c5['a_better']} | {pair_c1_c5['b_better']} | {pair_c1_c5['same']} |

## By Question Type

| Type | Samples | cot_r1 - nocot | cot_r5 - nocot | cot_r5 - cot_r1 | noCoT > cot1 | cot1 > noCoT | cot1 > cot5 | cot5 > cot1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(qtype_lines)}

## Answer Length

| Run | Avg predicted answer words |
| --- | ---: |
| nocot_r1 | {avg_lens['nocot_r1']:.2f} |
| cot_r1 | {avg_lens['cot_r1']:.2f} |
| cot_r5 | {avg_lens['cot_r5']:.2f} |

## CoT Instruction Distribution

| Run | Instruction counts |
| --- | --- |
| cot_r1 | {dict(instr_counts_c1)} |
| cot_r5 | {dict(instr_counts_c5)} |

## no-CoT Correct / CoT-r1 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
{table_rows(nocot_wins_c1, cols)}

## CoT-r1 Correct / no-CoT Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
{table_rows(c1_wins_nocot, cols)}

## CoT-r1 Correct / CoT-r5 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
{table_rows(c1_wins_c5, cols)}

## CoT-r5 Correct / CoT-r1 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
{table_rows(c5_wins_c1, cols)}

## Chinese Interpretation

主要结论：

1. no-CoT rounds=1 比 CoT rounds=1 高 45 分左右，说明显式 CoT 本身会带来明显损失。
2. CoT rounds=5 又比 CoT rounds=1 低 30 分左右，说明多轮上下文/状态继续带来额外损失。
3. 因此 CoT 变差有两层原因：第一层是显式推理让模型更容易发散或被中间结论带偏；第二层是多轮 ONION 上下文会积累噪声。
4. 题型表里 `cot_r1 - nocot` 越负，说明该类题越不适合显式 CoT；`cot_r5 - cot_r1` 越负，说明该类题越容易被多轮上下文伤害。

建议：

- 最终主线继续使用 direct no-CoT。
- 如果要保留 CoT，应使用很短的结构化 visual cues，而不是完整推理链。
- CoT 更适合作为低置信样本的 verifier，而不是默认 generator。
"""
    OUT_MD.write_text(report)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
