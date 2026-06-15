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
DEFAULT_COCO_PATH = "/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_annotations"

METHODS = [
    ("no_cot", "qwen3-VL-4B_forward2_no_cot_rounds1"),
    ("reflective_r3", "qwen3-VL-4B_forward2_reflective_answer_first_caption_r3_rounds1_3shards_gpu4"),
    ("answer_first_no_caption", "qwen3-VL-4B_forward2_answer_first_locked_no_caption_rounds1_3shards_gpu6"),
    ("reflective_empty_review", "qwen3-VL-4B_forward2_reflective_review_empty_context_caption_r3_3shards_gpu6"),
    ("candidate_marker_mcts", "qwen3-VL-4B_forward2_candidate_judge_marker_mcts_3shards_gpu3"),
    ("rag_protected_n400", "qwen3-VL-4B_forward2_strategy_rag_protected_reflective_train400_val_3shards"),
    ("multi_strategy_router_n400", "qwen3-VL-4B_forward2_multi_strategy_router_train400_val_3shards"),
]

STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "with", "and", "or", "is",
    "are", "be", "being", "this", "that", "these", "those", "it", "its", "their", "his",
    "her", "man", "woman", "person", "people", "someone", "something", "what", "which",
}

COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown", "gray", "grey",
    "purple", "pink", "silver", "gold", "tan", "beige",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text):
    return [t for t in normalize(text).split() if t and t not in STOPWORDS]


def answer_match(pred, gold):
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    return f" {gold_norm} " in f" {pred_norm} " or f" {pred_norm} " in f" {gold_norm} "


def semantic_answer_match(pred, gold):
    if answer_match(pred, gold):
        return True
    pred_toks = tokens(pred)
    gold_toks = tokens(gold)
    pred_nums = {NUMBER_WORDS.get(t, t) for t in pred_toks}
    gold_nums = {NUMBER_WORDS.get(t, t) for t in gold_toks}
    return bool(pred_nums and gold_nums and pred_nums & gold_nums)


def classify_question(question):
    text = str(question).lower()
    if any(cue in text for cue in ("text", "word", "letter", "sign", "read", "says", "written", "logo")):
        return "text_ocr"
    if any(cue in text for cue in ("how many", "number of", "count")):
        return "count"
    if "color" in text or "colour" in text:
        return "color"
    if any(cue in text for cue in ("where", "which side", "left", "right", "behind", "front", "next to", "under", "above")):
        return "spatial"
    if any(cue in text for cue in ("why", "used for", "use for", "purpose", "probably", "most likely", "likely", "suggests")):
        return "knowledge_reasoning"
    if any(cue in text for cue in ("what kind", "what type", "which animal", "what animal", "what food", "what object", "what item", "what device")):
        return "category_object"
    if any(cue in text for cue in ("doing", "play", "playing", "holding", "wearing", "eating", "looking")):
        return "action_attribute"
    return "general"


def classify_answer(gold_answers):
    joined = " ".join(normalize(a) for a in gold_answers)
    gold_set = set(joined.split())
    if not joined:
        return "empty"
    if gold_set <= {"yes", "no"}:
        return "yes_no"
    if any(re.fullmatch(r"\d+", normalize(a)) or normalize(a) in NUMBER_WORDS for a in gold_answers):
        return "number"
    if gold_set & COLORS:
        return "color"
    if any(len(normalize(a).split()) >= 3 for a in gold_answers):
        return "phrase"
    return "short_answer"


def extract_question(prompt):
    match = re.search(r"Question:\s*(.+?)(?:\n===|\nAnswer:|\Z)", str(prompt), flags=re.S)
    if match:
        return match.group(1).strip().splitlines()[0].strip()
    return ""


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_annotations(coco_path, split):
    path = os.path.join(coco_path, f"aokvqa_v1p0_{split}.json")
    data = read_json(path)
    anns = {}
    for sample in data:
        key = f"{sample['image_id']}<->{sample['question_id']}"
        anns[key] = {
            "question": sample.get("question", ""),
            "direct_answers": sample.get("direct_answers", []),
            "choices": sample.get("choices", []),
            "correct_choice_idx": sample.get("correct_choice_idx", ""),
            "rationales": sample.get("rationales", []),
        }
    return anns


def load_method(out_root, dirname):
    rows = {}
    for path in glob.glob(os.path.join(out_root, dirname, "prompt_samples", "sample_*.json")):
        try:
            entry = read_json(path)
        except Exception:
            continue
        if not isinstance(entry, list) or len(entry) < 4:
            continue
        state = entry[6] if len(entry) > 6 and isinstance(entry[6], dict) else {}
        pred_candidates = state.get("pred_candidates", [])
        selected_objects = state.get("selected_objects", [])
        if not isinstance(pred_candidates, list):
            pred_candidates = [pred_candidates]
        if not isinstance(selected_objects, list):
            selected_objects = [selected_objects]
        rows[entry[0]] = {
            "key": entry[0],
            "answer": str(entry[1]),
            "question": extract_question(entry[2] if len(entry) > 2 else ""),
            "score": float(entry[3]),
            "pred_candidates": [str(x) for x in pred_candidates],
            "selected_objects": [str(x) for x in selected_objects],
            "evidence_summary": str(state.get("evidence_summary", "")),
            "path": path,
        }
    return rows


def candidate_gold_covered(rec_by_method, gold_answers):
    all_preds = []
    all_objects = []
    for rec in rec_by_method.values():
        all_preds.append(rec["answer"])
        all_preds.extend(rec["pred_candidates"])
        all_objects.extend(rec["selected_objects"])
    pred_covered = any(answer_match(pred, gold) for pred in all_preds for gold in gold_answers)
    semantic_pred_covered = any(semantic_answer_match(pred, gold) for pred in all_preds for gold in gold_answers)
    gold_toks = set()
    for gold in gold_answers:
        gold_toks.update(tokens(gold))
    obj_toks = set()
    for obj in all_objects:
        obj_toks.update(tokens(obj))
    object_covered = bool(gold_toks and obj_toks and (gold_toks & obj_toks))
    return pred_covered, semantic_pred_covered, object_covered


def method_consensus(rec_by_method):
    preds = [normalize(rec["answer"]) for rec in rec_by_method.values() if normalize(rec["answer"])]
    if not preds:
        return "", 0, 0.0
    counter = Counter(preds)
    pred, count = counter.most_common(1)[0]
    return pred, count, count / len(preds)


def build_failure_tags(
    question_type,
    answer_type,
    pred_gold_covered,
    semantic_pred_gold_covered,
    object_gold_covered,
    consensus_ratio,
    gold_answers,
    rec_by_method,
):
    tags = []
    if not pred_gold_covered:
        tags.append("gold_not_in_any_prediction")
    else:
        tags.append("gold_seen_but_not_selected")
    if semantic_pred_gold_covered and not pred_gold_covered:
        tags.append("possible_format_mismatch")
    if not object_gold_covered and answer_type not in {"yes_no", "number", "phrase"}:
        tags.append("gold_object_not_selected")
    if consensus_ratio >= 0.67:
        tags.append("wrong_consensus")
    if question_type in {"text_ocr", "count", "color", "spatial"}:
        tags.append(f"needs_{question_type}")
    if question_type == "knowledge_reasoning":
        tags.append("needs_external_or_commonsense")
    preds = [normalize(rec["answer"]) for rec in rec_by_method.values()]
    if len(set(preds)) >= 4:
        tags.append("high_disagreement")
    if any(normalize(a) in {"yes", "no"} for a in gold_answers):
        tags.append("binary_answer")
    return tags


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def write_counter_table(f, title, counter, total, limit=None):
    f.write(f"\n## {title}\n\n")
    f.write("| Item | Count | Share |\n")
    f.write("|---|---:|---:|\n")
    rows = counter.most_common(limit)
    for item, count in rows:
        f.write(f"| `{item}` | {count} | {pct(count, total):.1f}% |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--coco-path", default=DEFAULT_COCO_PATH)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    annotations = load_annotations(args.coco_path, args.split)
    methods = {}
    for name, dirname in METHODS:
        rows = load_method(args.out_root, dirname)
        if rows:
            methods[name] = rows
            print(f"{name}: loaded {len(rows)} samples")
        else:
            print(f"{name}: no samples, skipped")

    if not methods:
        raise SystemExit("No method samples loaded")
    common_keys = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if not common_keys:
        raise SystemExit("No common samples found")

    all_wrong_cases = []
    for key in common_keys:
        rec_by_method = {name: rows[key] for name, rows in methods.items()}
        if max(rec["score"] for rec in rec_by_method.values()) > 0:
            continue
        ann = annotations.get(key, {})
        question = ann.get("question") or next(iter(rec_by_method.values()))["question"]
        gold_answers = ann.get("direct_answers", [])
        question_type = classify_question(question)
        answer_type = classify_answer(gold_answers)
        pred_gold_covered, semantic_pred_gold_covered, object_gold_covered = candidate_gold_covered(rec_by_method, gold_answers)
        consensus_pred, consensus_count, consensus_ratio = method_consensus(rec_by_method)
        tags = build_failure_tags(
            question_type,
            answer_type,
            pred_gold_covered,
            semantic_pred_gold_covered,
            object_gold_covered,
            consensus_ratio,
            gold_answers,
            rec_by_method,
        )
        all_wrong_cases.append({
            "key": key,
            "question": question,
            "question_type": question_type,
            "answer_type": answer_type,
            "gold_answers": gold_answers,
            "choices": ann.get("choices", []),
            "correct_choice_idx": ann.get("correct_choice_idx", ""),
            "rationales": ann.get("rationales", []),
            "pred_gold_covered": pred_gold_covered,
            "semantic_pred_gold_covered": semantic_pred_gold_covered,
            "object_gold_covered": object_gold_covered,
            "consensus_pred": consensus_pred,
            "consensus_count": consensus_count,
            "consensus_ratio": consensus_ratio,
            "failure_tags": tags,
            "methods": rec_by_method,
        })

    os.makedirs(args.report_dir, exist_ok=True)
    csv_path = os.path.join(args.report_dir, "all_wrong_cases.csv")
    fieldnames = [
        "key", "question", "question_type", "answer_type", "gold_answers", "choices",
        "correct_choice_idx", "pred_gold_covered", "semantic_pred_gold_covered",
        "object_gold_covered", "consensus_pred", "consensus_count", "consensus_ratio",
        "failure_tags",
    ]
    for name in methods:
        fieldnames.extend([f"{name}_answer", f"{name}_objects", f"{name}_candidates"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in all_wrong_cases:
            row = {k: case[k] for k in fieldnames if k in case}
            row["gold_answers"] = " | ".join(map(str, case["gold_answers"]))
            row["choices"] = " | ".join(map(str, case["choices"]))
            row["failure_tags"] = " | ".join(case["failure_tags"])
            for name, rec in case["methods"].items():
                row[f"{name}_answer"] = rec["answer"]
                row[f"{name}_objects"] = " | ".join(rec["selected_objects"])
                row[f"{name}_candidates"] = " | ".join(rec["pred_candidates"])
            writer.writerow(row)

    qtype_counter = Counter(case["question_type"] for case in all_wrong_cases)
    atype_counter = Counter(case["answer_type"] for case in all_wrong_cases)
    tag_counter = Counter(tag for case in all_wrong_cases for tag in case["failure_tags"])
    consensus_counter = Counter(case["consensus_pred"] for case in all_wrong_cases if case["consensus_ratio"] >= 0.67)

    full_qtype = Counter()
    for key in common_keys:
        ann = annotations.get(key, {})
        question = ann.get("question") or next(iter(methods.values()))[key]["question"]
        full_qtype[classify_question(question)] += 1

    md_path = os.path.join(args.report_dir, "all_wrong_cases_report.md")
    with open(md_path, "w") as f:
        f.write("# All-Wrong Case Diagnostic\n\n")
        f.write(f"Methods: `{', '.join(methods.keys())}`\n\n")
        f.write(f"Common evaluated samples: `{len(common_keys)}`\n\n")
        f.write(
            f"All-wrong samples: `{len(all_wrong_cases)}` "
            f"({pct(len(all_wrong_cases), len(common_keys)):.2f}% of validation)\n\n"
        )
        f.write("Here `all-wrong` means every included strategy has score `0` on the same question.\n")

        write_counter_table(f, "Failure Tags", tag_counter, len(all_wrong_cases))

        f.write("\n## Question Type: All-Wrong vs Full Val\n\n")
        f.write("| Type | All-Wrong | All-Wrong Share | Full Val | Full Share | Enrichment |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for qtype, count in qtype_counter.most_common():
            full_count = full_qtype[qtype]
            all_share = pct(count, len(all_wrong_cases))
            full_share = pct(full_count, len(common_keys))
            enrich = all_share / full_share if full_share else 0.0
            f.write(f"| `{qtype}` | {count} | {all_share:.1f}% | {full_count} | {full_share:.1f}% | {enrich:.2f}x |\n")

        write_counter_table(f, "Gold Answer Type", atype_counter, len(all_wrong_cases))
        write_counter_table(f, "Frequent Wrong Consensus Answers", consensus_counter, len(all_wrong_cases), limit=20)

        pred_missing = sum(1 for case in all_wrong_cases if not case["pred_gold_covered"])
        semantic_seen = sum(1 for case in all_wrong_cases if case["semantic_pred_gold_covered"])
        obj_missing = sum(1 for case in all_wrong_cases if not case["object_gold_covered"])
        wrong_consensus = sum(1 for case in all_wrong_cases if case["consensus_ratio"] >= 0.67)
        high_disagreement = sum(1 for case in all_wrong_cases if "high_disagreement" in case["failure_tags"])
        f.write("\n## Coverage Summary\n\n")
        f.write("| Diagnostic | Count | Share |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Gold answer never appears in any strategy answer/candidate | {pred_missing} | {pct(pred_missing, len(all_wrong_cases)):.1f}% |\n")
        f.write(f"| Gold answer appears after semantic normalization | {semantic_seen} | {pct(semantic_seen, len(all_wrong_cases)):.1f}% |\n")
        f.write(f"| Gold answer token absent from selected objects | {obj_missing} | {pct(obj_missing, len(all_wrong_cases)):.1f}% |\n")
        f.write(f"| At least two thirds of strategies converge to same wrong answer | {wrong_consensus} | {pct(wrong_consensus, len(all_wrong_cases)):.1f}% |\n")
        f.write(f"| Strategies strongly disagree | {high_disagreement} | {pct(high_disagreement, len(all_wrong_cases)):.1f}% |\n")

        f.write("\n## Representative Cases\n\n")
        for case in all_wrong_cases[:35]:
            f.write(f"### `{case['key']}`\n\n")
            f.write(f"- Question: {case['question']}\n")
            f.write(f"- Gold: {', '.join(map(str, case['gold_answers']))}\n")
            f.write(f"- Type: `{case['question_type']}` / `{case['answer_type']}`\n")
            f.write(f"- Tags: `{', '.join(case['failure_tags'])}`\n")
            f.write(f"- Consensus: `{case['consensus_pred']}` ({case['consensus_count']}/{len(methods)})\n")
            preds = "; ".join(f"{name}={rec['answer']}" for name, rec in case["methods"].items())
            f.write(f"- Predictions: {preds}\n\n")

        f.write("\n## Files\n\n")
        f.write(f"- CSV: `{csv_path}`\n")

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
