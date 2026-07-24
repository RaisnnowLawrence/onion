#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import Counter

from official_vqa_answer_processor import normalize_vqa_answer


def load_round_state(sample):
    for item in reversed(sample if isinstance(sample, list) else []):
        if isinstance(item, dict) and item.get("type") == "round_state":
            return item
    return {}


def is_correct(answer, gold):
    return bool(answer) and normalize_vqa_answer(answer) == normalize_vqa_answer(gold)


def main():
    parser = argparse.ArgumentParser(description="Analyze DyFo node-region oracle diagnostics.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--questions",
        default="/data2/lizhengxue/datasets/gqa/testdev_balanced_questions.json",
    )
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, "region_audit_report")
    os.makedirs(output_dir, exist_ok=True)
    with open(args.questions, "r", encoding="utf-8") as handle:
        questions = json.load(handle)

    rows = []
    errors = []
    for sample_path in sorted(glob.glob(os.path.join(args.run_dir, "prompt_samples", "sample_*.json"))):
        with open(sample_path, "r", encoding="utf-8") as handle:
            sample = json.load(handle)
        state = load_round_state(sample)
        trace = state.get("dyfo_decision_trace") or {}
        audit = trace.get("region_audit") or {}
        key = str(sample[0]) if sample else ""
        question_id = key.split("<->")[-1]
        question_row = questions.get(question_id, {})
        gold = str(question_row.get("answer", ""))
        if audit.get("error") or not audit.get("nodes"):
            errors.append({"key": key, "sample_path": sample_path, "error": audit.get("error", "missing audit")})
            continue

        nodes = [node for node in audit["nodes"] if node.get("depth", 0) > 0]
        valid_nodes = [
            node for node in nodes
            if node.get("all_targets_present", False) and node.get("local_answer")
        ]
        correct_nodes = [node for node in valid_nodes if is_correct(node.get("local_answer", ""), gold)]
        best_node = next((node for node in nodes if node.get("is_best_node")), {})
        pure_answer = str(trace.get("pure_baseline_answer", ""))
        candidate_answer = str(trace.get("dyfo_candidate_answer", trace.get("best_focus_answer", "")))
        final_answer = str(state.get("pred_answer", trace.get("token_confidence_final_answer", "")))
        masked_answer = str(audit.get("best_masked_answer", ""))
        pure_correct = is_correct(pure_answer, gold)
        candidate_correct = is_correct(candidate_answer, gold)
        final_correct = is_correct(final_answer, gold)
        best_correct = is_correct(best_node.get("local_answer", ""), gold)
        oracle_correct = bool(correct_nodes)
        masked_correct = is_correct(masked_answer, gold)
        search_failure = pure_correct and not oracle_correct
        selection_failure = oracle_correct and not candidate_correct
        retention_pass_no_correct = bool(valid_nodes) and not oracle_correct
        mask_changed_answer = (
            bool(masked_answer) and normalize_vqa_answer(masked_answer) != normalize_vqa_answer(pure_answer)
        )
        mask_breaks_correct_pure = pure_correct and not masked_correct

        row = {
            "key": key,
            "question_id": question_id,
            "question": question_row.get("question", audit.get("question", "")),
            "gold_answer": gold,
            "original_image_path": audit.get("original_image_path", ""),
            "asset_dir": audit.get("asset_dir", ""),
            "pure_answer": pure_answer,
            "pure_correct": int(pure_correct),
            "dyfo_candidate_answer": candidate_answer,
            "candidate_correct": int(candidate_correct),
            "final_answer": final_answer,
            "final_correct": int(final_correct),
            "best_node_answer": best_node.get("local_answer", ""),
            "best_node_correct": int(best_correct),
            "node_oracle_correct": int(oracle_correct),
            "correct_node_count": len(correct_nodes),
            "valid_node_count": len(valid_nodes),
            "focused_node_count": audit.get("focused_node_count", len(nodes)),
            "unique_region_count": audit.get("unique_region_count", 0),
            "pairwise_iou_mean": audit.get("pairwise_iou_mean", 0.0),
            "pairwise_iou_max": audit.get("pairwise_iou_max", 0.0),
            "best_crop_area_ratio": audit.get("best_crop_area_ratio", 0.0),
            "masked_answer": masked_answer,
            "masked_correct": int(masked_correct),
            "mask_changed_answer": int(mask_changed_answer),
            "mask_breaks_correct_pure": int(mask_breaks_correct_pure),
            "search_failure": int(search_failure),
            "selection_failure": int(selection_failure),
            "retention_pass_no_correct_node": int(retention_pass_no_correct),
            "correct_node_indices": ";".join(str(node.get("index")) for node in correct_nodes),
            "sample_path": sample_path,
        }
        rows.append(row)

    csv_path = os.path.join(output_dir, "region_audit_samples.csv")
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(output_dir, "region_audit_errors.json"), "w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)

    total = len(rows)
    count = lambda field: sum(int(row[field]) for row in rows)
    avg = lambda field: sum(float(row[field]) for row in rows) / total if total else 0.0
    valid_node_total = sum(int(row["valid_node_count"]) for row in rows)
    unique_total = sum(int(row["unique_region_count"]) for row in rows)
    focused_total = sum(int(row["focused_node_count"]) for row in rows)
    category_counts = Counter()
    for row in rows:
        if row["search_failure"]:
            category_counts["search_failure"] += 1
        if row["selection_failure"]:
            category_counts["selection_failure"] += 1
        if row["retention_pass_no_correct_node"]:
            category_counts["retention_pass_no_correct_node"] += 1

    report_path = os.path.join(output_dir, "REGION_AUDIT_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# GQA DyFo Region-Oracle Audit\n\n")
        handle.write(f"- Run directory: `{args.run_dir}`\n")
        handle.write(f"- Analyzed samples: **{total}**\n")
        handle.write(f"- Audit errors: **{len(errors)}**\n\n")
        handle.write("## Accuracy decomposition\n\n")
        handle.write("| Metric | Count | Rate |\n|---|---:|---:|\n")
        for label, field in [
            ("Pure original-image correct", "pure_correct"),
            ("Best selected node correct", "best_node_correct"),
            ("Weighted DyFo candidate correct", "candidate_correct"),
            ("Any valid node correct (node oracle)", "node_oracle_correct"),
            ("Final routed answer correct", "final_correct"),
        ]:
            value = count(field)
            handle.write(f"| {label} | {value} | {100.0 * value / total if total else 0.0:.2f}% |\n")
        handle.write("\n## Failure localization\n\n")
        handle.write("| Diagnostic | Count | Rate |\n|---|---:|---:|\n")
        for label, field in [
            ("Search failure: Pure correct, no valid node correct", "search_failure"),
            ("Selection failure: oracle node exists, candidate wrong", "selection_failure"),
            ("Retention passed but no node answered correctly", "retention_pass_no_correct_node"),
            ("Masking best region changes Pure answer", "mask_changed_answer"),
            ("Masking best region breaks a correct Pure answer", "mask_breaks_correct_pure"),
        ]:
            value = count(field)
            handle.write(f"| {label} | {value} | {100.0 * value / total if total else 0.0:.2f}% |\n")
        handle.write("\n## Region diversity\n\n")
        handle.write(f"- Mean pairwise node IoU: **{avg('pairwise_iou_mean'):.4f}**\n")
        handle.write(f"- Mean maximum pairwise node IoU: **{avg('pairwise_iou_max'):.4f}**\n")
        handle.write(f"- Mean best-crop area ratio: **{avg('best_crop_area_ratio'):.4f}**\n")
        handle.write(
            f"- Unique-region ratio: **{unique_total / focused_total if focused_total else 0.0:.4f}** "
            f"({unique_total}/{focused_total}, IoU < 0.90)\n"
        )
        handle.write(f"- Mean valid nodes per sample: **{valid_node_total / total if total else 0.0:.2f}**\n\n")
        handle.write("## High-signal cases\n\n")
        handle.write("The CSV contains every sample. Representative failure cases are listed below.\n\n")
        for title, field in [
            ("Search failures", "search_failure"),
            ("Selection failures", "selection_failure"),
            ("High-overlap searches", "pairwise_iou_mean"),
        ]:
            handle.write(f"### {title}\n\n")
            selected = (
                sorted(rows, key=lambda row: float(row[field]), reverse=True)[:20]
                if field == "pairwise_iou_mean"
                else [row for row in rows if row[field]][:20]
            )
            for row in selected:
                handle.write(
                    f"- `{row['key']}`: {row['question']} | GT `{row['gold_answer']}` | "
                    f"Pure `{row['pure_answer']}` | DyFo `{row['dyfo_candidate_answer']}` | "
                    f"assets `{row['asset_dir']}`\n"
                )
            handle.write("\n")

    metrics = {
        "samples": total,
        "errors": len(errors),
        "pure_accuracy": count("pure_correct") / total if total else 0.0,
        "best_node_accuracy": count("best_node_correct") / total if total else 0.0,
        "candidate_accuracy": count("candidate_correct") / total if total else 0.0,
        "node_oracle_accuracy": count("node_oracle_correct") / total if total else 0.0,
        "final_accuracy": count("final_correct") / total if total else 0.0,
        "search_failures": category_counts["search_failure"],
        "selection_failures": category_counts["selection_failure"],
        "mean_pairwise_iou": avg("pairwise_iou_mean"),
        "unique_region_ratio": unique_total / focused_total if focused_total else 0.0,
    }
    with open(os.path.join(output_dir, "region_audit_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
