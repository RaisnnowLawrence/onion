"""Answer scoring, dataset evaluation, and result merging utilities."""

import datetime
import glob
import json
import os
import re

from dataset_utils import (
    load_generic_vqa_answer_annotations,
    load_gqa_answer_annotations,
    load_hallusionbench_answer_annotations,
    load_mme_answer_annotations,
    load_textvqa_answer_annotations,
    normalize_dataset_name,
)
from official_vqa_answer_processor import normalize_vqa_answer


def process_answer(answer):
    answer = str(answer).replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)


def official_direct_answer_score(pred_answer, direct_answers):
    """Official VQA-style DA score after answer normalization: min(1, matches / 3)."""
    normalized_pred = normalize_vqa_answer(pred_answer)
    num_match = sum(normalized_pred == normalize_vqa_answer(answer) for answer in direct_answers)
    return min(1.0, num_match / 3.0)


def open_ended_answer_score(pred_answer, direct_answers):
    """Score single-answer QA by exact match and VQA-style multi-answer sets by consensus."""
    normalized_pred = normalize_vqa_answer(pred_answer)
    choice_pred = re.match(r"^\s*([a-z])(?:\s*[:.)\-]|\s*$)", str(pred_answer).strip().lower())
    if len(direct_answers) < 3:
        normalized_gold = [normalize_vqa_answer(answer) for answer in direct_answers]
        if all(re.fullmatch(r"[a-z]", answer) for answer in normalized_gold):
            pred_choice = choice_pred.group(1) if choice_pred else normalized_pred[:1]
            return 1.0 if pred_choice in normalized_gold else 0.0
        return 1.0 if normalized_pred in normalized_gold else 0.0
    return official_direct_answer_score(pred_answer, direct_answers)


def legacy_normalized_direct_answer_score(pred_answer, direct_answers):
    """Old internal score kept only for explicit backward-compatibility checks."""
    processed_pred_answer = process_answer(pred_answer)
    counter = 0
    for answer in direct_answers:
        if processed_pred_answer == process_answer(answer):
            counter += 1
    return min(1.0, float(counter) * 0.3)


def normalize_yes_no_answer(answer):
    text = normalize_vqa_answer(answer)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if text in ("true", "present"):
        return "yes"
    if text in ("false", "absent"):
        return "no"
    return text


def yes_no_answer_score(pred_answer, direct_answers):
    pred = normalize_yes_no_answer(pred_answer)
    gold = normalize_yes_no_answer(direct_answers[0] if direct_answers else "")
    return 1.0 if pred == gold else 0.0


def dataset_accuracy_label(dataset_name, full=False):
    dataset_name = normalize_dataset_name(dataset_name)
    labels = {
        "aokvqa": "A-OKVQA",
        "okvqa": "OK-VQA",
        "vqav2": "VQAv2",
        "gqa": "GQA",
        "textvqa": "TextVQA",
        "infoseek": "InfoSeek",
        "pope": "POPE",
        "mme": "MME",
        "mme_realworld": "MME-RealWorld",
        "hallusionbench": "HallusionBench",
        "mmstar": "MMStar",
    }
    label = labels.get(dataset_name, "官方DA")
    return f"全量{label}诊断" if full else f"{label}准确率"


def load_official_da_eval_keys(args):
    args.dataset_name = normalize_dataset_name(args.dataset_name)
    if args.dataset_name == "mme":
        answer_by_key, official_keys = load_mme_answer_annotations(args)
        return official_keys
    if args.dataset_name == "mme_realworld":
        answer_by_key, official_keys = load_generic_vqa_answer_annotations(args, "mme_realworld")
        return official_keys
    if args.dataset_name == "hallusionbench":
        answer_by_key, official_keys = load_hallusionbench_answer_annotations(args)
        return official_keys
    if args.dataset_name == "mmstar":
        answer_by_key, official_keys = load_generic_vqa_answer_annotations(args, "mmstar")
        return official_keys
    if args.dataset_name == "gqa":
        answer_by_key, official_keys = load_gqa_answer_annotations(args)
        return official_keys
    if args.dataset_name == "textvqa":
        answer_by_key, official_keys = load_textvqa_answer_annotations(args)
        return official_keys
    if args.dataset_name == "infoseek":
        answer_by_key, official_keys = load_generic_vqa_answer_annotations(args, "infoseek")
        return official_keys
    if args.dataset_name == "pope":
        answer_by_key, official_keys = load_direct_answer_annotations(args)
        return official_keys
    if args.dataset_name in ("okvqa", "vqav2"):
        prefix = "v2_" if args.dataset_name == "vqav2" else ""
        question_file = os.path.join(args.coco_path, f"{prefix}OpenEnded_mscoco_{args.split_name}2014_questions.json")
        try:
            questions = json.load(open(question_file, "r"))["questions"]
        except FileNotFoundError:
            return set()
        return {
            str(sample["image_id"]) + "<->" + str(sample["question_id"])
            for sample in questions
        }

    anno_file = os.path.join(args.coco_path, f"aokvqa_v1p0_{args.split_name}.json")
    try:
        annotations = json.load(open(anno_file, "r"))
    except FileNotFoundError:
        return set()
    return {
        str(sample["image_id"]) + "<->" + str(sample["question_id"])
        for sample in annotations
        if sample.get("difficult_direct_answer") is False
    }


def load_direct_answer_annotations(args):
    args.dataset_name = normalize_dataset_name(args.dataset_name)
    if args.dataset_name == "mme":
        return load_mme_answer_annotations(args)

    if args.dataset_name == "mme_realworld":
        return load_generic_vqa_answer_annotations(args, "mme_realworld")

    if args.dataset_name == "hallusionbench":
        return load_hallusionbench_answer_annotations(args)

    if args.dataset_name == "mmstar":
        return load_generic_vqa_answer_annotations(args, "mmstar")

    if args.dataset_name == "gqa":
        return load_gqa_answer_annotations(args)

    if args.dataset_name == "textvqa":
        return load_textvqa_answer_annotations(args)

    if args.dataset_name == "infoseek":
        return load_generic_vqa_answer_annotations(args, "infoseek")

    if args.dataset_name == "pope":
        subsets = ["random", "popular", "adversarial"] if args.split_name == "all" else [args.split_name]
        answer_by_key = {}
        official_keys = set()
        for subset in subsets:
            anno_file = os.path.join(args.coco_path, f"coco_pope_{subset}.json")
            try:
                f = open(anno_file, "r")
            except FileNotFoundError:
                continue
            with f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    image_id = int(sample["image"].split("_")[-1].split(".")[0])
                    key = f"{image_id}<->{subset}_{sample['question_id']}"
                    answer_by_key[key] = [str(sample["label"]).lower()]
                    official_keys.add(key)
        return answer_by_key, official_keys

    if args.dataset_name in ("okvqa", "vqav2"):
        prefix = "v2_" if args.dataset_name == "vqav2" else ""
        answer_file = os.path.join(args.coco_path, f"{prefix}mscoco_{args.split_name}2014_annotations.json")
        try:
            annotations = json.load(open(answer_file, "r"))["annotations"]
        except FileNotFoundError:
            return {}, set()
        answer_by_key = {}
        official_keys = set()
        for sample in annotations:
            key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            answer_by_key[key] = [ans["answer"] for ans in sample.get("answers", [])]
            official_keys.add(key)
        return answer_by_key, official_keys

    anno_file = os.path.join(args.coco_path, f"aokvqa_v1p0_{args.split_name}.json")
    try:
        annotations = json.load(open(anno_file, "r"))
    except FileNotFoundError:
        return {}, set()
    answer_by_key = {}
    official_keys = set()
    for sample in annotations:
        key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
        answer_by_key[key] = sample.get("direct_answers", [])
        if sample.get("difficult_direct_answer") is False:
            official_keys.add(key)
    return answer_by_key, official_keys


def direct_answer_eval_report(args, answers):
    if args.choice_only:
        acc = sum(float(a[3]) for a in answers) if answers else 0.0
        total = len(answers)
        pct = acc * 100.0 / total if total else 0.0
        return {
            "primary_label": "MC准确率",
            "primary_pct": pct,
            "primary_sum": acc,
            "primary_total": total,
            "lines": [f"MC准确率: {pct:.2f}% ({acc:.2f}/{total})"],
        }

    answer_by_key, official_keys = load_direct_answer_annotations(args)
    official_scores = []
    legacy_official_scores = []
    official_full_scores = []
    legacy_all_scores = []

    for a in answers:
        key = a[0]
        pred = a[1]
        gold = answer_by_key.get(key)
        if gold is None:
            continue
        if args.dataset_name in ("pope", "mme", "hallusionbench"):
            official_score = yes_no_answer_score(pred, gold)
            legacy_score = official_score
        else:
            official_score = open_ended_answer_score(pred, gold)
            legacy_score = legacy_normalized_direct_answer_score(pred, gold)
        official_full_scores.append(official_score)
        legacy_all_scores.append(legacy_score)
        if key in official_keys:
            official_scores.append(official_score)
            legacy_official_scores.append(legacy_score)

    def _summarize(scores):
        total = len(scores)
        score_sum = sum(scores)
        pct = score_sum * 100.0 / total if total else 0.0
        return pct, score_sum, total

    official_pct, official_sum, official_total = _summarize(official_scores)
    legacy_official_pct, legacy_official_sum, legacy_official_total = _summarize(legacy_official_scores)
    official_full_pct, official_full_sum, official_full_total = _summarize(official_full_scores)
    legacy_full_pct, legacy_full_sum, legacy_full_total = _summarize(legacy_all_scores)

    primary_label = dataset_accuracy_label(args.dataset_name, full=False)
    if args.eval_all_direct_answers:
        primary_label = dataset_accuracy_label(args.dataset_name, full=True)
        primary_pct, primary_sum, primary_total = official_full_pct, official_full_sum, official_full_total
    else:
        primary_pct, primary_sum, primary_total = official_pct, official_sum, official_total

    return {
        "primary_label": primary_label,
        "primary_pct": primary_pct,
        "primary_sum": primary_sum,
        "primary_total": primary_total,
        "official_pct": official_pct,
        "official_sum": official_sum,
        "official_total": official_total,
        "legacy_official_pct": legacy_official_pct,
        "legacy_official_sum": legacy_official_sum,
        "legacy_official_total": legacy_official_total,
        "official_full_pct": official_full_pct,
        "official_full_sum": official_full_sum,
        "official_full_total": official_full_total,
        "legacy_full_pct": legacy_full_pct,
        "legacy_full_sum": legacy_full_sum,
        "legacy_full_total": legacy_full_total,
        "lines": [
            f"{dataset_accuracy_label(args.dataset_name, full=False)}: {official_pct:.2f}% ({official_sum:.2f}/{official_total})",
            f"旧指标@{dataset_accuracy_label(args.dataset_name, full=False).replace('准确率', '')}: {legacy_official_pct:.2f}% ({legacy_official_sum:.2f}/{legacy_official_total})",
            f"{dataset_accuracy_label(args.dataset_name, full=True)}: {official_full_pct:.2f}% ({official_full_sum:.2f}/{official_full_total})",
            f"旧指标@全量诊断: {legacy_full_pct:.2f}% ({legacy_full_sum:.2f}/{legacy_full_total})",
        ],
    }


def official_da_eval_answers(args, answers):
    if args.choice_only or args.eval_all_direct_answers:
        eval_answers = answers
        label = "全量准确率"
    else:
        eval_keys = load_official_da_eval_keys(args)
        eval_answers = [a for a in answers if a[0] in eval_keys]
        label = dataset_accuracy_label(args.dataset_name, full=False)
    if not eval_answers:
        return 0.0, 0.0, 0, label
    acc = sum(float(a[3]) for a in eval_answers)
    return acc * 100.0 / len(eval_answers), acc, len(eval_answers), label


def write_official_prediction_file(args, answers, output_dir, output_name):
    predictions = {}
    for a in answers:
        qid = a[0].split('<->')[1] if '<->' in a[0] else a[0]
        if args.choice_only:
            predictions[qid] = {"multiple_choice": a[1]}
        else:
            predictions[qid] = {"direct_answer": a[1]}
    out_path = os.path.join(output_dir, f"predictions_{args.split_name}_{output_name}")
    json.dump(predictions, open(out_path, "w"))
    print(f"[merge] official prediction 已保存: {out_path}")


def merge_results(args):
    """汇总多shard的逐样本推理结果，计算全量准确率并生成最终JSON。"""
    import glob

    prompt_dir = os.path.join(args.output_path, "prompt_samples")
    format_dir = os.path.join(args.output_path, "format_samples")

    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "sample_*.json")))
    if not prompt_files:
        print(f"[merge] 错误: {prompt_dir} 中没有找到 sample_*.json 文件")
        return

    print(f"[merge] 找到 {len(prompt_files)} 个样本文件")

    answers = []
    full_answers = []

    for fpath in prompt_files:
        with open(fpath) as f:
            entry = json.load(f)
        answers.append(entry)

        basename = os.path.basename(fpath)
        format_fpath = os.path.join(format_dir, basename)
        if os.path.isfile(format_fpath):
            with open(format_fpath) as f:
                full_answers.append(json.load(f))

    report = direct_answer_eval_report(args, answers)
    acc_pct = report["primary_pct"]

    print(f"\n{'='*50}")
    for line in report["lines"]:
        print(line)
    print(f"{'='*50}\n")

    # 如果指定了summary_log，将准确率写入汇总日志
    if args.summary_log:
        with open(args.summary_log, 'a') as f:
            for line in report["lines"]:
                f.write(line + "\n")

    # 生成合并后的最终JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_answer_dir = os.path.join(args.output_path, f"prompt_answer_{timestamp}")
    format_answer_dir = os.path.join(args.output_path, f"format_answer_{timestamp}")
    os.makedirs(prompt_answer_dir, exist_ok=True)
    os.makedirs(format_answer_dir, exist_ok=True)

    output_name = f"VisualCOT_{args.caption_type}_n{args.n_shot}_repeat{args.n_ensemble}_{args.similarity_metric}_{acc_pct:.2f}.json"
    json.dump(full_answers, open(os.path.join(prompt_answer_dir, output_name), 'w'))
    print(f"[merge] prompt_answer 已保存: {prompt_answer_dir}/{output_name}")

    format_prediction = []
    for a in answers:
        rec = {
            "answer": a[1],
            "question_id": a[0].split('<->')[1] if '<->' in a[0] else a[0],
        }
        if args.chain_of_thoughts and len(a) > 5:
            rec["thoughts"] = a[5]
        format_prediction.append(rec)

    json.dump(format_prediction, open(os.path.join(format_answer_dir, output_name), 'w'))
    print(f"[merge] format_answer 已保存: {format_answer_dir}/{output_name}")
    write_official_prediction_file(args, answers, format_answer_dir, output_name)
