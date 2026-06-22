#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forward_code"))
from official_vqa_answer_processor import normalize_vqa_answer


def official_direct_answer_score(pred_answer, direct_answers):
    normalized_pred = normalize_vqa_answer(pred_answer)
    num_match = sum(normalized_pred == normalize_vqa_answer(answer) for answer in direct_answers)
    return min(1.0, num_match / 3.0)


def score_label(score):
    if abs(score) < 1e-9:
        return "0"
    text = f"{score:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def image_path(raw_image_dir, split, image_id):
    return os.path.join(
        raw_image_dir,
        f"{split}2017",
        f"COCO_{split}2014_{int(image_id):012d}.jpg",
    )


def load_predictions(prompt_dir):
    predictions = {}
    for fpath in glob.glob(os.path.join(prompt_dir, "sample_*.json")):
        rec = json.load(open(fpath))
        predictions[rec[0]] = {
            "key": rec[0],
            "direct_answer": rec[1],
            "prompt": rec[2],
            "saved_score": rec[3],
            "selected_objects": rec[5] if len(rec) > 5 else [],
            "round_state": rec[6] if len(rec) > 6 else {},
            "file": fpath,
        }
    return predictions


def load_annotations(anno_file):
    annotations = {}
    for sample in json.load(open(anno_file)):
        key = f"{sample['image_id']}<->{sample['question_id']}"
        annotations[key] = sample
    return annotations


def clean_translation_text(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def translate_with_qwen(texts, model_name, batch_size, max_new_tokens, cache_file=None, existing=None):
    if not texts:
        return {}

    from qwen_utils import chat_with_qwen_vl, initialize_qwen

    model, processor, _ = initialize_qwen(model_name)
    translations = {} if existing is None else dict(existing)
    items = sorted(set(str(text) for text in texts if str(text).strip()))

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        print(f"[translate] batch {start // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size}: {len(batch)} items", flush=True)
        payload = [{"id": i, "text": text} for i, text in enumerate(batch)]
        prompt = (
            "Translate each English VQA question or short answer into concise, natural Chinese.\n"
            "Keep numbers, brand names, and proper nouns as literal as possible.\n"
            "Do not explain. Do not add notes.\n"
            "Return only a JSON object mapping each id to Chinese translation.\n"
            f"Items:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        response = chat_with_qwen_vl(
            model,
            processor,
            prompt,
            image_path=None,
            max_new_tokens=max_new_tokens,
            use_images=False,
        )
        try:
            parsed = json.loads(clean_translation_text(response))
        except json.JSONDecodeError:
            parsed = {}
            for line in response.splitlines():
                if ":" not in line:
                    continue
                left, right = line.split(":", 1)
                left = left.strip().strip('"')
                if left.isdigit():
                    parsed[left] = right.strip().strip('",')
        for idx, source in enumerate(batch):
            translations[source] = str(parsed.get(str(idx), source)).strip()
        if cache_file:
            json.dump(translations, open(cache_file, "w"), ensure_ascii=False, indent=2)

    return translations


def load_or_create_translations(texts, cache_file, mode, model_name, batch_size, max_new_tokens):
    existing = {}
    if os.path.isfile(cache_file):
        existing = json.load(open(cache_file))

    all_texts = set(str(t) for t in texts if str(t).strip())
    if mode == "qwen":
        missing = sorted(text for text in all_texts if not str(existing.get(text, "")).strip())
    else:
        missing = sorted(all_texts - set(existing))
    if mode == "qwen" and missing:
        existing = translate_with_qwen(
            missing,
            model_name,
            batch_size,
            max_new_tokens,
            cache_file=cache_file,
            existing=existing,
        )
    else:
        for text in missing:
            existing[text] = ""
        if missing:
            json.dump(existing, open(cache_file, "w"), ensure_ascii=False, indent=2)
    return existing


def md_escape(text):
    return str(text).replace("\n", " ").strip()


def write_group_report(out_dir, label, rows, translations):
    out_path = os.path.join(out_dir, f"score_{label}.md")
    with open(out_path, "w") as f:
        f.write(f"# Direct Errors: Score {label.replace('p', '.')}\n\n")
        f.write(f"Total samples: {len(rows)}\n\n")
        for idx, row in enumerate(rows, 1):
            ann = row["annotation"]
            pred = row["prediction"]
            q = ann.get("question", "")
            gold = ann.get("direct_answers", [])
            direct = pred.get("direct_answer", "")
            f.write(f"## {idx}. {row['key']}\n\n")
            f.write(f"- Image path: `{row['image_path']}`\n")
            f.write(f"- Question: {md_escape(q)}\n")
            f.write(f"- 问题中文翻译: {md_escape(translations.get(q, ''))}\n")
            f.write(f"- Gold answers: {md_escape(gold)}\n")
            gold_zh = [translations.get(str(ans), "") for ans in gold]
            f.write(f"- 答案中文翻译: {md_escape(gold_zh)}\n")
            f.write(f"- Direct answer: {md_escape(direct)}\n")
            f.write(f"- Direct回答中文翻译: {md_escape(translations.get(str(direct), ''))}\n")
            f.write(f"- Official score: {row['score']:.3f}\n")
            f.write("\n其他信息:\n\n")
            other = {
                "split": ann.get("split"),
                "image_id": ann.get("image_id"),
                "question_id": ann.get("question_id"),
                "choices": ann.get("choices"),
                "correct_choice_idx": ann.get("correct_choice_idx"),
                "difficult_direct_answer": ann.get("difficult_direct_answer"),
                "rationales": ann.get("rationales"),
                "selected_objects": pred.get("selected_objects"),
                "round_state": pred.get("round_state"),
                "prompt_sample_file": pred.get("file"),
            }
            f.write("```json\n")
            f.write(json.dumps(other, ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", required=True)
    parser.add_argument("--anno_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--raw_image_dir", default="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco17")
    parser.add_argument("--split", default="val")
    parser.add_argument("--translate", choices=["none", "qwen"], default="none")
    parser.add_argument("--translation_model", default="qwen3-VL-4B")
    parser.add_argument("--translation_batch_size", type=int, default=5)
    parser.add_argument("--translation_max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--official_scored_only",
        action="store_true",
        help="Keep only A-OKVQA official direct-answer scored samples (difficult_direct_answer == False).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    predictions = load_predictions(args.prompt_dir)
    annotations = load_annotations(args.anno_file)

    groups = defaultdict(list)
    translation_texts = set()

    for key, pred in sorted(predictions.items()):
        ann = annotations.get(key)
        if not ann:
            continue
        if args.official_scored_only and ann.get("difficult_direct_answer"):
            continue
        score = official_direct_answer_score(pred["direct_answer"], ann.get("direct_answers", []))
        if score >= 1.0:
            continue
        row = {
            "key": key,
            "score": score,
            "annotation": ann,
            "prediction": pred,
            "image_path": image_path(args.raw_image_dir, args.split, ann["image_id"]),
        }
        groups[score_label(score)].append(row)
        translation_texts.add(ann.get("question", ""))
        translation_texts.add(str(pred.get("direct_answer", "")))
        for ans in ann.get("direct_answers", []):
            translation_texts.add(str(ans))

    cache_file = os.path.join(args.out_dir, "translations_cache.json")
    translations = load_or_create_translations(
        translation_texts,
        cache_file,
        args.translate,
        args.translation_model,
        args.translation_batch_size,
        args.translation_max_new_tokens,
    )

    summary = {
        "prompt_dir": args.prompt_dir,
        "anno_file": args.anno_file,
        "split": args.split,
        "total_non_full": sum(len(rows) for rows in groups.values()),
        "groups": {label: len(rows) for label, rows in sorted(groups.items())},
        "translation_mode": args.translate,
        "official_scored_only": args.official_scored_only,
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Direct Non-Full-Score Case Reports\n\n")
        f.write("Grouped by official direct-answer score.\n\n")
        f.write("```json\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n```\n")

    for label, rows in sorted(groups.items()):
        write_group_report(args.out_dir, label, rows, translations)


if __name__ == "__main__":
    main()
