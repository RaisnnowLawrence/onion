import csv
import json
import zipfile
from pathlib import Path


REPO = Path("/data2/lizhengxue/WorkSpace/onion")
OUT_DIR = REPO / "ablation_summary"
GQA_ROOT = Path("/data2/lizhengxue/datasets/gqa")
VG_ROOT = Path("/data2/lizhengxue/datasets/visualgenome")

EXPERIMENTS = {
    "weighted_vote": "/data2/lizhengxue/WorkSpace/onion_output/gqa/gqa_testdev_qwen4b_dyfo_weighted_vote_gpu5_3shards_20260705_005703/prompt_answer_20260705_103318/VisualCOT_vinvl_n0_repeat1_imagequestion_58.30.json",
    "best_focus_answer": "/data2/lizhengxue/WorkSpace/onion_output/gqa/gqa_testdev_qwen4b_dyfo_best_focus_answer_gpu7_3shards_20260705_005703/prompt_answer_20260705_104638/VisualCOT_vinvl_n0_repeat1_imagequestion_57.51.json",
}


def load_testdev_meta():
    with zipfile.ZipFile(GQA_ROOT / "questions1.2.zip") as zf:
        data = json.load(zf.open("testdev_balanced_questions.json"))

    meta = {}
    for idx, (qid, sample) in enumerate(data.items()):
        image_id = str(sample.get("imageId") or sample.get("image_id") or "")
        key = f"{image_id}<->{qid}"
        meta[key] = {
            "idx": idx,
            "image_id": image_id,
            "question": str(sample.get("question", "")),
            "answer": str(sample.get("answer", "")),
        }
    return meta


def resolve_image_path(image_id):
    candidates = [
        GQA_ROOT / "images" / f"{image_id}.jpg",
        GQA_ROOT / "testdev_images" / f"{image_id}.jpg",
        VG_ROOT / f"{image_id}.jpg",
        VG_ROOT / "VG_100K" / f"{image_id}.jpg",
        VG_ROOT / "VG_100K_2" / f"{image_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def md_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_errors(label, pred_path, meta, limit=100):
    predictions = json.load(open(pred_path, "r", encoding="utf-8"))
    errors = []
    for row in predictions:
        if isinstance(row, list) and len(row) == 1 and isinstance(row[0], list):
            row = row[0]
        if not isinstance(row, list) or len(row) < 4:
            continue
        key = row[0]
        pred = str(row[1])
        score = float(row[3])
        if score != 0.0 or key not in meta:
            continue
        item = meta[key]
        errors.append(
            {
                "experiment": label,
                "idx": item["idx"],
                "key": key,
                "image_path": resolve_image_path(item["image_id"]),
                "question": item["question"],
                "gold_answer": item["answer"],
                "pred_answer": pred,
                "score": score,
            }
        )
    errors.sort(key=lambda row: row["idx"])
    return errors[:limit]


def write_csv(path, rows):
    fieldnames = [
        "experiment",
        "idx",
        "key",
        "image_path",
        "question",
        "gold_answer",
        "pred_answer",
        "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, title, rows, include_experiment=False):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if include_experiment:
            f.write("| # | experiment | idx | image_path | question | correct_answer | wrong_answer |\n")
            f.write("|---:|---|---:|---|---|---|---|\n")
        else:
            f.write("| # | idx | image_path | question | correct_answer | wrong_answer |\n")
            f.write("|---:|---:|---|---|---|---|\n")

        for row_id, row in enumerate(rows, start=1):
            if include_experiment:
                f.write(
                    f"| {row_id} | {row['experiment']} | {row['idx']} | "
                    f"`{md_escape(row['image_path'])}` | {md_escape(row['question'])} | "
                    f"{md_escape(row['gold_answer'])} | {md_escape(row['pred_answer'])} |\n"
                )
            else:
                f.write(
                    f"| {row_id} | {row['idx']} | `{md_escape(row['image_path'])}` | "
                    f"{md_escape(row['question'])} | {md_escape(row['gold_answer'])} | "
                    f"{md_escape(row['pred_answer'])} |\n"
                )


def main():
    meta = load_testdev_meta()
    outputs = {}

    for label, pred_path in EXPERIMENTS.items():
        rows = load_errors(label, pred_path, meta, limit=100)
        outputs[label] = rows
        write_csv(OUT_DIR / f"gqa_testdev_{label}_error_analysis_100.csv", rows)
        write_md(
            OUT_DIR / f"gqa_testdev_{label}_error_analysis_100.md",
            f"GQA testdev {label} Error Analysis - 100 Examples",
            rows,
        )

    combined = outputs["weighted_vote"][:50] + outputs["best_focus_answer"][:50]
    write_csv(OUT_DIR / "gqa_testdev_dyfo_error_analysis_combined_100.csv", combined)
    write_md(
        OUT_DIR / "gqa_testdev_dyfo_error_analysis_combined_100.md",
        "GQA testdev DyFo Error Analysis - Combined 100 Examples",
        combined,
        include_experiment=True,
    )

    print(f"weighted_vote_errors={len(outputs['weighted_vote'])}")
    print(OUT_DIR / "gqa_testdev_weighted_vote_error_analysis_100.md")
    print(f"best_focus_answer_errors={len(outputs['best_focus_answer'])}")
    print(OUT_DIR / "gqa_testdev_best_focus_answer_error_analysis_100.md")
    print(f"combined_errors={len(combined)}")
    print(OUT_DIR / "gqa_testdev_dyfo_error_analysis_combined_100.md")


if __name__ == "__main__":
    main()
