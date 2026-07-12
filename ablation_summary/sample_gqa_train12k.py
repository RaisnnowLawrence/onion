#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a fixed-size GQA train subset.")
    parser.add_argument("--input", default="/data2/lizhengxue/datasets/gqa/train_balanced_questions.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=12000)
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    ids_path = Path(args.ids_output)

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = list(payload.items())
    if args.count > len(items):
        raise ValueError(f"requested {args.count} samples from only {len(items)} records")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(items)), args.count))
    selected = {items[i][0]: items[i][1] for i in selected_indices}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False)

    with ids_path.open("w", encoding="utf-8") as f:
        for i in selected_indices:
            qid, sample = items[i]
            image_id = sample.get("imageId") or sample.get("image_id") or ""
            f.write(f"{i}\t{image_id}\t{qid}\n")

    print(f"seed={args.seed}")
    print(f"count={len(selected)}")
    print(f"output={out_path}")
    print(f"ids_output={ids_path}")


if __name__ == "__main__":
    main()
