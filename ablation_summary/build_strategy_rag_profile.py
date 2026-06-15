#!/usr/bin/env python3
import argparse
import glob
import json
import os


def iter_records(patterns):
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if not paths and os.path.isfile(pattern):
            paths = [pattern]
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield path, json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", required=True,
                        help="profile JSONL path or glob; repeat for multiple strategies")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = {}
    for _, rec in iter_records(args.profile):
        key = rec.get("key")
        strategy = rec.get("strategy")
        if not key or not strategy:
            continue
        out = merged.setdefault(key, {
            "key": key,
            "image_id": rec.get("image_id"),
            "question": rec.get("question", ""),
            "question_type": rec.get("question_type", ""),
            "split": rec.get("split", ""),
            "scores": {},
            "answers": {},
        })
        out["scores"][strategy] = float(rec.get("score", 0.0))
        out["answers"][strategy] = rec.get("pred_answer", "")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for key in sorted(merged):
            f.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    complete = sum(1 for rec in merged.values() if len(rec.get("scores", {})) >= 2)
    print(f"wrote {len(merged)} samples to {args.output}")
    print(f"samples with >=2 strategies: {complete}")


if __name__ == "__main__":
    main()
