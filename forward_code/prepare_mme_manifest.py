import argparse
import glob
import json
import os
import shutil

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mme_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_name", default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, f"{args.split_name}_manifest.jsonl")

    parquet_files = sorted(glob.glob(os.path.join(args.mme_root, f"{args.split_name}-*.parquet")))
    if not parquet_files and args.split_name == "test":
        parquet_files = sorted(glob.glob(os.path.join(args.mme_root, "test-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.mme_root} for split={args.split_name}")

    count = 0
    with open(manifest_path, "w") as out:
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            rows = table.to_pydict()
            num_rows = table.num_rows
            for local_idx in range(num_rows):
                image_idx = count
                image_path = os.path.join(image_dir, f"{image_idx:06d}.jpg")
                if not os.path.isfile(image_path):
                    image_payload = rows["image"][local_idx]
                    image_bytes = image_payload.get("bytes")
                    if image_bytes is None:
                        source_path = image_payload.get("path")
                        shutil.copyfile(source_path, image_path)
                    else:
                        with open(image_path, "wb") as img_out:
                            img_out.write(image_bytes)

                question_id = str(rows["question_id"][local_idx])
                rec = {
                    "image_idx": image_idx,
                    "question_id": question_id,
                    "key": f"{image_idx}<->{question_id}",
                    "question": str(rows["question"][local_idx]),
                    "answer": str(rows["answer"][local_idx]).lower(),
                    "category": str(rows["category"][local_idx]),
                    "image_path": image_path,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"manifest={manifest_path}")
    print(f"samples={count}")


if __name__ == "__main__":
    main()
