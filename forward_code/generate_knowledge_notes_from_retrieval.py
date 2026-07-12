#!/usr/bin/env python3
"""Generate NoteMR-style Knowledge Notes from an offline retrieval cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from tqdm import tqdm

from qwen_utils import chat_with_qwen_vl, initialize_qwen


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def iter_records(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("key", key)
                records.append(item)
        return records
    return data


def load_existing(path: Path) -> Dict[str, dict]:
    existing = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record.get("question_id") or record.get("qid") or "")
            key = str(record.get("key") or "")
            if key:
                existing[key] = record
            if qid:
                existing[qid] = record
    return existing


def format_retrieved_knowledge(ctxs: Iterable[dict], top_k: int, max_chars: int) -> str:
    lines = []
    for idx, ctx in enumerate(list(ctxs)[:top_k], start=1):
        text = str(ctx.get("text") or ctx.get("contents") or ctx.get("passage") or "").strip()
        if not text:
            continue
        title = ctx.get("title") or ctx.get("id") or ctx.get("rank") or idx
        source = ctx.get("source") or "preflmr_gs112k"
        lines.append(f"Knowledge {idx} ({source}:{title}): {truncate_text(text, 500)}")
    return truncate_text("\n".join(lines), max_chars)


def build_prompt(question: str, retrieved_knowledge: str, max_words: int) -> str:
    return (
        "You are generating Knowledge Notes for a visual question answering system.\n"
        "Use the question and retrieved passages to keep only useful background knowledge. "
        "Ignore misleading, overly generic, or unrelated passages.\n"
        "Do not answer the question directly. Write concise notes that help a later model answer.\n"
        f"Question: {question}\n"
        f"Retrieved knowledge:\n{retrieved_knowledge}\n"
        f"Return Knowledge Notes in no more than {max_words} words."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--engine", default="qwen3-VL-4B")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retrieved_max_chars", type=int, default=1800)
    parser.add_argument("--note_max_chars", type=int, default=700)
    parser.add_argument("--note_max_words", type=int, default=80)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path) if args.resume else {}

    records = iter_records(Path(args.retrieval_file))
    shard_records = [
        record for idx, record in enumerate(records)
        if idx % args.num_shards == args.shard_id
    ]
    if args.max_records > 0:
        shard_records = shard_records[: args.max_records]

    print(
        "[knowledge_notes] total=%s shard=%s/%s assigned=%s existing_lookup_keys=%s"
        % (len(records), args.shard_id, args.num_shards, len(shard_records), len(existing))
    )

    model, processor, _ = initialize_qwen(args.engine)
    model.eval()

    mode = "a" if args.resume else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for record in tqdm(shard_records, desc="Generate Knowledge Notes"):
            question_id = str(record.get("question_id") or record.get("qid") or "")
            image_id = record.get("image_id")
            key = str(record.get("key") or (f"{image_id}<->{question_id}" if image_id is not None and question_id else ""))
            if args.resume and ((key and key in existing) or (question_id and question_id in existing)):
                continue

            retrieved_knowledge = format_retrieved_knowledge(
                record.get("ctxs") or record.get("selected_knowledge") or [],
                args.top_k,
                args.retrieved_max_chars,
            )
            prompt = build_prompt(str(record.get("question", "")), retrieved_knowledge, args.note_max_words)
            with torch.inference_mode():
                note = chat_with_qwen_vl(
                    model,
                    processor,
                    prompt,
                    image_path=None,
                    use_images=False,
                    max_new_tokens=args.max_new_tokens,
                )
            note = truncate_text(note, args.note_max_chars)
            selected_knowledge = []
            for ctx in (record.get("ctxs") or record.get("selected_knowledge") or [])[: args.top_k]:
                text = str(ctx.get("text") or "").strip()
                if not text:
                    continue
                selected_knowledge.append({
                    "source": ctx.get("source", "preflmr_gs112k"),
                    "title": ctx.get("title") or ctx.get("id") or "",
                    "text": text,
                    "score": ctx.get("score", 0.0),
                    "rank": ctx.get("rank", len(selected_knowledge) + 1),
                })
            output_record = {
                "key": key,
                "question_id": question_id,
                "image_id": image_id,
                "question": record.get("question", ""),
                "knowledge_note": note,
                "selected_knowledge": selected_knowledge,
            }
            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

    print(f"[knowledge_notes] wrote {written} new records to {output_path}")


if __name__ == "__main__":
    main()
