#!/usr/bin/env python3
"""Build a query-focused Wiki21M subcorpus for A-OKVQA.

This is a lightweight first-stage retriever for the "BM25 coarse recall ->
FLMR rerank/index" route. It streams Wiki21M once to collect top passages per
question, then streams it again to write only the selected passages.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "had", "he", "her", "his", "how", "i", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "there", "these", "this", "those",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with", "would",
    "could", "should", "does", "do", "did", "can", "will", "best", "most", "likely",
    "probably", "image", "photo", "picture", "seen", "shown", "look", "looks", "man",
    "woman", "person", "people", "thing", "object", "one", "two", "three",
}


def tokenize(text: str) -> List[str]:
    return [
        tok.strip("'_-")
        for tok in TOKEN_RE.findall(text.lower())
        if len(tok.strip("'_-")) >= 3 and tok.strip("'_-") not in STOPWORDS
    ]


def load_json(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_query_texts(
    train_file: Path,
    val_file: Path,
    include_train_gt: bool = True,
    query_scope: str = "train_val",
) -> Iterable[Tuple[str, str, str]]:
    if query_scope == "val":
        split_paths = (("val", val_file),)
    elif query_scope == "train":
        split_paths = (("train", train_file),)
    else:
        split_paths = (("train", train_file), ("val", val_file))

    for split, path in split_paths:
        for item in load_json(path):
            qid = str(item["question_id"])
            parts = [item.get("question", "")]
            choices = item.get("choices") or []
            if choices:
                parts.append(" ".join(map(str, choices)))
            if split == "train" and include_train_gt:
                parts.extend(map(str, item.get("direct_answers") or []))
                parts.extend(map(str, item.get("rationales") or []))
            yield split, qid, " ".join(p for p in parts if p)


def build_query_index(args) -> Tuple[List[dict], Dict[str, List[Tuple[int, float]]]]:
    queries: List[dict] = []
    raw_tf: List[Counter] = []
    df = Counter()

    for split, qid, text in iter_query_texts(
        Path(args.train_file),
        Path(args.val_file),
        args.include_train_gt,
        args.query_scope,
    ):
        terms = tokenize(text)
        tf = Counter(terms)
        if not tf:
            continue
        raw_tf.append(tf)
        df.update(tf.keys())
        queries.append({"split": split, "question_id": qid, "text": text})

    n_queries = len(queries)
    term_to_queries: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for idx, tf in enumerate(raw_tf):
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        for term, count in tf.items():
            if df[term] > args.max_query_df:
                continue
            idf = math.log((n_queries + 1.0) / (df[term] + 0.5))
            term_to_queries[term].append((idx, (1.0 + math.log(count)) * idf / norm))

    return queries, term_to_queries


def update_heap(heap: List[Tuple[float, int]], item: Tuple[float, int], k: int) -> None:
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def iter_wiki_rows(path: Path) -> Iterable[Tuple[int, str, str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            wiki_id = str(row.get("id") or row_idx)
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if text:
                yield row_idx, wiki_id, title, text


def coarse_recall(args, queries: List[dict], term_to_queries: Dict[str, List[Tuple[int, float]]]):
    heaps: List[List[Tuple[float, int]]] = [[] for _ in queries]
    best_scores: Dict[int, float] = {}
    hit_counts: Counter = Counter()
    start = time.time()

    for row_idx, wiki_id, title, text in iter_wiki_rows(Path(args.wiki21m_file)):
        terms = tokenize(title + " " + title + " " + text[: args.max_passage_chars])
        if not terms:
            continue
        tf = Counter(terms)
        scores = defaultdict(float)
        updates = 0
        for term, count in tf.items():
            q_entries = term_to_queries.get(term)
            if not q_entries:
                continue
            passage_weight = 1.0 + math.log(count)
            for qidx, qweight in q_entries:
                scores[qidx] += passage_weight * qweight
                updates += 1
                if updates >= args.max_updates_per_passage:
                    break
            if updates >= args.max_updates_per_passage:
                break
        if scores:
            length_norm = 1.0 / math.sqrt(20.0 + min(len(terms), args.max_passage_terms))
            for qidx, score in scores.items():
                final_score = score * length_norm
                if final_score >= args.min_score:
                    update_heap(heaps[qidx], (final_score, row_idx), args.top_k_per_query)
                    if final_score > best_scores.get(row_idx, float("-inf")):
                        best_scores[row_idx] = final_score
                    hit_counts[row_idx] += 1

        if args.max_wiki_rows and row_idx + 1 >= args.max_wiki_rows:
            break
        if (row_idx + 1) % args.progress_every == 0:
            selected = len(best_scores)
            elapsed = time.time() - start
            print(
                f"[coarse] rows={row_idx + 1:,} selected={selected:,} "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    union = set()
    per_query_hits = {}
    for qidx, heap in enumerate(heaps):
        ranked = sorted(heap, reverse=True)
        per_query_hits[str(qidx)] = [
            {"score": score, "wiki_row_idx": row_idx}
            for score, row_idx in ranked
        ]
        for _, row_idx in ranked:
            union.add(row_idx)

    ranked_rows = sorted(
        union,
        key=lambda idx: (best_scores.get(idx, 0.0), hit_counts[idx]),
        reverse=True,
    )
    if args.max_subcorpus and len(ranked_rows) > args.max_subcorpus:
        ranked_rows = ranked_rows[: args.max_subcorpus]
    return set(ranked_rows), best_scores, hit_counts, per_query_hits


def write_subcorpus(args, selected_rows: set, best_scores: Dict[int, float], hit_counts: Counter) -> int:
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kid", "text", "title", "wiki_id", "wiki_row_idx", "coarse_score", "query_hits"])
        writer.writeheader()
        for row_idx, wiki_id, title, text in iter_wiki_rows(Path(args.wiki21m_file)):
            if row_idx not in selected_rows:
                continue
            passage = f"{title}: {text}" if title else text
            writer.writerow(
                {
                    "kid": f"wiki21m:{wiki_id}",
                    "text": passage,
                    "title": title,
                    "wiki_id": wiki_id,
                    "wiki_row_idx": row_idx,
                    "coarse_score": f"{best_scores.get(row_idx, 0.0):.6f}",
                    "query_hits": hit_counts[row_idx],
                }
            )
            count += 1
            if count % 50000 == 0:
                print(f"[write] written={count:,}", flush=True)
        return count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki21m_file", default="/data2/lizhengxue/datasets/wiki21m/psgs_w100.tsv")
    parser.add_argument("--train_file", default="/data2/lizhengxue/datasets/aokvqa/aokvqa_v1p0_train.json")
    parser.add_argument("--val_file", default="/data2/lizhengxue/datasets/aokvqa/aokvqa_v1p0_val.json")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--manifest_file", required=True)
    parser.add_argument("--top_k_per_query", type=int, default=50)
    parser.add_argument("--max_subcorpus", type=int, default=300000)
    parser.add_argument("--max_query_df", type=int, default=1200)
    parser.add_argument("--max_passage_chars", type=int, default=900)
    parser.add_argument("--max_passage_terms", type=int, default=180)
    parser.add_argument("--max_updates_per_passage", type=int, default=8000)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--max_wiki_rows", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=200000)
    parser.add_argument("--include_train_gt", action="store_true")
    parser.add_argument("--query_scope", choices=["val", "train", "train_val"], default="train_val")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[subcorpus] building query index", flush=True)
    queries, term_to_queries = build_query_index(args)
    print(
        f"[subcorpus] queries={len(queries):,} terms={len(term_to_queries):,} "
        f"top_k_per_query={args.top_k_per_query} max_subcorpus={args.max_subcorpus:,}",
        flush=True,
    )
    selected_rows, best_scores, hit_counts, per_query_hits = coarse_recall(args, queries, term_to_queries)
    print(f"[subcorpus] selected_rows={len(selected_rows):,}; writing csv", flush=True)
    written = write_subcorpus(args, selected_rows, best_scores, hit_counts)
    manifest = {
        "wiki21m_file": args.wiki21m_file,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "output_csv": args.output_csv,
        "num_queries": len(queries),
        "num_terms": len(term_to_queries),
        "num_selected_rows": len(selected_rows),
        "num_written": written,
        "top_k_per_query": args.top_k_per_query,
        "max_subcorpus": args.max_subcorpus,
        "max_query_df": args.max_query_df,
        "max_wiki_rows": args.max_wiki_rows,
        "include_train_gt": args.include_train_gt,
        "query_scope": args.query_scope,
    }
    Path(args.manifest_file).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
