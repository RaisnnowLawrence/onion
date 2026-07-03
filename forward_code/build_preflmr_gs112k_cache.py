#!/usr/bin/env python3
"""Build OK-VQA knowledge cache by retrieving GS112K with local PreFLMR.

This script follows the NoteMR-style two-stage idea:
1. Offline retrieval: image + question -> top-k GS112K passages.
2. Save a per-question cache that onion can later consume without online corpus scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


FLMR_ROOT = Path("/data2/lizhengxue/WorkSpace/opensource/FLMR-main")
COLBERT_ROOT = FLMR_ROOT / "third_party" / "ColBERT"
DEFAULT_BERT_BASE = Path("/data2/lizhengxue/cuixingyu/data/cuixingyu/bishe/model/sup-simcse-bert-base-uncased")


def _detect_cuda_home() -> str | None:
    cuda_version = torch.version.cuda
    candidates: List[Path] = []
    if cuda_version:
        major_minor = ".".join(cuda_version.split(".")[:2])
        candidates.append(Path(f"/usr/local/cuda-{major_minor}"))
    candidates.extend([Path("/usr/local/cuda"), Path("/usr/local/cuda-12.4"), Path("/usr/local/cuda-12.8")])
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            return str(candidate)
    return None


def patch_runtime() -> None:
    """Patch small version mismatches between local FLMR/ColBERT and current env."""
    os.environ.setdefault("HF_HOME", "/data2/lizhengxue/WorkSpace/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data2/lizhengxue/WorkSpace/.cache/torch_extensions")
    cuda_home = os.environ.get("CUDA_HOME") or _detect_cuda_home()
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["LD_LIBRARY_PATH"] = str(Path(cuda_home) / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        cuda_bin = str(Path(cuda_home) / "bin")
    else:
        cuda_bin = ""
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [python_bin]
    if cuda_bin:
        path_parts.append(cuda_bin)
    path_parts.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(path_parts)
    if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    sys.path.insert(0, str(FLMR_ROOT))
    sys.path.insert(0, str(COLBERT_ROOT))

    transformers.AdamW = torch.optim.AdamW
    sys.modules["transformers"].AdamW = torch.optim.AdamW

    from transformers.models.bert.configuration_bert import BertConfig

    original_from_pretrained = BertConfig.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        cfg = original_from_pretrained(*args, **kwargs)
        if getattr(cfg, "_attn_implementation", None) is None:
            cfg._attn_implementation = "eager"
        return cfg

    BertConfig.from_pretrained = patched_from_pretrained


def read_gs112k(path: Path, max_corpus: int | None = None) -> Tuple[List[str], List[str]]:
    passage_ids: List[str] = []
    passages: List[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"GS112K file must contain a text column: {path}")
        id_field = "kid" if "kid" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            passage_ids.append(str(row.get(id_field, len(passage_ids))))
            passages.append(text)
            if max_corpus and len(passages) >= max_corpus:
                break
    return passage_ids, passages


def read_wiki21m(path: Path, max_corpus: int | None = None) -> Tuple[List[str], List[str]]:
    passage_ids: List[str] = []
    passages: List[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"Wiki21M TSV must contain id/text/title columns: {path}")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            title = (row.get("title") or "").strip()
            passage_ids.append(str(row.get("id") or len(passage_ids)))
            passages.append(f"{title}: {text}" if title else text)
            if max_corpus and len(passages) >= max_corpus:
                break
    return passage_ids, passages


class Wiki21MLookup:
    """Random-access lookup for retrieved wiki21m doc_idx without loading 13GB TSV."""

    def __init__(self, path: Path, offsets_path: Optional[Path] = None, max_corpus: int | None = None):
        self.path = path
        self.offsets_path = offsets_path or path.with_suffix(path.suffix + ".offsets")
        self.max_corpus = max_corpus
        self.offsets = self._load_or_build_offsets()

    def _load_or_build_offsets(self) -> List[int]:
        if self.offsets_path.exists():
            offsets: List[int] = []
            with self.offsets_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        offsets.append(int(line))
                        if self.max_corpus and len(offsets) >= self.max_corpus:
                            break
            return offsets

        self.offsets_path.parent.mkdir(parents=True, exist_ok=True)
        offsets = []
        with self.path.open("rb") as src, self.offsets_path.open("w", encoding="utf-8") as dst:
            _ = src.readline()  # header
            while True:
                pos = src.tell()
                line = src.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
                    dst.write(str(pos) + "\n")
                    if self.max_corpus and len(offsets) >= self.max_corpus:
                        break
        return offsets

    def get(self, doc_idx: int) -> dict:
        if doc_idx < 0 or doc_idx >= len(self.offsets):
            return {"id": str(doc_idx), "title": "", "text": ""}
        with self.path.open("rb") as f:
            f.seek(self.offsets[doc_idx])
            line = f.readline().decode("utf-8", errors="replace")
        row = next(csv.DictReader(["id\ttext\ttitle\n", line], delimiter="\t"))
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        return {"id": str(row.get("id") or doc_idx), "title": title, "text": text}


def read_corpus(args) -> Tuple[List[str], List[str]]:
    if args.corpus_format == "gs112k":
        return read_gs112k(Path(args.corpus_file), args.max_corpus)
    if args.corpus_format == "wiki21m":
        return read_wiki21m(Path(args.corpus_file), args.max_corpus)
    raise ValueError(f"Unsupported corpus_format={args.corpus_format}")


def load_okvqa_questions(path: Path, max_questions: int | None = None) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if max_questions:
        questions = questions[:max_questions]
    return questions


def load_aokvqa_questions(path: Path, max_questions: int | None = None) -> List[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if max_questions:
        questions = questions[:max_questions]
    return questions


def load_questions(args) -> List[dict]:
    if args.dataset_name == "okvqa":
        questions = load_okvqa_questions(Path(args.question_file), args.max_questions)
    elif args.dataset_name == "aokvqa":
        questions = load_aokvqa_questions(Path(args.question_file), args.max_questions)
    else:
        raise ValueError(f"Unsupported dataset_name={args.dataset_name}")
    if args.num_shards > 1:
        questions = [item for idx, item in enumerate(questions) if idx % args.num_shards == args.shard_id]
    return questions


def coco_val_image_path(coco_root: Path, image_id: int, dataset_name: str = "okvqa") -> Path:
    if dataset_name == "aokvqa":
        filenames = [
            f"COCO_val2014_{int(image_id):012d}.jpg",
            f"COCO_val2017_{int(image_id):012d}.jpg",
            f"{int(image_id):012d}.jpg",
        ]
        subdirs = ["val2017", "val2014", ""]
    else:
        filenames = [f"COCO_val2014_{int(image_id):012d}.jpg", f"{int(image_id):012d}.jpg"]
        subdirs = ["val2014", ""]
    candidates = [
        coco_root / subdir / filename if subdir else coco_root / filename
        for subdir in subdirs
        for filename in filenames
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find COCO val image for image_id={image_id}: {candidates}")


def batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_preflmr(args):
    checkpoint = Path(args.preflmr_checkpoint)
    query_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, subfolder="query_tokenizer", trust_remote_code=True
    )
    context_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, subfolder="context_tokenizer", trust_remote_code=True
    )
    transformers.FLMRQueryEncoderTokenizer = query_tokenizer.__class__
    transformers.FLMRContextEncoderTokenizer = context_tokenizer.__class__

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.transformer_mapping_config_base = args.bert_base
    config.vision_model_version = args.image_processor

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        config=config,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
        low_cpu_mem_usage=True,
    )
    model.config.vision_model_version = args.image_processor
    if hasattr(model, "vision_model_version"):
        model.vision_model_version = args.image_processor
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)

    image_processor = AutoImageProcessor.from_pretrained(args.image_processor)
    return model, query_tokenizer, image_processor


def build_or_load_index(args, passages: List[str], model) -> str:
    from flmr import index_custom_collection

    return index_custom_collection(
        custom_collection=passages,
        model=model,
        index_root_path=args.index_root,
        index_experiment_name=args.index_experiment,
        index_name=args.index_name,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        overwrite=args.overwrite_index,
        use_gpu=args.device != "cpu",
        indexing_batch_size=args.index_batch_size,
        model_temp_folder=str(Path(args.output_dir) / "tmp_preflmr_model"),
        nranks=1,
    )


def build_lookup(args, passage_ids: List[str], passages: List[str]):
    if args.corpus_format == "wiki21m" and args.use_lazy_wiki_lookup:
        return Wiki21MLookup(Path(args.corpus_file), Path(args.wiki_offsets_file) if args.wiki_offsets_file else None, args.max_corpus)
    return None


def _lookup_passage(args, lookup, passage_ids: List[str], passages: List[str], doc_idx: int) -> dict:
    if lookup is not None:
        item = lookup.get(doc_idx)
        return {
            "id": item["id"],
            "title": item.get("title", ""),
            "text": item.get("text", ""),
        }
    return {
        "id": passage_ids[doc_idx],
        "title": f"{args.corpus_format}:{passage_ids[doc_idx]}",
        "text": passages[doc_idx],
    }


def retrieve(args, questions: List[dict], passage_ids: List[str], passages: List[str], model, query_tokenizer, image_processor):
    from flmr import create_searcher, search_custom_collection

    searcher = create_searcher(
        index_root_path=args.index_root,
        index_experiment_name=args.index_experiment,
        index_name=args.index_name,
        nbits=args.nbits,
        use_gpu=args.device != "cpu",
    )

    results_json: List[dict] = []
    results_jsonl: List[dict] = []
    lookup = build_lookup(args, passage_ids, passages)

    query_counter = 0
    for batch in tqdm(list(batched(questions, args.query_batch_size)), desc="PreFLMR search"):
        query_texts = [f"{args.query_instruction} : {item['question']}" for item in batch]
        query_ids = list(range(query_counter, query_counter + len(batch)))
        query_counter += len(batch)
        image_paths = [
            coco_val_image_path(Path(args.coco_root), int(item["image_id"]), args.dataset_name)
            for item in batch
        ]

        images = [Image.open(path).convert("RGB") for path in image_paths]
        enc = query_tokenizer(query_texts, padding=True, truncation=True, return_tensors="pt")
        pixels = image_processor(images, return_tensors="pt")["pixel_values"]
        model_inputs = {
            "input_ids": enc["input_ids"].to(args.device),
            "attention_mask": enc["attention_mask"].to(args.device),
            "pixel_values": pixels.to(args.device),
        }
        with torch.inference_mode():
            query_embeddings = model.query(**model_inputs).late_interaction_output.detach().cpu()

        query_map: Dict[int, str] = dict(zip(query_ids, query_texts))
        ranking = search_custom_collection(
            searcher=searcher,
            queries=query_map,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=args.top_k,
            remove_zero_tensors=True,
            centroid_search_batch_size=args.centroid_search_batch_size,
        )
        ranking_dict = ranking.todict()

        for local_qid, item in zip(query_ids, batch):
            qid = str(item["question_id"])
            ctxs = []
            for rank, doc in enumerate(ranking_dict[local_qid], start=1):
                doc_idx, _, score = doc
                passage = _lookup_passage(args, lookup, passage_ids, passages, doc_idx)
                ctxs.append(
                    {
                        "id": passage["id"],
                        "rank": rank,
                        "score": float(score),
                        "title": passage.get("title", ""),
                        "text": passage["text"],
                    }
                )
            note_mr_record = {
                "question_id": qid,
                "image_id": int(item["image_id"]),
                "question": item["question"],
                "retrieval_query": query_map[local_qid],
                "ctxs": ctxs,
            }
            onion_record = {
                "question_id": qid,
                "image_id": int(item["image_id"]),
                "question": item["question"],
                "retrieval_query": query_map[local_qid],
                "selected_knowledge": [
                    {
                        "source": args.knowledge_source_name,
                        "title": ctx.get("title") or f"{args.corpus_format}:{ctx['id']}",
                        "text": ctx["text"],
                        "score": ctx["score"],
                        "rank": ctx["rank"],
                    }
                    for ctx in ctxs
                ],
            }
            results_json.append(note_mr_record)
            results_jsonl.append(onion_record)

    return results_json, results_jsonl


def write_outputs(args, note_mr_records: List[dict], onion_records: List[dict]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    note_mr_path = output_dir / args.note_mr_output_name
    onion_path = output_dir / args.onion_output_name
    note_mr_path.write_text(json.dumps(note_mr_records, ensure_ascii=False, indent=2), encoding="utf-8")
    with onion_path.open("w", encoding="utf-8") as f:
        for record in onion_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "preflmr_checkpoint": args.preflmr_checkpoint,
        "corpus_file": args.corpus_file,
        "corpus_format": args.corpus_format,
        "question_file": args.question_file,
        "dataset_name": args.dataset_name,
        "coco_root": args.coco_root,
        "top_k": args.top_k,
        "query_instruction": args.query_instruction,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_records": len(note_mr_records),
        "note_mr_json": str(note_mr_path),
        "onion_jsonl": str(onion_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflmr_checkpoint", default="/data2/lizhengxue/WorkSpace/PreTrainModel/PreFLMR/PreFLMR_ViT-B")
    parser.add_argument("--bert_base", default=str(DEFAULT_BERT_BASE))
    parser.add_argument("--image_processor", default="/data2/lizhengxue/WorkSpace/PreTrainModel/OpenAI/clip-vit-base-patch32")
    parser.add_argument("--gs112k_corpus", default="/data2/lizhengxue/datasets/gs112k/okvqa_train_clean_corpus.csv")
    parser.add_argument("--corpus_file", default="")
    parser.add_argument("--corpus_format", choices=["gs112k", "wiki21m"], default="gs112k")
    parser.add_argument("--knowledge_source_name", default="")
    parser.add_argument("--dataset_name", choices=["okvqa", "aokvqa"], default="okvqa")
    parser.add_argument("--question_file", default="/data2/lizhengxue/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--coco_root", default="/data2/lizhengxue/datasets/coco14")
    parser.add_argument("--output_dir", default="/data2/lizhengxue/WorkSpace/onion_output/okvqa/preflmr_gs112k_cache")
    parser.add_argument("--index_root", default="/data2/lizhengxue/WorkSpace/onion_output/preflmr_indexes")
    parser.add_argument("--index_experiment", default="gs112k")
    parser.add_argument("--index_name", default="preflmr_vitb_gs112k_train_clean")
    parser.add_argument("--note_mr_output_name", default="preflmr_gs112k_retriever.json")
    parser.add_argument("--onion_output_name", default="preflmr_gs112k_knowledge_cache.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--query_instruction",
        default="Using the image, retrieve background knowledge that helps answer the question",
    )
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--doc_maxlen", type=int, default=512)
    parser.add_argument("--index_batch_size", type=int, default=64)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--centroid_search_batch_size", type=int, default=None)
    parser.add_argument("--max_corpus", type=int, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--skip_index_build", action="store_true")
    parser.add_argument("--build_index_only", action="store_true")
    parser.add_argument("--use_lazy_wiki_lookup", action="store_true")
    parser.add_argument("--wiki_offsets_file", default="")
    parser.add_argument("--overwrite_index", action="store_true")
    args = parser.parse_args()
    if not args.corpus_file:
        args.corpus_file = args.gs112k_corpus
    if not args.knowledge_source_name:
        args.knowledge_source_name = f"preflmr_{args.corpus_format}"
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Invalid shard_id={args.shard_id}, num_shards={args.num_shards}")
    return args


def main() -> None:
    args = parse_args()
    patch_runtime()
    if args.skip_index_build and args.corpus_format == "wiki21m" and args.use_lazy_wiki_lookup:
        passage_ids, passages = [], []
    else:
        passage_ids, passages = read_corpus(args)
    questions = [] if args.build_index_only else load_questions(args)
    print(
        f"Loaded {len(passages) if passages else 'lazy'} {args.corpus_format} passages "
        f"and {len(questions)} {args.dataset_name} questions for shard {args.shard_id}/{args.num_shards}."
    )
    model, query_tokenizer, image_processor = load_preflmr(args)
    if args.skip_index_build:
        print(f"Skip index build; using existing index: {args.index_experiment}/{args.index_name}")
    else:
        index_path = build_or_load_index(args, passages, model)
        print(f"Using index: {index_path}")
    if args.build_index_only:
        print("Build-index-only mode finished.")
        return
    note_mr_records, onion_records = retrieve(
        args, questions, passage_ids, passages, model, query_tokenizer, image_processor
    )
    write_outputs(args, note_mr_records, onion_records)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
