"""Dataset registry, path resolution, and runtime policies for Onion."""

import os

from aokvqa_utils import (
    aokvqa_dataset,
    gqa_dataset,
    hallusionbench_dataset,
    infoseek_dataset,
    mme_dataset,
    mme_realworld_dataset,
    mmstar_dataset,
    okvqa_dataset,
    pope_dataset,
    textvqa_dataset,
    vqav2_dataset,
)


DATASET_FACTORIES = {
    "aokvqa": aokvqa_dataset,
    "okvqa": okvqa_dataset,
    "vqav2": vqav2_dataset,
    "gqa": gqa_dataset,
    "textvqa": textvqa_dataset,
    "infoseek": infoseek_dataset,
    "pope": pope_dataset,
    "mme": mme_dataset,
    "mme_realworld": mme_realworld_dataset,
    "hallusionbench": hallusionbench_dataset,
    "mmstar": mmstar_dataset,
}

YES_NO_PROMPT_DATASETS = frozenset({"pope", "mme", "hallusionbench"})
YES_NO_SCORING_DATASETS = frozenset({"pope", "mme"})
SCENE_GRAPH_OPTIONAL_DATASETS = frozenset({"mme"})


def get_dataset_name(args):
    """Return a normalized-enough dataset name from an argparse namespace."""
    return str(getattr(args, "dataset_name", "") or "").strip().lower()


def is_dataset(args, *names):
    """Test the configured dataset without repeating defensive ``getattr`` calls."""
    return get_dataset_name(args) in names


def uses_yes_no_prompt(args):
    return get_dataset_name(args) in YES_NO_PROMPT_DATASETS


def uses_yes_no_scoring(args):
    return get_dataset_name(args) in YES_NO_SCORING_DATASETS


def can_skip_scene_graph(args):
    return get_dataset_name(args) in SCENE_GRAPH_OPTIONAL_DATASETS


def image_key_from_sample_key(key, args, image_dict=None):
    """Resolve ``image_id<->question_id`` keys, including legacy FVQA mappings."""
    if is_dataset(args, "fvqa"):
        if image_dict is None:
            raise ValueError("FVQA image-key resolution requires image_dict")
        return image_dict[key]

    image_key = str(key).split("<->", 1)[0]
    return int(image_key) if image_key.isdigit() else image_key


def normalize_dataset_name(name):
    name = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "a_okvqa": "aokvqa",
        "a_ok_vqa": "aokvqa",
        "ok_vqa": "okvqa",
        "vqa_v2": "vqav2",
        "vqa2": "vqav2",
        "text_vqa": "textvqa",
        "info_seek": "infoseek",
        "mme_real_world": "mme_realworld",
        "mme_realworld": "mme_realworld",
        "mme_rw": "mme_realworld",
        "mm_star": "mmstar",
    }
    return aliases.get(name, name)


def resolve_dataset_paths(args):
    args.dataset_name = normalize_dataset_name(args.dataset_name)
    dataset_dirs = {
        "aokvqa": "aokvqa",
        "okvqa": "okvqa",
        "vqav2": "vqav2",
        "gqa": "gqa",
        "textvqa": "textvqa",
        "infoseek": "infoseek",
        "pope": "pope",
        "mme": "mme",
        "mme_realworld": "mme-realworld",
        "hallusionbench": "hallusionbench",
        "mmstar": "mmstar",
    }
    default_coco_path = "/data2/lizhengxue/datasets/aokvqa"
    if args.dataset_name in dataset_dirs and (
        not args.coco_path or os.path.abspath(args.coco_path) == os.path.abspath(default_coco_path)
    ):
        candidate = os.path.join(args.dataset_root, dataset_dirs[args.dataset_name])
        if args.dataset_name == "mme_realworld" and not os.path.isdir(candidate):
            for dirname in ("mme_realworld", "MME-RealWorld", "mme"):
                alt = os.path.join(args.dataset_root, dirname)
                if os.path.isdir(alt):
                    candidate = alt
                    break
        args.coco_path = candidate

    default_raw_image_dir = "/data2/lizhengxue/datasets/coco17"
    if not args.raw_image_dir or os.path.abspath(args.raw_image_dir) == os.path.abspath(default_raw_image_dir):
        if args.dataset_name in ("aokvqa",):
            args.raw_image_dir = os.path.join(args.dataset_root, "coco17")
        elif args.dataset_name in ("okvqa", "vqav2", "pope"):
            args.raw_image_dir = os.path.join(args.dataset_root, "coco14")
        elif args.dataset_name == "gqa":
            args.raw_image_dir = os.path.join(args.dataset_root, "visualgenome")
        elif args.dataset_name in ("textvqa", "infoseek", "mme_realworld", "hallusionbench", "mmstar"):
            args.raw_image_dir = args.coco_path
        elif args.dataset_name == "mme":
            args.raw_image_dir = args.coco_path

    if args.dataset_name == "mme_realworld" and not os.path.isdir(args.coco_path):
        print(f"[dataset] MME-RealWorld path not found: {args.coco_path}; pass --coco_path when available.")
    if args.dataset_name == "infoseek" and not os.path.isdir(args.coco_path):
        print(f"[dataset] InfoSeek path not found: {args.coco_path}; pass --coco_path when available.")
    return args


def build_dataset(args):
    """Construct the configured dataset, preserving A-OKVQA as the fallback."""
    factory = DATASET_FACTORIES.get(get_dataset_name(args), aokvqa_dataset)
    return factory(args)
