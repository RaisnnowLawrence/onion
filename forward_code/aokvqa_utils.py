import os
import csv
import argparse
import numpy as np
import multiprocessing
import json
import time
import torch
import random
import openai
from tqdm import tqdm
from transformers import GPT2Tokenizer
import pdb
import pickle
import glob
import io
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import datetime


def bounding_box_matching(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
        return 0
    intersection = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    if union <= 0:
        return 0
    return intersection / union


class aokvqa_dataset:

    def __init__(self, args):
        # 加载参数
        self.args = args
        # 加载数据集
        self.load_dataset(args)
        self.load_similarity()

    def load_dataset(self, args):
        """
        加载数据集和相关缓存数据。
        根据参数配置加载验证集/测试集、训练集、图像描述、场景图等数据。

        Args:
            args: 包含配置参数的对象,包括：
                - test_only: 是否仅测试模式
                - raw_image_dir: 原始图像目录
                - coco_path: COCO数据集路径
                - start, end: 数据切片起始和结束比例（用于分布式训练）
                - caption_type: 描述类型（如'vinvl_ocr')
                - sg_path: 场景图数据路径
                - concept_caption_path: 概念描述路径
                - choice_only: 是否仅加载选择题选项
        """

        # 根据测试模式确定数据集划分名称
        split = args.split_name
        
        # 设置原始图像目录路径（如 .../val2017 或 .../test2017）
        self.raw_image_dir = os.path.join(self.args.raw_image_dir, "%s2017" % split)

        val_anno_file = f'{args.coco_path}/aokvqa_v1p0_{split}.json'

        # 加载验证/测试集的标注数据
        # 返回元组: (_, 答案字典, 问题字典, 原理字典, 选项字典)
        _, self.answer_dict, self.question_dict, self.rationale_dict, self.choices_dict = \
            self.load_anno(
                None,  # 不加载额外的标注文件
                val_anno_file,  # AOK-VQA问题文件
                val_anno_file,  # 同上（可能为占位）
                choice_only=args.choice_only  # 是否仅加载选择题
            )
        
        # 获取验证集的所有问题ID作为键
        self.val_keys = list(self.question_dict.keys())
        self.direct_answer_eval_keys = self.load_direct_answer_eval_keys(val_anno_file)

        ## 加载缓存的文本数据（COCO图像描述和标签）
        self.inputtext_dict = self.load_cachetext()
        
        # 加载训练集的上下文数据（用于训练或上下文学习）
        self.traincontext_caption_dict, self.traincontext_answer_dict, \
        self.traincontext_question_dict, self.traincontext_rationale_dict, \
        self.traincontext_choices_dict = \
            self.load_anno(
                '%s/captions_train2017.json' % args.coco_annotation_path,  # COCO训练集描述
                '%s/aokvqa_v1p0_train.json' % args.coco_path,   # AOK-VQA训练集问题
                '%s/aokvqa_v1p0_train.json' % args.coco_path,   # 同上（可能为占位）
                choice_only=args.choice_only
            )
        
        # 设置交互式训练数据（此处与普通训练数据相同）
        self.traincontext_interactive_answer_dict = self.traincontext_answer_dict
        self.traincontext_interactive_question_dict = self.traincontext_question_dict
        
        # 获取训练集的所有问题ID
        self.train_keys = list(self.traincontext_answer_dict.keys())
        # 交互式训练键与普通训练键相同
        self.train_interactive_keys = self.train_keys
        
        # 设置场景图相关目录路径
        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")           # 场景图主目录
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")      # 属性目录
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)  # 概念描述目录

        self.train_ocr_text = {}
        self.val_ocr_text = {}
        if getattr(args, "use_ocr_context", False) or args.caption_type == "vinvl_ocr":
            train_ocr = args.ocr_train_file or os.path.join(args.sg_path, "coco17_ocr_train.json")
            val_ocr = args.ocr_val_file or os.path.join(args.sg_path, f"coco17_ocr_{split}.json")
            self.load_ocr(train_ocr, val_ocr, self.sg_attr_dir, thres=args.ocr_conf_threshold)

    def load_direct_answer_eval_keys(self, anno_file):
        """官方A-OKVQA DA评测只统计 difficult_direct_answer 为 False 的样本。"""
        eval_keys = set()
        try:
            annotations = json.load(open(anno_file, 'r'))
        except FileNotFoundError:
            return eval_keys
        for sample in annotations:
            if sample.get('difficult_direct_answer') is False:
                key = str(sample['image_id']) + '<->' + str(sample['question_id'])
                eval_keys.add(key)
        return eval_keys

    def load_ocr(self, train_ocr, val_ocr, sg_path, thres=0.2):
        if not os.path.isfile(train_ocr) or not os.path.isfile(val_ocr):
            print(f"[ocr] OCR文件不存在，跳过OCR上下文: train={train_ocr}, val={val_ocr}")
            self.train_ocr_text = {}
            self.val_ocr_text = {}
            return

        def _load_one(ocr_file):
            ocr_dict = json.load(open(ocr_file))
            output = {}
            for key in ocr_dict:
                tmp_ocr_list = ocr_dict[key]
                image_id = int(key.split("_")[-1])
                ocr_text = {}
                if len(tmp_ocr_list) > 0:
                    sg_file = os.path.join(sg_path, f"{str(image_id).zfill(12)}.json")
                    if not os.path.isfile(sg_file):
                        sg_file = os.path.join(sg_path, f"{image_id}.json")
                    if not os.path.isfile(sg_file):
                        output[image_id] = ocr_text
                        continue

                    obj_list = json.load(open(sg_file))
                    for tmp_ocr in tmp_ocr_list:
                        box = tmp_ocr.get("box")
                        text = str(tmp_ocr.get("text", "")).strip()
                        conf = float(tmp_ocr.get("conf", 0))
                        if not box or not text or conf <= thres:
                            continue

                        if isinstance(box[0], list):
                            box = [box[0][0], box[0][1], box[1][0], box[2][1]]
                        max_match_val = -1
                        max_match_obj = ""
                        for obj in obj_list[0]:
                            match_val = bounding_box_matching(box, obj["rect"])
                            if match_val > max_match_val:
                                max_match_obj = obj["class"]
                                max_match_val = match_val
                        if max_match_obj:
                            ocr_text[max_match_obj] = f"Text {text} is on the {max_match_obj}."
                output[image_id] = ocr_text
            return output

        self.train_ocr_text = _load_one(train_ocr)
        self.val_ocr_text = _load_one(val_ocr)
        print(f"[ocr] 已加载OCR上下文: train={len(self.train_ocr_text)}, val={len(self.val_ocr_text)}")
    
    def load_anno(self, coco_caption_file, answer_anno_file, question_anno_file, choice_only=False):
        """
        加载并处理标注数据,包括COCO图像描述、AOK-VQA问题、答案、选项和原理解释。
        
        此方法将原始JSON标注文件转换为易于查询的字典格式,键为"image_id<->question_id"的组合。
        
        Args:
            coco_caption_file (str): COCO图像描述文件的路径,为None则不加载
            answer_anno_file (str): AOK-VQA答案标注文件路径,包含答案、选项和原理
            question_anno_file (str): AOK-VQA问题标注文件路径,包含问题文本
            choice_only (bool): 如果为True,仅加载选择题的正确答案索引；否则加载直接答案
        
        Returns:
            tuple: 包含五个字典的元组,格式为:
                (caption_dict, answer_dict, question_dict, rationales_dict, choices_dict)
                其中每个字典的键都是"image_id<->question_id"格式的字符串
        """
        # 1. 加载原始JSON文件
        # 如果提供了COCO描述文件,则加载
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, 'r'))
            # 如果文件是字典格式且有'annotations'键,则提取注释列表
            if type(coco_caption) == type({}): 
                coco_caption = coco_caption['annotations']
        
        # 加载AOK-VQA的答案标注文件
        answer_anno = json.load(open(answer_anno_file, 'r'))
        # 加载AOK-VQA的问题标注文件
        question_anno = json.load(open(question_anno_file, 'r'))

        # 2. 构建COCO图像描述字典
        # 格式: {image_id: [caption1, caption2, ...]}
        caption_dict = {}
        if coco_caption_file is not None:
            for sample in coco_caption:
                image_id = sample['image_id']
                if image_id not in caption_dict:
                    # 为每个图像ID创建描述列表
                    caption_dict[image_id] = [sample['caption']]
                else:
                    # 追加额外的描述（每张图片通常有多个描述）
                    caption_dict[image_id].append(sample['caption'])

        # 3. 构建答案字典
        # 格式: {"image_id<->question_id": answer_data}
        answer_dict = {}
        for sample in answer_anno:
            # 创建复合键：结合图像ID和问题ID,确保唯一性
            key = str(sample['image_id']) + '<->' + str(sample['question_id'])
            
            if key not in answer_dict:  # 避免重复
                if choice_only:
                    # 选择题模式：存储正确答案的索引
                    if 'correct_choice_idx' in sample:
                        answer_dict[key] = sample["correct_choice_idx"]
                    else:
                        # 如果没有正确答案索引,默认为0（第一个选项）
                        answer_dict[key] = 0
                else:
                    # 非选择题模式：存储直接答案列表
                    if 'direct_answers' in sample:
                        answer_dict[key] = sample["direct_answers"]
                    else:
                        # 如果没有直接答案,存储空列表
                        answer_dict[key] = [""]

        # 4. 构建问题字典
        # 格式: {"image_id<->question_id": question_text}
        question_dict = {}
        for sample in question_anno:
            key = str(sample['image_id']) + '<->' + str(sample['question_id'])
            if key not in question_dict:
                question_dict[key] = sample['question']

        # 5. 构建原理解释字典
        # 格式: {"image_id<->question_id": [rationale1, rationale2, ...]}
        rationales_dict = {}
        for sample in answer_anno:
            key = str(sample['image_id']) + '<->' + str(sample['question_id'])
            if key not in rationales_dict:
                if 'rationales' in sample:
                    # 存储原理解释列表（通常有多个）
                    rationales_dict[key] = sample['rationales']
                else:
                    # 如果没有原理解释,存储空字符串
                    rationales_dict[key] = ""

        # 6. 构建选项字典
        # 格式: {"image_id<->question_id": [choice1, choice2, ...]}
        choices_dict = {}
        for sample in answer_anno:
            key = str(sample['image_id']) + '<->' + str(sample['question_id'])
            # 注意：这里没有检查key是否已存在,因为每个问题都应该有选项
            choices_dict[key] = sample['choices']

        # 返回所有构建的字典
        # 描述字典,答案字典,问题字典,原理字典,选项字典
        return caption_dict, answer_dict, question_dict, rationales_dict, choices_dict

    def load_cachetext(self):
        read_tsv = csv.reader(open(self.args.valcaption_file, 'r'), delimiter="\t")
        caption_dict = {}
        if 'tag' in self.args.caption_type:
            tags_dict = self.load_tags()
        if self.args.caption_type == 'vinvl_tag':
            for row in read_tsv:
                tag_text = tags_dict.get(int(row[0]), "")
                suffix = ('. ' + tag_text) if tag_text else ""
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [
                        row[1].split('caption": "')[1].split('", "conf"')[0] + suffix]
                else:
                    caption_dict[int(row[0])].append(
                        row[1].split('caption": "')[1].split('", "conf"')[0] + suffix)
        else:
            for row in read_tsv:
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [row[1].split('caption": "')[1].split('", "conf"')[0]]
                else:
                    caption_dict[int(row[0])].append(row[1].split('caption": "')[1].split('", "conf"')[0])
        return caption_dict

    def load_tags(self):
        tags_dict = {}
        for split_name in ("test", "val", "train"):
            tagging_pred_file = os.path.join(self.args.tag_path, f"{split_name}.score.json.tsv")
            if not os.path.isfile(tagging_pred_file):
                print(f"[tags] tag file missing, skip: {tagging_pred_file}")
                continue
            read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
            for row in read_tsv:
                image_id, tags = int(row[0]), json.loads(row[1])
                tag_str = ', '.join([x['class'] for x in tags])
                tags_dict[image_id] = tag_str
        return tags_dict
    
    def load_similarity(self):
        split = self.args.split_name
        val_idx = json.load(open('%s/aokvqa_qa_line2sample_idx_%s2017.json' % (self.args.similarity_path, split), 'r'))
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if self.args.similarity_metric == 'question':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load('%s/coco_clip_vitb16_%s2017_aokvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
        elif self.args.similarity_metric == 'imagequestion':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load('%s/coco_clip_vitb16_%s2017_aokvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
            self.image_train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy' % self.args.similarity_path)
            self.image_val_feature = np.load(
                '%s/coco_clip_vitb16_%s2017_aokvqa_convertedidx_image.npy' % (self.args.similarity_path, split))
    
    # 根据图片id加载图片
    def find_image(self, img_key):
        return Image.open(self.find_image_path(img_key)).convert("RGB")

    # 获取图片路径
    def find_image_path(self, img_key):
        split = self.args.split_name
        filename = "COCO_%s2014_%012d.jpg" % (split, img_key)
        candidates = [
            os.path.join(self.args.raw_image_dir, f"{split}2014", filename),
            os.path.join(self.args.raw_image_dir, f"{split}2017", filename),
            os.path.join(self.args.raw_image_dir, filename),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]


class okvqa_dataset(aokvqa_dataset):
    """OK-VQA loader using the same runtime interface as aokvqa_dataset."""

    def load_dataset(self, args):
        if args.choice_only:
            raise ValueError("OK-VQA is open-ended VQA in this project; --choice_only is not supported.")

        split = args.split_name
        if split not in ("train", "val"):
            raise ValueError(f"OK-VQA only has train/val annotations here, got split_name={split}")

        _, self.answer_dict, self.question_dict = self.load_ok_anno(
            None,
            f"{args.coco_path}/mscoco_{split}2014_annotations.json",
            f"{args.coco_path}/OpenEnded_mscoco_{split}2014_questions.json",
        )
        self.rationale_dict = {key: "" for key in self.question_dict}
        self.choices_dict = {}
        self.val_keys = list(self.question_dict.keys())
        self.direct_answer_eval_keys = set(self.val_keys)

        self.inputtext_dict = self.load_cachetext()

        _, self.traincontext_answer_dict, self.traincontext_question_dict = self.load_ok_anno(
            None,
            f"{args.coco_path}/mscoco_train2014_annotations.json",
            f"{args.coco_path}/OpenEnded_mscoco_train2014_questions.json",
        )
        self.traincontext_rationale_dict = {key: "" for key in self.traincontext_question_dict}
        self.traincontext_choices_dict = {}
        train_image_ids = {int(key.split("<->")[0]) for key in self.traincontext_question_dict}
        self.traincontext_caption_dict = {image_id: [""] for image_id in train_image_ids}
        self.traincontext_interactive_answer_dict = self.traincontext_answer_dict
        self.traincontext_interactive_question_dict = self.traincontext_question_dict

        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.train_interactive_keys = self.train_keys

        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)

        self.train_ocr_text = {}
        self.val_ocr_text = {}

    def load_ok_anno(self, coco_caption_file, answer_anno_file, question_anno_file):
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, "r"))
            if isinstance(coco_caption, dict):
                coco_caption = coco_caption["annotations"]
        answer_anno = json.load(open(answer_anno_file, "r"))
        question_anno = json.load(open(question_anno_file, "r"))

        caption_dict = {}
        if coco_caption_file is not None:
            for sample in coco_caption:
                caption_dict.setdefault(sample["image_id"], []).append(sample["caption"])

        answer_dict = {}
        for sample in answer_anno["annotations"]:
            key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            answer_dict[key] = [ans["answer"] for ans in sample.get("answers", [])]

        question_dict = {}
        for sample in question_anno["questions"]:
            key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            question_dict[key] = sample["question"]

        return caption_dict, answer_dict, question_dict

    def load_similarity(self):
        split = self.args.split_name
        val_idx = json.load(open(f"{self.args.similarity_path}/okvqa_qa_line2sample_idx_{split}2014.json", "r"))
        self.valkey2idx = {val_idx[ii]: int(ii) for ii in val_idx}

        self.train_feature = np.load(f"{self.args.similarity_path}/coco_clip_vitb16_train2014_okvqa_question.npy")
        self.val_feature = np.load(f"{self.args.similarity_path}/coco_clip_vitb16_{split}2014_okvqa_question.npy")
        self.train_idx = json.load(open(f"{self.args.similarity_path}/okvqa_qa_line2sample_idx_train2014.json", "r"))

        if self.args.similarity_metric == "imagequestion":
            self.image_train_feature = np.load(
                f"{self.args.similarity_path}/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy"
            )
            self.image_val_feature = np.load(
                f"{self.args.similarity_path}/coco_clip_vitb16_{split}2014_okvqa_convertedidx_image.npy"
            )

    def find_image(self, img_key):
        return Image.open(self.find_image_path(img_key)).convert("RGB")

    def find_image_path(self, img_key):
        split = self.args.split_name
        if split == "train":
            coco14_path = os.path.join(
                self.args.raw_image_dir,
                "train2014",
                "COCO_train2014_%012d.jpg" % img_key,
            )
            if os.path.isfile(coco14_path):
                return coco14_path
            return os.path.join(
                self.args.raw_image_dir,
                "train2014_image",
                "train2014",
                "COCO_train2014_%012d.jpg" % img_key,
            )
        return os.path.join(
            self.args.raw_image_dir,
            "val2014",
            "COCO_val2014_%012d.jpg" % img_key,
        )


class pope_dataset(aokvqa_dataset):
    """POPE yes/no hallucination benchmark loader.

    POPE annotation files are JSONL and use COCO val2014 images. `split_name=all`
    concatenates random, popular, and adversarial subsets while preserving the
    subset name in the question id.
    """

    def load_dataset(self, args):
        if args.choice_only:
            raise ValueError("POPE is a yes/no open-ended benchmark; --choice_only is not supported.")

        self.raw_image_dir = args.raw_image_dir
        self.image_filename_dict = {}

        subsets = ["random", "popular", "adversarial"] if args.split_name == "all" else [args.split_name]
        self.answer_dict = {}
        self.question_dict = {}
        self.rationale_dict = {}
        self.choices_dict = {}

        for subset in subsets:
            anno_file = os.path.join(args.coco_path, f"coco_pope_{subset}.json")
            with open(anno_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    image_id = int(sample["image"].split("_")[-1].split(".")[0])
                    key = f"{image_id}<->{subset}_{sample['question_id']}"
                    self.answer_dict[key] = [str(sample["label"]).lower()]
                    self.question_dict[key] = sample["text"]
                    self.rationale_dict[key] = ""
                    self.choices_dict[key] = ["yes", "no"]
                    self.image_filename_dict[image_id] = sample["image"]

        self.val_keys = list(self.question_dict.keys())
        self.direct_answer_eval_keys = set(self.val_keys)

        self.inputtext_dict = self.load_cachetext()

        # Reuse A-OKVQA train annotations as few-shot context. POPE itself has no train split.
        self.traincontext_caption_dict, self.traincontext_answer_dict, self.traincontext_question_dict, \
        self.traincontext_rationale_dict, self.traincontext_choices_dict = \
            self.load_anno(
                '%s/captions_train2017.json' % args.coco_annotation_path,
                '%s/aokvqa_v1p0_train.json' % args.aokvqa_context_path,
                '%s/aokvqa_v1p0_train.json' % args.aokvqa_context_path,
                choice_only=False,
            )
        self.traincontext_interactive_answer_dict = self.traincontext_answer_dict
        self.traincontext_interactive_question_dict = self.traincontext_question_dict
        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.train_interactive_keys = self.train_keys

        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)

        self.train_ocr_text = {}
        self.val_ocr_text = {}

    def load_similarity(self):
        self.valkey2idx = {}

    def find_image(self, img_key):
        return Image.open(self.find_image_path(img_key)).convert("RGB")

    def find_image_path(self, img_key):
        filename = self.image_filename_dict.get(int(img_key), "COCO_val2014_%012d.jpg" % int(img_key))
        candidates = [
            os.path.join(self.args.raw_image_dir, "val2014", filename),
            os.path.join(self.args.raw_image_dir, filename),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]


def load_mme_parquet_records(args, save_images=False):
    """Load MME parquet records and optionally materialize embedded images."""
    manifest_file = getattr(args, "mme_manifest_file", "")
    if manifest_file and os.path.isfile(manifest_file):
        records = []
        with open(manifest_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append({
                    "image_idx": int(rec["image_idx"]),
                    "question_id": str(rec["question_id"]),
                    "key": rec.get("key") or f"{int(rec['image_idx'])}<->{rec['question_id']}",
                    "question": str(rec.get("question", "")),
                    "answer": str(rec.get("answer", "")).lower(),
                    "category": str(rec.get("category", "")),
                    "image_path": rec["image_path"],
                })
        return records

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("MME parquet loading requires pandas.") from exc

    parquet_files = sorted(glob.glob(os.path.join(args.coco_path, f"{args.split_name}-*.parquet")))
    if not parquet_files and args.split_name == "test":
        parquet_files = sorted(glob.glob(os.path.join(args.coco_path, "test-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No MME parquet files found under {args.coco_path} for split={args.split_name}")

    image_cache_dir = os.path.join(args.cache_path, "mme_images")
    if save_images:
        os.makedirs(image_cache_dir, exist_ok=True)

    records = []
    row_offset = 0
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
        except ImportError as exc:
            raise ImportError(
                "MME parquet loading requires pyarrow or fastparquet in the active Python environment."
            ) from exc
        for local_idx, row in df.iterrows():
            image_idx = row_offset + int(local_idx)
            question_id = str(row.get("question_id", image_idx))
            image_path = os.path.join(image_cache_dir, f"{image_idx:06d}.jpg")
            if save_images and not os.path.isfile(image_path):
                image_obj = row["image"]
                if isinstance(image_obj, Image.Image):
                    image = image_obj.convert("RGB")
                elif isinstance(image_obj, dict):
                    image_bytes = image_obj.get("bytes")
                    if image_bytes is None and image_obj.get("path"):
                        image = Image.open(image_obj["path"]).convert("RGB")
                    else:
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                elif isinstance(image_obj, (bytes, bytearray)):
                    image = Image.open(io.BytesIO(image_obj)).convert("RGB")
                else:
                    raise TypeError(f"Unsupported MME image payload type: {type(image_obj)}")
                image.save(image_path)
            records.append({
                "image_idx": image_idx,
                "question_id": question_id,
                "key": f"{image_idx}<->{question_id}",
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")).lower(),
                "category": str(row.get("category", "")),
                "image_path": image_path,
            })
        row_offset += len(df)
    return records


def load_mme_answer_annotations(args):
    records = load_mme_parquet_records(args, save_images=False)
    answer_by_key = {rec["key"]: [rec["answer"]] for rec in records}
    return answer_by_key, set(answer_by_key)


class mme_dataset(aokvqa_dataset):
    """MME yes/no benchmark loader backed by local parquet files."""

    def load_dataset(self, args):
        if args.choice_only:
            raise ValueError("MME is evaluated as a yes/no benchmark here; --choice_only is not supported.")

        self.raw_image_dir = args.raw_image_dir
        self.image_path_dict = {}
        self.category_dict = {}

        records = load_mme_parquet_records(args, save_images=True)
        self.answer_dict = {}
        self.question_dict = {}
        self.rationale_dict = {}
        self.choices_dict = {}
        self.inputtext_dict = {}
        for rec in records:
            key = rec["key"]
            image_idx = rec["image_idx"]
            self.answer_dict[key] = [rec["answer"]]
            self.question_dict[key] = rec["question"]
            self.rationale_dict[key] = ""
            self.choices_dict[key] = ["yes", "no"]
            self.inputtext_dict[image_idx] = [""]
            self.image_path_dict[image_idx] = rec["image_path"]
            self.category_dict[key] = rec["category"]

        self.val_keys = list(self.question_dict.keys())
        self.direct_answer_eval_keys = set(self.val_keys)

        # MME has no train split in this local release; experiments use n_shot=0.
        self.traincontext_caption_dict = {}
        self.traincontext_answer_dict = {}
        self.traincontext_question_dict = {}
        self.traincontext_rationale_dict = {}
        self.traincontext_choices_dict = {}
        self.traincontext_interactive_answer_dict = {}
        self.traincontext_interactive_question_dict = {}
        self.train_keys = []
        self.train_interactive_keys = []

        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)

        self.train_ocr_text = {}
        self.val_ocr_text = {}

    def load_similarity(self):
        self.valkey2idx = {}

    def find_image(self, img_key):
        return Image.open(self.find_image_path(img_key)).convert("RGB")

    def find_image_path(self, img_key):
        return self.image_path_dict[int(img_key)]

# 根据图片id加载图片
def find_image(args, img_key):
    split = args.split_name
    img_full_path = os.path.join(args.raw_image_dir, "COCO_%s2014_%012d.jpg" % (split, img_key))
    return Image.open(img_full_path).convert("RGB")

# 获取图片路径
def find_image_path(args, img_key):
    split = args.split_name
    img_full_path = os.path.join(args.raw_image_dir,  "COCO_%s2014_%012d.jpg" % (split, img_key))
    return img_full_path

# 加载预计算的文本缓存（COCO图像描述和标签）
def load_aokvqa_dataset_v1(args):
    """
    加载数据集和相关缓存数据。
    根据参数配置加载验证集/测试集、训练集、图像描述、场景图等数据。
    
    Args:
        args: 包含配置参数的对象,包括：
            - test_only: 是否仅测试模式
            - raw_image_dir: 原始图像目录
            - coco_path: COCO数据集路径
            - start, end: 数据切片起始和结束比例（用于分布式训练）
            - caption_type: 描述类型（如'vinvl_ocr')
            - sg_path: 场景图数据路径
            - concept_caption_path: 概念描述路径
            - choice_only: 是否仅加载选择题选项
    """
    
    # 根据测试模式确定数据集划分名称
    split = args.split_name
    
    # 设置原始图像目录路径（如 .../val2017 或 .../test2017）
    raw_image_dir = os.path.join(args.raw_image_dir, "%s2017" % split)
    
    # 加载验证/测试集的标注数据
    # 返回元组: (_, 答案字典, 问题字典, 原理字典, 选项字典)
    _, answer_dict, question_dict, rationale_dict, choices_dict = \
        load_anno(
            None,  # 不加载额外的标注文件
            f'{args.coco_path}/aokvqa_v1p0_{split}.json',  # AOK-VQA问题文件
            f'{args.coco_path}/aokvqa_v1p0_{split}.json',  # 同上（可能为占位）
            choice_only=args.choice_only  # 是否仅加载选择题
        )
    
    # 获取验证集的所有问题ID作为键
    val_keys = list(question_dict.keys())

    return val_keys, raw_image_dir, answer_dict, question_dict, rationale_dict, choices_dict




def load_anno(coco_caption_file, answer_anno_file, question_anno_file, choice_only=False):
    """
    加载并处理标注数据,包括COCO图像描述、AOK-VQA问题、答案、选项和原理解释。
    
    此方法将原始JSON标注文件转换为易于查询的字典格式,键为"image_id<->question_id"的组合。
    
    Args:
        coco_caption_file (str): COCO图像描述文件的路径,为None则不加载
        answer_anno_file (str): AOK-VQA答案标注文件路径,包含答案、选项和原理
        question_anno_file (str): AOK-VQA问题标注文件路径,包含问题文本
        choice_only (bool): 如果为True,仅加载选择题的正确答案索引；否则加载直接答案
    
    Returns:
        tuple: 包含五个字典的元组,格式为:
            (caption_dict, answer_dict, question_dict, rationales_dict, choices_dict)
            其中每个字典的键都是"image_id<->question_id"格式的字符串
    """
    
    # 1. 加载原始JSON文件
    # 如果提供了COCO描述文件,则加载
    if coco_caption_file is not None:
        coco_caption = json.load(open(coco_caption_file, 'r'))
        # 如果文件是字典格式且有'annotations'键,则提取注释列表
        if type(coco_caption) == type({}): 
            coco_caption = coco_caption['annotations']
    
    # 加载AOK-VQA的答案标注文件
    answer_anno = json.load(open(answer_anno_file, 'r'))
    # 加载AOK-VQA的问题标注文件
    question_anno = json.load(open(question_anno_file, 'r'))

    # 2. 构建COCO图像描述字典
    # 格式: {image_id: [caption1, caption2, ...]}
    caption_dict = {}
    if coco_caption_file is not None:
        for sample in coco_caption:
            image_id = sample['image_id']
            if image_id not in caption_dict:
                # 为每个图像ID创建描述列表
                caption_dict[image_id] = [sample['caption']]
            else:
                # 追加额外的描述（每张图片通常有多个描述）
                caption_dict[image_id].append(sample['caption'])

    # 3. 构建答案字典
    # 格式: {"image_id<->question_id": answer_data}
    answer_dict = {}
    for sample in answer_anno:
        # 创建复合键：结合图像ID和问题ID,确保唯一性
        key = str(sample['image_id']) + '<->' + str(sample['question_id'])
        
        if key not in answer_dict:  # 避免重复
            if choice_only:
                # 选择题模式：存储正确答案的索引
                if 'correct_choice_idx' in sample:
                    answer_dict[key] = sample["correct_choice_idx"]
                else:
                    # 如果没有正确答案索引,默认为0（第一个选项）
                    answer_dict[key] = 0
            else:
                # 非选择题模式：存储直接答案列表
                if 'direct_answers' in sample:
                    answer_dict[key] = sample["direct_answers"]
                else:
                    # 如果没有直接答案,存储空列表
                    answer_dict[key] = [""]

    # 4. 构建问题字典
    # 格式: {"image_id<->question_id": question_text}
    question_dict = {}
    for sample in question_anno:
        key = str(sample['image_id']) + '<->' + str(sample['question_id'])
        if key not in question_dict:
            question_dict[key] = sample['question']

    # 5. 构建原理解释字典
    # 格式: {"image_id<->question_id": [rationale1, rationale2, ...]}
    rationales_dict = {}
    for sample in answer_anno:
        key = str(sample['image_id']) + '<->' + str(sample['question_id'])
        if key not in rationales_dict:
            if 'rationales' in sample:
                # 存储原理解释列表（通常有多个）
                rationales_dict[key] = sample['rationales']
            else:
                # 如果没有原理解释,存储空字符串
                rationales_dict[key] = ""

    # 6. 构建选项字典
    # 格式: {"image_id<->question_id": [choice1, choice2, ...]}
    choices_dict = {}
    for sample in answer_anno:
        key = str(sample['image_id']) + '<->' + str(sample['question_id'])
        # 注意：这里没有检查key是否已存在,因为每个问题都应该有选项
        choices_dict[key] = sample['choices']

    # 返回所有构建的字典
    return caption_dict, answer_dict, question_dict, rationales_dict, choices_dict
