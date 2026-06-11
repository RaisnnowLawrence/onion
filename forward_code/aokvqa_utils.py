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

        # 加载验证/测试集的标注数据
        # 返回元组: (_, 答案字典, 问题字典, 原理字典, 选项字典)
        _, self.answer_dict, self.question_dict, self.rationale_dict, self.choices_dict = \
            self.load_anno(
                None,  # 不加载额外的标注文件
                f'{args.coco_path}/aokvqa_v1p0_{split}.json',  # AOK-VQA问题文件
                f'{args.coco_path}/aokvqa_v1p0_{split}.json',  # 同上（可能为占位）
                choice_only=args.choice_only  # 是否仅加载选择题
            )
        
        # 获取验证集的所有问题ID作为键
        self.val_keys = list(self.question_dict.keys())

        ## 加载缓存的文本数据（COCO图像描述和标签）
        self.inputtext_dict = self.load_cachetext()
        
        # 加载训练集的上下文数据（用于训练或上下文学习）
        self.traincontext_caption_dict, self.traincontext_answer_dict, \
        self.traincontext_question_dict, self.traincontext_rationale_dict, \
        self.traincontext_choices_dict = \
            self.load_anno(
                '%s/captions_train2017.json' % args.coco_path,  # COCO训练集描述
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
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [
                        row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])]]
                else:
                    caption_dict[int(row[0])].append(
                        row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])])
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
        split = self.args.split_name
        img_full_path = os.path.join(self.args.raw_image_dir,  "val2017/", "COCO_%s2014_%012d.jpg" % (split, img_key))
        return Image.open(img_full_path).convert("RGB")

    # 获取图片路径
    def find_image_path(self, img_key):
        split = self.args.split_name
        img_full_path = os.path.join(self.args.raw_image_dir,  "val2017/", "COCO_%s2014_%012d.jpg" % (split, img_key))
        return img_full_path

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
