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
import re
import math
from tqdm import tqdm
from transformers import GPT2Tokenizer
import pdb
import pickle
import glob
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import datetime
import base64
from collections import Counter, defaultdict
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer
from lang_sam import LangSAM

from sam_utils import process_langsam_results_to_visualization, combine_masks_max_simple, clean_string_basic

from aokvqa_utils import aokvqa_dataset, okvqa_dataset, pope_dataset, mme_dataset, load_mme_answer_annotations
from qwen_utils import chat_with_qwen_vl, chat_with_qwen_vllm, string_to_list_if_possible
from mcts import MCTSQuestionSample
from official_vqa_answer_processor import normalize_vqa_answer


def process_answer(answer):
    answer = str(answer).replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)


def official_direct_answer_score(pred_answer, direct_answers):
    """Official VQA-style DA score after answer normalization: min(1, matches / 3)."""
    normalized_pred = normalize_vqa_answer(pred_answer)
    num_match = sum(normalized_pred == normalize_vqa_answer(answer) for answer in direct_answers)
    return min(1.0, num_match / 3.0)


def legacy_normalized_direct_answer_score(pred_answer, direct_answers):
    """Old internal score kept only for explicit backward-compatibility checks."""
    processed_pred_answer = process_answer(pred_answer)
    counter = 0
    for answer in direct_answers:
        if processed_pred_answer == process_answer(answer):
            counter += 1
    return min(1.0, float(counter) * 0.3)


def normalize_yes_no_answer(answer):
    text = normalize_vqa_answer(answer)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if text in ("true", "present"):
        return "yes"
    if text in ("false", "absent"):
        return "no"
    return text


def yes_no_answer_score(pred_answer, direct_answers):
    pred = normalize_yes_no_answer(pred_answer)
    gold = normalize_yes_no_answer(direct_answers[0] if direct_answers else "")
    return 1.0 if pred == gold else 0.0


class onion:    
    def __init__(self, args, dataset):

        self.dataset = dataset
        self.args = args
        self.messages = None
        self.attention_object = []
        self.qwen_global_caption_cache = {}
        self.qwen_local_caption_cache = {}
        self.external_knowledge_corpus = None
        self.external_knowledge_index = None
        self.strategy_profile = {}
        self.val_ocr_text = getattr(dataset, "val_ocr_text", {})
        self.train_ocr_text = getattr(dataset, "train_ocr_text", {})
        self.last_dyfo_visual_evidence = ""
        self.last_dyfo_focus_image_path = None
        self.train_keys = getattr(dataset, "train_keys", [])
        
        # 引擎初始化
        self.initialize_qwen(self.args.engine)

        # 图像处理部分按需初始化。Direct/非视觉增强实验不需要加载
        # GroundingDINO + SAM，否则多 shard 同时启动时容易产生很高的显存峰值。
        self.sam = None

        # 加载caption部分
        self.caption_qwen = self.load_caption_qwen()

        # 加载 WIT 外部知识。只有知识增强路线需要这份较大的本地知识表。
        self.wit_knowkedge = self.load_wit_knowkedge() if args.use_knowledge_enhance else {}

        if getattr(args, "strategy_profile_path", ""):
            self.strategy_profile = self.load_strategy_profile(args.strategy_profile_path)

        if args.with_clip_verify or args.choice_only or args.use_clip_thought_verify:
            model = CLIPTextModel.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            model = model.cuda()
            processor = CLIPProcessor.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            self.clip_model, self.clip_processor = model, processor

        # MCTS图像增强所需：加载完整CLIPModel（视觉+文本）用于reward计算
        self.clip_full_model = None
        self.clip_full_processor = None
        if args.use_image_enhance and getattr(args, "mcts_action_mode", "all") != "dyfo_evidence":
            self.clip_full_model = CLIPModel.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            self.clip_full_model = self.clip_full_model.cuda()
            self.clip_full_processor = CLIPProcessor.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")

        self.temp_question = "What is the person doing?"

    def _truncate_text(self, text, max_chars=500):
        """Keep accumulated evidence compact enough for repeated prompt injection."""
        if text is None:
            return ""
        text = str(text).replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _extract_answer_from_response(self, response):
        response_clean = str(response).strip()
        strategy = self.args.answer_extraction_strategy
        if strategy == "raw":
            return response_clean
        if strategy == "last_line":
            lines = [line.strip() for line in response_clean.split("\n") if line.strip()]
            return lines[-1] if lines else response_clean
        if strategy == "strict_final":
            import re
            matches = re.findall(r"(?:final\s+answer|answer)\s*:\s*(.+)", response_clean, flags=re.IGNORECASE)
            if matches:
                return matches[-1].strip()
            return response_clean

        answer_marker = 'Answer:'
        last_answer_idx = response_clean.rfind(answer_marker)
        if last_answer_idx != -1:
            extracted_answer = response_clean[last_answer_idx + len(answer_marker):].strip()
            if extracted_answer.lower().startswith('the answer is'):
                extracted_answer = extracted_answer[len('the answer is'):].strip()
            return extracted_answer
        lines = [line.strip() for line in response_clean.split('\n') if line.strip()]
        return lines[-1] if lines else response_clean

    def _clean_short_answer(self, answer):
        import re

        cleaned = str(answer).strip()
        cleaned = cleaned.split("\n")[0].strip()
        cleaned = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^the\s+answer\s+is\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.split(r"\s+(?:because|since|as|therefore)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        cleaned = cleaned.strip(" \t\"'`.,;:!?")
        return cleaned

    def _safe_rule_postprocess_answer(self, answer):
        import re

        cleaned = str(answer).strip()
        cleaned = cleaned.split("\n")[0].strip()
        cleaned = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^the\s+answer\s+is\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.split(r"\s+(?:because|since|as|therefore)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        cleaned = cleaned.strip(" \t\"'`.,;:!?")
        return cleaned

    def _postprocess_answer(self, answer):
        mode = getattr(self.args, "answer_postprocess", "none")
        if mode == "none":
            return answer
        if mode == "safe_rules":
            return self._safe_rule_postprocess_answer(answer)
        if mode == "legacy_visualcot":
            return process_answer(self._safe_rule_postprocess_answer(answer))
        return answer

    def _format_direct_answer_instruction(self, question, prompt_before_answer):
        if getattr(self.args, "dataset_name", "") in ("pope", "mme"):
            return (
                "=== Answer with only yes or no.\n"
                "%s" % prompt_before_answer
            )
        style = getattr(self.args, "direct_prompt_style", "default")
        if style == "answer_first_strict":
            return (
                "=== Answer with only one word or a short noun phrase.\n"
                "Do not explain. Do not write a full sentence. Do not add punctuation.\n"
                "Answer:"
            )
        if style == "type_specialist":
            qtype = self._classify_vqa_question_type(question)
            if self._question_is_count(question):
                constraint = "This is a counting question. Answer with a number only."
            elif self._question_is_ocr(question):
                constraint = "This is a text-reading question. Answer with the visible text only."
            elif "color" in str(question).lower():
                constraint = "This is a color question. Answer with color word(s) only."
            elif qtype == "visual_detail":
                constraint = "Answer with the directly visible detail as a short phrase."
            elif qtype == "category":
                constraint = "Answer with the object/category name as a short noun phrase."
            elif qtype == "knowledge":
                constraint = "Use the image first, then answer with the shortest plausible phrase."
            else:
                constraint = "Answer with a single word or short phrase."
            return (
                "=== %s\n"
                "Do not explain. Do not write a full sentence.\n"
                "Answer:"
            ) % constraint
        return (
            "=== Please fill in the answer with a short phrase or a single word:\n"
            "%s" % (prompt_before_answer)
        )

    def _build_direct_context_for_style(self, question, caption, regional_context, ocr_context):
        if getattr(self.args, "direct_prompt_style", "default") != "context_gated":
            return caption
        parts = []
        if self._question_is_ocr(question):
            if ocr_context:
                parts.append("OCR/Text evidence: " + ocr_context)
            if regional_context:
                parts.append(regional_context)
        elif self._question_is_count(question) or "color" in str(question).lower():
            if regional_context:
                parts.append(regional_context)
            if caption:
                parts.append(caption)
        else:
            if caption:
                parts.append(caption)
        return "\n".join(part for part in parts if part)

    def _parse_rephrased_questions(self, response, original_question):
        import re

        questions = []
        seen = {str(original_question).strip().lower()}
        for line in str(response).splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[\).:])\s*", "", line).strip()
            line = line.strip("\"'")
            if not line or line.lower().startswith(("question", "rephrase")):
                continue
            if not line.endswith("?") and "?" in line:
                line = line[:line.find("?") + 1]
            norm = line.lower()
            if norm in seen:
                continue
            questions.append(line)
            seen.add(norm)
            if len(questions) >= self.args.rephrase_num_questions:
                break
        return questions

    def _format_rephrase_generation_prompt(self, question, question_type):
        mode = self.args.rephrase_generation_mode
        if mode == "visual_focus":
            instruction = (
                "Make the visual target and relation clearer, while preserving exactly the same meaning. "
                "Do not add any new assumption."
            )
        elif mode == "answer_type":
            instruction = (
                "Rewrite the question so the expected answer type is explicit, such as number, color, object, place, text, or action. "
                "Do not change what is being asked."
            )
        elif mode == "mixed":
            instruction = (
                "Produce diverse but semantically equivalent rewrites: one simpler, one visual-target focused, "
                "and one answer-type focused when possible."
            )
        else:
            instruction = "Make the question simpler and clearer without changing its meaning."
        return (
            "Rewrite the visual question into %d semantically equivalent questions.\n"
            "%s\n"
            "The rewrites must ask for the same answer as the original question.\n"
            "Do not answer the question. Do not add choices. Output one question per line.\n"
            "Question type: %s\n"
            "Original question: %s"
        ) % (self.args.rephrase_num_questions, instruction, question_type, question)

    def _format_rephrase_direct_prompt(self, question, choice_text, context):
        prompt = (
            "Answer the visual question with a single word or short phrase.\n"
            "Do not explain. Do not write a full sentence.\n"
        )
        if context:
            prompt += "Brief Context: %s\n" % self._truncate_text(context, self.args.rephrase_context_max_chars)
        prompt += "Question: %s%s\nAnswer:" % (question, choice_text)
        return prompt

    def _rephrase_context(self, cur_caption, regional_context, ocr_context):
        mode = self.args.rephrase_answer_context
        if mode == "empty":
            return ""
        if mode == "regional":
            return "\n".join(part for part in (cur_caption, regional_context) if part)
        if mode == "ocr_regional":
            return "\n".join(part for part in (cur_caption, regional_context, ocr_context) if part)
        return cur_caption

    def _question_rephrase_should_trigger(self, question, question_type):
        trigger = self.args.rephrase_trigger
        if trigger == "always":
            return True
        if trigger == "risky_qtype":
            return (
                self._question_is_count(question)
                or self._question_is_ocr(question)
                or "color" in str(question).lower()
                or question_type in ("visual_detail", "category")
            )
        if trigger == "complex_qtype":
            return question_type in ("knowledge", "visual_detail", "category")
        return True

    def _rephrase_vote_proposal(self, initial_answer, answer_records):
        normalized_counts = {}
        norm_to_answer = {}
        for rec in answer_records:
            answer = self._clean_short_answer(rec.get("answer", ""))
            if not answer or self._looks_like_visual_cue_list(answer):
                continue
            norm = process_answer(answer)
            normalized_counts[norm] = normalized_counts.get(norm, 0) + 1
            norm_to_answer.setdefault(norm, answer)
        if not normalized_counts:
            return "", "", 0, normalized_counts
        initial_norm = process_answer(initial_answer)
        best_norm = max(normalized_counts, key=normalized_counts.get)
        best_votes = normalized_counts[best_norm]
        if best_norm != initial_norm and best_votes >= self.args.rephrase_consensus_threshold:
            return norm_to_answer[best_norm], best_norm, best_votes, normalized_counts
        return "", best_norm, best_votes, normalized_counts

    def _format_rephrase_review_prompt(self, original_question, choice_text, context, initial_answer,
                                       rephrase_questions, answer_records, proposed_answer):
        qa_lines = []
        for rec in answer_records:
            qa_lines.append("Q: %s\nA: %s" % (rec.get("question", ""), rec.get("answer", "")))
        return (
            "You are conservatively checking whether question rephrasing found a better short answer.\n"
            "The original direct answer is usually safer. Revise only if the rephrased questions are semantically equivalent "
            "and the proposed answer is clearly better supported by the image.\n"
            "If uncertain, keep the original answer.\n"
            "Brief Context: %s\n"
            "Original Question: %s%s\n"
            "Original Direct Answer: %s\n"
            "Rephrased QA:\n%s\n"
            "Proposed Answer: %s\n"
            "Output exactly:\n"
            "Decision: keep / revise\n"
            "Final Answer: <short answer>"
        ) % (
            self._truncate_text(context, self.args.rephrase_context_max_chars),
            original_question,
            choice_text,
            self._clean_short_answer(initial_answer),
            "\n---\n".join(qa_lines),
            self._clean_short_answer(proposed_answer),
        )

    def _extract_rephrase_review_answer(self, response, initial_answer, proposed_answer):
        import re

        text = str(response).strip()
        first_lines = "\n".join(text.splitlines()[:3]).lower()
        if "revise" not in first_lines:
            return self._clean_short_answer(initial_answer)
        matches = re.findall(r"final\s+answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if matches:
            answer = self._clean_short_answer(matches[-1])
        else:
            answer = self._clean_short_answer(proposed_answer)
        if not answer or self._looks_like_visual_cue_list(answer):
            return self._clean_short_answer(initial_answer)
        return answer

    def _run_rephrase_consistency(self, question, choice_text, cur_caption, regional_context, ocr_context,
                                  initial_answer, question_type, image_path):
        if not self._question_rephrase_should_trigger(question, question_type):
            return {
                "final_answer": self._clean_short_answer(initial_answer),
                "trace": "Rephrase Consistency skipped by trigger %s.\nFinal Answer: %s" % (
                    self.args.rephrase_trigger, self._clean_short_answer(initial_answer)
                ),
            }

        rephrase_prompt = self._format_rephrase_generation_prompt(question, question_type)
        rephrase_response = self._call_llm(
            rephrase_prompt, image_path=None, max_new_tokens=self.args.rephrase_generation_max_tokens
        )
        rephrased_questions = self._parse_rephrased_questions(rephrase_response, question)
        context = self._rephrase_context(cur_caption, regional_context, ocr_context)
        answer_records = []
        for rq in rephrased_questions:
            direct_prompt = self._format_rephrase_direct_prompt(rq, choice_text, context)
            direct_response = self._call_llm(
                direct_prompt, image_path=image_path, max_new_tokens=self.args.rephrase_answer_max_tokens
            )
            answer_records.append({
                "question": rq,
                "prompt": direct_prompt,
                "response": direct_response,
                "answer": self._clean_short_answer(self._extract_answer_from_response(direct_response)),
            })

        proposed_answer, best_norm, best_votes, vote_counts = self._rephrase_vote_proposal(
            initial_answer, answer_records
        )
        final_answer = self._clean_short_answer(initial_answer)
        review_prompt = ""
        review_response = ""
        arbitration = self.args.rephrase_arbitration

        if arbitration == "keep_baseline":
            final_answer = self._clean_short_answer(initial_answer)
        elif arbitration == "majority_if_consensus":
            if proposed_answer:
                final_answer = self._clean_short_answer(proposed_answer)
        elif arbitration == "all_agree":
            if proposed_answer and best_votes >= max(1, len(answer_records)):
                final_answer = self._clean_short_answer(proposed_answer)
        elif arbitration == "conservative_review":
            if proposed_answer:
                review_prompt = self._format_rephrase_review_prompt(
                    question, choice_text, context, initial_answer,
                    rephrased_questions, answer_records, proposed_answer
                )
                review_response = self._call_llm(
                    review_prompt, image_path=image_path, max_new_tokens=self.args.rephrase_review_max_tokens
                )
                final_answer = self._extract_rephrase_review_answer(
                    review_response, initial_answer, proposed_answer
                )

        trace = (
            "Rephrase Consistency\n"
            "Trigger: %s\n"
            "Generation Mode: %s\n"
            "Arbitration: %s\n"
            "Initial Answer: %s\n"
            "Rephrase Prompt:\n%s\n"
            "Rephrase Response:\n%s\n"
            "Answer Records:\n%s\n"
            "Vote Counts: %s\n"
            "Proposed Answer: %s (norm=%s votes=%s)\n"
            "Review Prompt:\n%s\n"
            "Review Response:\n%s\n"
            "Final Answer: %s"
        ) % (
            self.args.rephrase_trigger,
            self.args.rephrase_generation_mode,
            arbitration,
            self._clean_short_answer(initial_answer),
            rephrase_prompt,
            rephrase_response,
            json.dumps(answer_records, ensure_ascii=False, indent=2),
            json.dumps(vote_counts, ensure_ascii=False),
            proposed_answer,
            best_norm,
            best_votes,
            review_prompt,
            review_response,
            final_answer,
        )
        return {"final_answer": final_answer, "trace": trace, "answer_records": answer_records}

    def _extract_structured_cot_answer(self, response):
        import re

        response_clean = str(response).strip()
        matches = re.findall(r"(?:final\s+answer|answer)\s*:\s*(.+)", response_clean, flags=re.IGNORECASE)
        if matches:
            return self._clean_short_answer(matches[-1])

        lines = [line.strip() for line in response_clean.split("\n") if line.strip()]
        if not lines:
            return response_clean
        return self._clean_short_answer(lines[-1])

    def _extract_first_answer_line(self, response):
        import re

        response_clean = str(response).strip()
        match = re.search(r"(?:^|\n)\s*answer\s*:\s*(.+)", response_clean, flags=re.IGNORECASE)
        if match:
            return self._clean_short_answer(match.group(1))
        lines = [line.strip() for line in response_clean.split("\n") if line.strip()]
        return self._clean_short_answer(lines[0]) if lines else response_clean

    def _looks_like_visual_cue_list(self, answer):
        cleaned = str(answer).strip()
        if not cleaned:
            return True
        comma_parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(comma_parts) >= 3:
            return True
        words = cleaned.split()
        if len(words) > 8:
            return True
        cue_words = ("visible", "object", "cue", "image", "background", "foreground")
        return any(word in cleaned.lower() for word in cue_words) and len(words) > 3

    def _classify_vqa_question_type(self, question):
        text = str(question).lower()
        text_cues = ("text", "word", "letter", "sign", "read", "says", "written", "logo", "number")
        visual_detail_cues = (
            "how many", "count", "what color", "which color", "color", "where", "which side",
            "left", "right", "behind", "front", "next to", "wearing", "holding", "doing",
            "mouth", "hand", "what is in", "what are in", "what is on", "is there", "are there"
        )
        knowledge_cues = (
            "why", "used for", "use for", "purpose", "probably", "most likely", "event",
            "sport", "game", "season", "weather", "celebrated", "celebrating"
        )
        category_cues = (
            "what kind", "what type", "which animal", "what animal", "what food", "what object",
            "what item", "what device", "what appliance", "made of"
        )
        if any(cue in text for cue in text_cues):
            return "text_ocr"
        if any(cue in text for cue in visual_detail_cues):
            return "visual_detail"
        if any(cue in text for cue in knowledge_cues):
            return "knowledge"
        if any(cue in text for cue in category_cues):
            return "category"
        return "general"

    def _candidate_evidence_scope(self, question_type):
        if question_type == "text_ocr":
            return "Use OCR/text evidence and the original image heavily. Avoid relying on generic captions."
        if question_type == "visual_detail":
            return "Use visible image details, selected objects, regional evidence, and marked images if available."
        if question_type == "knowledge":
            return "Use the image to identify the scene/object, then use only directly relevant commonsense evidence."
        if question_type == "category":
            return "Use object identity, attributes, and local visual evidence. Avoid over-specific guesses."
        return "Use the original image first; treat text evidence as secondary and non-authoritative."

    def _question_has_any(self, question, cues):
        question_l = str(question).lower()
        return any(cue in question_l for cue in cues)

    def _question_is_count(self, question):
        return self._question_has_any(question, ("how many", "number of", "count"))

    def _question_is_ocr(self, question):
        return self._question_has_any(
            question, ("text", "word", "letter", "sign", "read", "says", "written", "logo")
        )

    def _normalize_candidate_answer(self, answer):
        return process_answer(self._clean_short_answer(answer))

    def _dedupe_candidate_records(self, records):
        deduped = []
        seen = set()
        for rec in records:
            answer = self._clean_short_answer(rec.get("answer", ""))
            norm = self._normalize_candidate_answer(answer)
            if not norm or norm in seen or self._looks_like_visual_cue_list(answer):
                continue
            new_rec = dict(rec)
            new_rec["answer"] = answer
            new_rec["normalized"] = norm
            deduped.append(new_rec)
            seen.add(norm)
        return deduped

    def _candidate_consensus_answer(self, records):
        counts = {}
        first_answer = {}
        for rec in records:
            answer = self._clean_short_answer(rec.get("answer", ""))
            if self._looks_like_visual_cue_list(answer):
                continue
            norm = rec.get("normalized") or self._normalize_candidate_answer(answer)
            if not norm:
                continue
            counts[norm] = counts.get(norm, 0) + 1
            first_answer.setdefault(norm, answer)
        if not counts:
            return ""
        best_norm = max(counts, key=counts.get)
        if counts[best_norm] >= max(2, getattr(self.args, "candidate_judge_consensus_votes", 2)):
            return first_answer[best_norm]
        return ""

    def _format_candidate_prompt(self, question, choice_text, context, style, question_type):
        base = (
            "Answer the visual question with a single word or short phrase.\n"
            "Do not list objects. Do not write a long explanation.\n"
            "Question type: %s\n"
        ) % question_type
        if context:
            base += "Brief Context: %s\n" % self._truncate_text(context, 900)
        base += "Question: %s%s\n" % (question, choice_text)
        if style == "image_only":
            return (
                "Answer using the image itself. Ignore any hidden captions or prior evidence.\n"
                "Return only the short answer.\n"
                "Question: %s%s\nAnswer:"
            ) % (question, choice_text)
        if style == "answer_first_locked":
            return (
                base +
                "Give the answer before any reasoning and do not revise it after giving reasons.\n"
                "Output exactly:\n"
                "Answer: <short answer>\n"
                "Reasons:\n"
                "1. <visible reason>\n"
                "2. <visible reason>"
            )
        if style == "caption_only":
            return (
                "Answer using the image and this brief caption/context. Keep the answer short.\n"
                "Brief Context: %s\n"
                "Question: %s%s\nAnswer:"
            ) % (self._truncate_text(context, 900), question, choice_text)
        if style == "visual_detail":
            return (
                base +
                "Focus on directly visible local details. If the question asks count/color/text/location, inspect carefully.\n"
                "Answer:"
            )
        if style == "knowledge_guarded":
            return (
                base +
                "Use commonsense only after identifying visible evidence in the image. Do not answer from caption alone.\n"
                "Answer:"
            )
        if style == "count_specialist":
            return (
                "You are solving a visual counting question.\n"
                "First identify exactly what needs to be counted. Inspect the full image and relevant local objects.\n"
                "Return only a number word such as zero, one, two, three, four, five, six, seven, eight, nine, or ten.\n"
                "Do not use digits. Do not explain.\n"
                "Context: %s\n"
                "Question: %s%s\n"
                "Answer:"
            ) % (self._truncate_text(context, 1200), question, choice_text)
        if style == "ocr_specialist":
            return (
                "Answer the question by carefully reading visible text, signs, screens, logos, labels, or numbers in the image.\n"
                "Use OCR/context only as hints; verify against the image when possible.\n"
                "Return only the short answer.\n"
                "Context/OCR hints: %s\n"
                "Question: %s%s\n"
                "Answer:"
            ) % (self._truncate_text(context, 1200), question, choice_text)
        if style == "coverage_scan":
            return (
                "Answer after scanning the full image, selected objects, and regional/context evidence.\n"
                "Do not follow the first obvious guess if a smaller or background object better answers the question.\n"
                "Return only a single word or short phrase.\n"
                "Context: %s\n"
                "Question: %s%s\n"
                "Answer:"
            ) % (self._truncate_text(context, 1400), question, choice_text)
        if style == "contrastive":
            return (
                "A previous VQA answer may be biased toward the most obvious object.\n"
                "Generate a plausible alternative answer only if it is visually supported by the image or context.\n"
                "If no better alternative is visible, repeat the initial answer.\n"
                "Initial answer: %s\n"
                "Context: %s\n"
                "Question: %s%s\n"
                "Alternative final answer:"
            ) % (self._clean_short_answer(getattr(self, "_current_initial_answer", "")),
                 self._truncate_text(context, 1200), question, choice_text)
        return base + "Answer:"

    def _call_candidate_answer(self, label, prompt, image_path, extractor="short"):
        response = self._call_llm(prompt, image_path=image_path)
        if extractor == "first_answer":
            answer = self._extract_first_answer_line(response)
        elif extractor == "structured":
            answer = self._extract_structured_cot_answer(response)
        else:
            answer = self._clean_short_answer(self._extract_answer_from_response(response))
        return {
            "label": label,
            "answer": answer,
            "prompt": prompt,
            "response": response,
        }

    def _format_candidate_judge_prompt(self, question, choice_text, question_type, evidence_text, candidate_records):
        candidate_lines = []
        for idx, rec in enumerate(candidate_records, start=1):
            candidate_lines.append("%d. [%s] %s" % (idx, rec.get("label", "candidate"), rec.get("answer", "")))
        return (
            "You are a conservative VQA answer judge. The candidate answers were produced by different strategies.\n"
            "Your job is to choose the best final answer, not to freely invent a new one.\n"
            "Prefer a candidate answer that is directly supported by the image. If evidence is uncertain, prefer the "
            "direct or answer-first candidate over evidence-heavy candidates.\n"
            "%s\n"
            "Question type: %s\n"
            "Question: %s%s\n"
            "Candidate Answers:\n"
            "%s\n"
            "Evidence:\n"
            "%s\n"
            "Output exactly in this format:\n"
            "Evidence Check: supported / contradicted / uncertain\n"
            "Chosen Candidate: <number>\n"
            "Final Answer:"
        ) % (
            self._candidate_evidence_scope(question_type),
            question_type,
            question,
            choice_text,
            "\n".join(candidate_lines),
            evidence_text,
        )

    def _extract_candidate_judge_answer(self, response, candidate_records, fallback_answer):
        import re

        response_clean = str(response).strip()
        match = re.search(r"chosen\s+candidate\s*:\s*(\d+)", response_clean, flags=re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidate_records):
                return self._clean_short_answer(candidate_records[idx]["answer"])

        final_answer = self._extract_structured_cot_answer(response_clean)
        final_norm = self._normalize_candidate_answer(final_answer)
        for rec in candidate_records:
            if final_norm and final_norm == rec.get("normalized"):
                return self._clean_short_answer(rec["answer"])

        if self._looks_like_visual_cue_list(final_answer):
            return self._clean_short_answer(fallback_answer)
        if getattr(self.args, "candidate_judge_allow_new_answer", False):
            return final_answer
        return self._clean_short_answer(fallback_answer)

    def load_strategy_profile(self, profile_path):
        profile = {}
        if not profile_path or not os.path.isfile(profile_path):
            print(f"[strategy_router] profile missing, route defaults to direct: {profile_path}")
            return profile

        with open(profile_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("key")
                if not key:
                    continue
                if "scores" in rec:
                    profile[key] = rec
                    continue

                strategy = rec.get("strategy")
                if not strategy:
                    continue
                dst = profile.setdefault(key, {
                    "key": key,
                    "image_id": rec.get("image_id"),
                    "question": rec.get("question", ""),
                    "question_type": rec.get("question_type", ""),
                    "scores": {},
                    "answers": {},
                })
                dst["scores"][strategy] = float(rec.get("score", 0.0))
                dst["answers"][strategy] = rec.get("pred_answer", "")

        print(f"[strategy_router] loaded {len(profile)} strategy-profile samples from {profile_path}")
        return profile

    def _route_with_strategy_profile(self, key, question):
        if not self.strategy_profile:
            return {
                "strategy": self.args.strategy_router_default,
                "reason": "missing_profile",
                "neighbors": [],
                "direct_avg": 0.0,
                "cot_avg": 0.0,
                "rescue_rate": 0.0,
                "damage_rate": 0.0,
            }

        question_type = self._classify_vqa_question_type(question)
        topk = max(1, self.args.strategy_topk)
        context_multiplier = 6 if self.args.strategy_router_mode == "qtype_conditional" else 3
        context_keys = self.get_context_keys(key, self.args.strategy_retrieval_metric, topk * context_multiplier)
        if not context_keys:
            context_keys = []

        direct_name = self.args.strategy_direct_name
        cot_name = self.args.strategy_cot_name
        neighbors = []
        for ctx_key in context_keys:
            rec = self.strategy_profile.get(ctx_key)
            if not rec:
                continue
            scores = rec.get("scores", {})
            if direct_name not in scores:
                continue
            if self.args.strategy_router_mode != "direct_failure" and cot_name not in scores:
                continue
            if (
                self.args.strategy_router_mode == "qtype_conditional"
                and rec.get("question_type", "") != question_type
            ):
                continue
            direct_score = float(scores.get(direct_name, 0.0))
            cot_score = float(scores.get(cot_name, direct_score))
            neighbors.append({
                "key": ctx_key,
                "direct": direct_score,
                "cot": cot_score,
                "question_type": rec.get("question_type", ""),
            })
            if len(neighbors) >= topk:
                break

        if len(neighbors) < self.args.strategy_min_neighbors:
            return {
                "strategy": self.args.strategy_router_default,
                "reason": "too_few_neighbors",
                "neighbors": neighbors,
                "direct_avg": 0.0,
                "cot_avg": 0.0,
                "rescue_rate": 0.0,
                "damage_rate": 0.0,
            }

        direct_avg = sum(n["direct"] for n in neighbors) / len(neighbors)
        cot_avg = sum(n["cot"] for n in neighbors) / len(neighbors)
        direct_hard = [n for n in neighbors if n["direct"] <= self.args.strategy_direct_hard_threshold]
        direct_safe = [n for n in neighbors if n["direct"] >= self.args.strategy_direct_safe_threshold]
        rescue = [n for n in neighbors if n["direct"] <= self.args.strategy_direct_hard_threshold and n["cot"] > n["direct"]]
        complex_win = [n for n in neighbors if n["direct"] <= self.args.strategy_direct_hard_threshold and n["cot"] >= self.args.strategy_direct_safe_threshold]
        damage = [n for n in neighbors if n["direct"] >= self.args.strategy_direct_safe_threshold and n["cot"] <= self.args.strategy_direct_hard_threshold]
        direct_hard_rate = len(direct_hard) / len(neighbors)
        direct_safe_rate = len(direct_safe) / len(neighbors)
        rescue_rate = len(rescue) / len(neighbors)
        complex_win_rate = len(complex_win) / len(neighbors)
        damage_rate = len(damage) / len(neighbors)

        mode = self.args.strategy_router_mode
        if mode == "direct_failure":
            use_cot = direct_hard_rate >= self.args.strategy_min_direct_hard_rate
            reason = "direct_hard_neighbors" if use_cot else "direct_neighbors_safe"
        elif mode == "direct_vs_complex":
            use_cot = (
                complex_win_rate >= self.args.strategy_min_complex_win_rate
                or cot_avg - direct_avg >= self.args.strategy_margin
            )
            reason = "complex_win_neighbors" if use_cot else "direct_wins_neighbors"
        elif mode == "qtype_conditional":
            use_cot = (
                complex_win_rate >= self.args.strategy_min_complex_win_rate
                or (
                    cot_avg - direct_avg >= self.args.strategy_margin
                    and rescue_rate >= self.args.strategy_min_rescue_rate
                )
            )
            reason = "qtype_complex_neighbors" if use_cot else "qtype_direct_neighbors"
        elif mode == "conservative_risk":
            net_gain = rescue_rate - damage_rate
            use_cot = (
                net_gain >= self.args.strategy_min_net_gain
                and cot_avg - direct_avg >= self.args.strategy_margin
                and damage_rate <= self.args.strategy_max_damage_rate
            )
            reason = "positive_rescue_damage_tradeoff" if use_cot else "direct_safer_by_risk"
        else:
            use_cot = (
                cot_avg - direct_avg >= self.args.strategy_margin
                and rescue_rate >= self.args.strategy_min_rescue_rate
                and damage_rate <= self.args.strategy_max_damage_rate
            )
            reason = "cot_neighbors_win" if use_cot else "direct_default_or_safer"

        strategy = cot_name if use_cot else direct_name
        return {
            "strategy": strategy,
            "reason": reason,
            "neighbors": neighbors,
            "direct_avg": direct_avg,
            "cot_avg": cot_avg,
            "direct_hard_rate": direct_hard_rate,
            "direct_safe_rate": direct_safe_rate,
            "rescue_rate": rescue_rate,
            "complex_win_rate": complex_win_rate,
            "damage_rate": damage_rate,
        }

    def _route_with_multi_strategy_profile(self, key, question):
        default_strategy = self.args.multi_strategy_default
        if not self.strategy_profile:
            return {
                "strategy": default_strategy,
                "reason": "missing_profile",
                "neighbors": [],
                "strategy_avgs": {},
                "best_avg": 0.0,
                "default_avg": 0.0,
            }

        strategies = [s.strip() for s in self.args.multi_strategy_names.split(",") if s.strip()]
        if default_strategy not in strategies:
            strategies.insert(0, default_strategy)

        context_keys = self.get_context_keys(
            key, self.args.strategy_retrieval_metric, max(1, self.args.strategy_topk * 4)
        ) or []
        neighbors = []
        for ctx_key in context_keys:
            rec = self.strategy_profile.get(ctx_key)
            if not rec:
                continue
            scores = rec.get("scores", {})
            if default_strategy not in scores:
                continue
            available = {name: float(scores[name]) for name in strategies if name in scores}
            if len(available) < 2:
                continue
            neighbors.append({"key": ctx_key, "scores": available})
            if len(neighbors) >= self.args.strategy_topk:
                break

        if len(neighbors) < self.args.strategy_min_neighbors:
            return {
                "strategy": default_strategy,
                "reason": "too_few_neighbors",
                "neighbors": neighbors,
                "strategy_avgs": {},
                "best_avg": 0.0,
                "default_avg": 0.0,
            }

        sums = defaultdict(float)
        counts = Counter()
        for item in neighbors:
            for name, score in item["scores"].items():
                sums[name] += score
                counts[name] += 1

        avgs = {
            name: (sums[name] / counts[name])
            for name in strategies
            if counts[name] >= self.args.strategy_min_neighbors
        }
        default_avg = avgs.get(default_strategy, 0.0)
        if not avgs:
            return {
                "strategy": default_strategy,
                "reason": "no_strategy_avgs",
                "neighbors": neighbors,
                "strategy_avgs": {},
                "best_avg": 0.0,
                "default_avg": default_avg,
            }

        best_strategy = max(avgs, key=avgs.get)
        best_avg = avgs[best_strategy]
        if best_strategy == default_strategy:
            selected = default_strategy
            reason = "default_best"
        elif best_avg - default_avg >= self.args.multi_strategy_margin:
            selected = best_strategy
            reason = "best_neighbor_strategy"
        else:
            selected = default_strategy
            reason = "default_within_margin"

        return {
            "strategy": selected,
            "reason": reason,
            "neighbors": neighbors,
            "strategy_avgs": avgs,
            "best_avg": best_avg,
            "default_avg": default_avg,
        }

    def _format_protected_review_prompt(self, cur_caption, question, choice_text, initial_answer):
        return (
            "You are a conservative VQA reviewer. The initial answer is usually correct.\n"
            "Your task is not to freely reason from scratch. Only revise if the image or context gives direct, "
            "specific, high-confidence evidence that contradicts the initial answer.\n"
            "If evidence is incomplete, ambiguous, caption-like, or merely suggests another possibility, keep the initial answer.\n"
            "=== Brief Context:\n"
            "%s\n"
            "=== Question:\n"
            "Question: %s%s\n"
            "Initial Answer: %s\n"
            "Output exactly in this format:\n"
            "Support: <short evidence supporting the initial answer, or none>\n"
            "Contradiction: <short direct contradictory evidence, or none>\n"
            "Evidence Check: supported / contradicted / uncertain\n"
            "Confidence: high / medium / low\n"
            "Decision: keep / revise\n"
            "Final Answer:"
        ) % (cur_caption, question, choice_text, initial_answer)

    def _extract_protected_review_answer(self, response, initial_answer):
        response_clean = str(response).strip()
        first_lines = "\n".join(response_clean.splitlines()[:6]).lower()
        if "decision: revise" not in first_lines:
            return self._clean_short_answer(initial_answer)
        if "evidence check: contradicted" not in first_lines and "contradicted" not in first_lines:
            return self._clean_short_answer(initial_answer)
        if "confidence: high" not in first_lines:
            return self._clean_short_answer(initial_answer)
        revised = self._extract_structured_cot_answer(response_clean)
        if self._looks_like_visual_cue_list(revised):
            return self._clean_short_answer(initial_answer)
        return revised

    def _run_reflective_r3_runtime(self, question, choice_text, cur_caption, image_path):
        first_prompt = (
            "Answer the visual question with a single word or short phrase.\n"
            "Do not explain yet. The first response should contain only the answer.\n"
            "Brief Context: %s\n"
            "Question: %s%s\n"
            "Answer:"
        ) % (cur_caption, question, choice_text)
        first_response = self._call_llm(first_prompt, image_path=image_path)
        current_answer = self._extract_first_answer_line(first_response)
        rationale_prompt = self._format_reflective_rationale_prompt(
            cur_caption, question, choice_text, current_answer
        )
        rationale_response = self._call_llm(rationale_prompt, image_path=image_path)
        review_prompt = self._format_reflective_review_prompt(
            cur_caption, question, choice_text, current_answer, rationale_response
        )
        review_response = self._call_llm(review_prompt, image_path=image_path)
        final_answer = self._extract_reflective_review_answer(review_response, current_answer)
        transcript = (
            "Reflective R3 Runtime\n"
            "Round 1 Prompt:\n%s\n"
            "Round 1 Response:\n%s\n"
            "Evidence Prompt:\n%s\n"
            "Evidence Response:\n%s\n"
            "Review Prompt:\n%s\n"
            "Review Response:\n%s\n"
            "Final Answer: %s"
        ) % (
            first_prompt, first_response, rationale_prompt, rationale_response,
            review_prompt, review_response, final_answer,
        )
        return final_answer, transcript

    def _run_answer_first_locked_runtime(self, question, choice_text, image_path):
        prompt = self._format_candidate_prompt(
            question, choice_text, "", "answer_first_locked", "general"
        )
        response = self._call_llm(prompt, image_path=image_path)
        answer = self._extract_first_answer_line(response)
        return answer, "Answer First Locked Runtime\nPrompt:\n%s\nResponse:\n%s\nFinal Answer: %s" % (
            prompt, response, answer
        )

    def _is_complex_for_decomposition(self, question):
        if self.args.decompose_complexity_mode == "always":
            return True
        if self.args.decompose_complexity_mode == "never":
            return False

        text = str(question).lower()
        broad_cues = (
            "how many", "number of", "count", "what type", "what kind", "which one",
            "which of", "where", "which side", "left", "right", "front", "behind",
            "next to", "under", "above", "why", "used for", "use for", "purpose",
            "probably", "most likely", "likely", "might", "could", "brand", "sign",
            "text", "word", "letter", "says", "written", "logo", "first number",
            "second", "time", "hour", "percent", "associated", "famous", "made of",
        )
        conservative_cues = (
            "why", "used for", "use for", "purpose", "probably", "most likely",
            "likely", "brand", "sign", "text", "word", "letter", "says", "written",
            "logo", "first number", "license", "time", "hour", "percent", "in front",
            "behind", "associated", "famous",
        )
        cues = conservative_cues if self.args.decompose_complexity_mode == "conservative" else broad_cues
        return any(cue in text for cue in cues)

    def _format_decompose_prompt(self, question, choice_text, context, direct_answer, question_type):
        return (
            "You are solving a complex visual question by decomposing it into smaller evidence questions.\n"
            "Do not write free-form chain-of-thought. Use short, evidence-seeking subquestions.\n"
            "Each subquestion should check one visible detail, text/brand cue, spatial relation, count, or commonsense link needed by the original question.\n"
            "If the question is answerable directly, use only one simple subquestion.\n"
            "Return a single short final answer.\n"
            "Question type: %s\n"
            "Initial direct answer: %s\n"
            "Context: %s\n"
            "Original Question: %s%s\n"
            "Output exactly:\n"
            "Subquestions:\n"
            "1. <subquestion> -> <short answer>\n"
            "2. <subquestion> -> <short answer>\n"
            "Final Answer:"
        ) % (
            question_type,
            self._clean_short_answer(direct_answer),
            self._truncate_text(context, self.args.decompose_context_max_chars),
            question,
            choice_text,
        )

    def _format_decompose_verify_prompt(self, question, choice_text, context, direct_answer,
                                        decomposed_answer, decompose_response):
        return (
            "You are a conservative VQA verifier.\n"
            "The direct answer is usually safer for simple questions. The decomposed answer should replace it only when the subquestions provide clear, specific evidence that the direct answer missed.\n"
            "If the decomposed evidence is uncertain, generic, or only guesses from commonsense, keep the direct answer.\n"
            "Context: %s\n"
            "Question: %s%s\n"
            "Direct Answer: %s\n"
            "Decomposed Answer: %s\n"
            "Decomposition Trace:\n%s\n"
            "Output exactly:\n"
            "Evidence Check: direct_supported / decomposed_supported / uncertain\n"
            "Decision: keep_direct / use_decomposed\n"
            "Final Answer:"
        ) % (
            self._truncate_text(context, self.args.decompose_context_max_chars),
            question,
            choice_text,
            self._clean_short_answer(direct_answer),
            self._clean_short_answer(decomposed_answer),
            self._truncate_text(decompose_response, 1600),
        )

    def _extract_decompose_verify_answer(self, response, direct_answer, decomposed_answer):
        first_lines = "\n".join(str(response).splitlines()[:6]).lower()
        if "decision: use_decomposed" not in first_lines:
            return self._clean_short_answer(direct_answer)
        if "evidence check: decomposed_supported" not in first_lines and "decomposed_supported" not in first_lines:
            return self._clean_short_answer(direct_answer)
        final_answer = self._extract_structured_cot_answer(response)
        if not final_answer or self._looks_like_visual_cue_list(final_answer):
            return self._clean_short_answer(direct_answer)
        final_norm = self._normalize_candidate_answer(final_answer)
        decomposed_norm = self._normalize_candidate_answer(decomposed_answer)
        if decomposed_norm and final_norm != decomposed_norm:
            return self._clean_short_answer(direct_answer)
        return self._clean_short_answer(final_answer)

    def _run_complex_decompose_from_direct(self, question, choice_text, context,
                                           direct_answer, question_type, image_path):
        should_decompose = self._is_complex_for_decomposition(question)
        decompose_prompt = ""
        decompose_response = ""
        verify_prompt = ""
        verify_response = ""
        decomposed_answer = ""
        final_answer = self._clean_short_answer(direct_answer)

        if should_decompose:
            decompose_prompt = self._format_decompose_prompt(
                question, choice_text, context, final_answer, question_type
            )
            decompose_response = self._call_llm(decompose_prompt, image_path=image_path)
            decomposed_answer = self._clean_short_answer(
                self._extract_structured_cot_answer(decompose_response)
            )
            if self._looks_like_visual_cue_list(decomposed_answer) or not decomposed_answer:
                decomposed_answer = final_answer

            if self.args.decompose_verify:
                verify_prompt = self._format_decompose_verify_prompt(
                    question, choice_text, context, final_answer,
                    decomposed_answer, decompose_response
                )
                verify_response = self._call_llm(verify_prompt, image_path=image_path)
                final_answer = self._extract_decompose_verify_answer(
                    verify_response, final_answer, decomposed_answer
                )
            else:
                final_answer = decomposed_answer

        return {
            "should_decompose": should_decompose,
            "decompose_prompt": decompose_prompt,
            "decompose_response": decompose_response,
            "decomposed_answer": decomposed_answer,
            "verify_prompt": verify_prompt,
            "verify_response": verify_response,
            "final_answer": final_answer,
        }

    def _format_direct_verify_prompt(self, cur_caption, question, choice_text, initial_answer):
        policy_text = {
            "balanced": "Prefer keeping the initial answer unless the evidence clearly contradicts it.",
            "keep_stronger": (
                "Strongly prefer keeping the initial answer. Revise it only when the image or context "
                "provides clear, specific, and direct contradictory evidence."
            ),
            "conflict_only": (
                "You may revise the initial answer only if Evidence Check is contradicted. "
                "If the evidence is supported or uncertain, keep the initial answer exactly."
            ),
            "revise_freely": (
                "Use the evidence to choose the best answer, even if that means revising the initial answer."
            ),
            "no_fallback": "Prefer keeping the initial answer unless the evidence clearly contradicts it.",
        }.get(self.args.direct_verify_policy, "Prefer keeping the initial answer unless the evidence clearly contradicts it.")
        return (
            "Please verify an initial visual question answering result using the image and context.\n"
            "%s\n"
            "Do not replace the answer with object lists or visual cue lists.\n"
            "The final answer must be a single word or short phrase.\n"
            "===The context you need to refer to:\n"
            "Brief Context: %s\n"
            "===The question you need to answer:\n"
            "Question: %s%s\n"
            "Initial Answer: %s\n"
            "Output exactly in this format:\n"
            "Evidence Check: supported / contradicted / uncertain\n"
            "Evidence: <at most 3 short visual or contextual cues>\n"
            "Final Answer:"
        ) % (policy_text, cur_caption, question, choice_text, initial_answer)

    def _evidence_scope_enabled(self, kind):
        scope = getattr(self.args, "reviewer_evidence_scope", "all")
        if scope == "all":
            return True
        if scope == "selective":
            return kind in getattr(self, "_current_selective_evidence_kinds", {"caption"})
        if scope == "caption_object":
            return kind in ("caption", "object")
        if scope == "caption_only":
            return kind == "caption"
        if scope == "object_only":
            return kind == "object"
        if scope == "enhance_only":
            return kind in ("image", "caption_enhance", "knowledge")
        if scope == "no_caption":
            return kind != "caption"
        if scope == "no_objects":
            return kind != "object"
        return True

    def _selective_reviewer_evidence_kinds(self, question):
        question_l = str(question).lower()
        kinds = {"caption"}

        visual_detail_keywords = (
            "how many", "number", "count", "what color", "which color", "color",
            "where", "which side", "left", "right", "front", "behind", "next to",
            "sign", "text", "read", "says", "letter", "logo"
        )
        local_caption_keywords = (
            "wearing", "holding", "carrying", "mouth", "head", "hand", "face",
            "what type", "what kind", "what object", "which object", "animal",
            "person", "device", "appliance", "made of", "doing"
        )
        knowledge_keywords = (
            "used for", "use for", "why", "purpose", "probably", "celebrated",
            "celebrating", "event", "sport", "game", "weather", "season"
        )

        if any(keyword in question_l for keyword in visual_detail_keywords):
            kinds.add("image")
        if any(keyword in question_l for keyword in local_caption_keywords):
            kinds.add("caption_enhance")
        if any(keyword in question_l for keyword in knowledge_keywords):
            kinds.add("knowledge")
        return kinds

    def _build_reviewer_evidence(self, base_context, selected_objects, regional_context, ocr_context,
                                 enhance_caption, enhance_knowledge, enhance_image_path,
                                 qwen_global_caption="", qwen_local_caption="", dyfo_visual_evidence=""):
        evidence_lines = []

        if self._evidence_scope_enabled("caption") and base_context:
            evidence_lines.append("Caption evidence: %s" % self._truncate_text(base_context, 700))

        if self._evidence_scope_enabled("object") and selected_objects:
            evidence_lines.append("Selected object evidence: %s" % ", ".join(selected_objects))

        if self._evidence_scope_enabled("object") and regional_context:
            evidence_lines.append("Regional object evidence: %s" % self._truncate_text(regional_context, 700))

        if self._evidence_scope_enabled("caption_enhance") and enhance_caption:
            evidence_lines.append("Targeted caption evidence: %s" % self._truncate_text(enhance_caption, 700))

        if self._evidence_scope_enabled("knowledge") and enhance_knowledge:
            evidence_lines.append("Knowledge evidence: %s" % self._truncate_text(enhance_knowledge, 700))

        if self._evidence_scope_enabled("caption") and qwen_global_caption:
            evidence_lines.append("Qwen global caption evidence: %s" % self._truncate_text(qwen_global_caption, 500))

        if self._evidence_scope_enabled("caption") and qwen_local_caption:
            evidence_lines.append("Qwen local caption evidence: %s" % self._truncate_text(qwen_local_caption, 500))

        if self._evidence_scope_enabled("object") and ocr_context:
            evidence_lines.append("OCR evidence: %s" % self._truncate_text(ocr_context, 500))

        if self._evidence_scope_enabled("image") and enhance_image_path:
            evidence_lines.append(
                "Enhanced image evidence: an auxiliary marked/outlined image view is provided to inspect local visual details."
            )

        if self._evidence_scope_enabled("image") and dyfo_visual_evidence:
            evidence_lines.append("DyFo visual evidence: %s" % self._truncate_text(dyfo_visual_evidence, 900))

        if not evidence_lines:
            return "No extra evidence is available. Keep the initial answer unless the original image clearly contradicts it."

        return "\n".join("- " + line for line in evidence_lines)

    def _format_reviewer_evidence_prompt(self, question, choice_text, initial_answer, evidence_text):
        policy_text = {
            "balanced": "Prefer keeping the initial answer unless the evidence clearly contradicts it.",
            "keep_stronger": (
                "Strongly prefer keeping the initial answer. Revise it only when the evidence is visual, specific, "
                "and directly contradicts the initial answer."
            ),
            "conflict_only": (
                "You are a conservative answer reviewer. You may revise the initial answer only if Evidence Check "
                "is contradicted. If the evidence is supported or uncertain, keep the initial answer exactly."
            ),
            "revise_freely": "Use the evidence to choose the best answer, even if that revises the initial answer.",
            "no_fallback": "Prefer keeping the initial answer unless the evidence clearly contradicts it.",
        }.get(self.args.direct_verify_policy, "Prefer keeping the initial answer unless the evidence clearly contradicts it.")

        return (
            "Review an initial visual question answering result. The enhancement modules are evidence providers, "
            "not answer generators.\n"
            "%s\n"
            "Use only the provided image and the explicit evidence below. Do not invent visual details that are not visible or listed.\n"
            "Do not replace the answer with an object list, caption, or visual cue list.\n"
            "The final answer must be a single word or short phrase.\n"
            "=== Question:\n"
            "Question: %s%s\n"
            "Initial Answer: %s\n"
            "=== Evidence from enhancement modules:\n"
            "%s\n"
            "Output exactly in this format:\n"
            "Evidence Check: supported / contradicted / uncertain\n"
            "Evidence: <at most 3 short evidence points>\n"
            "Final Answer:"
        ) % (policy_text, question, choice_text, initial_answer, evidence_text)

    def _format_reflective_rationale_prompt(self, cur_caption, question, choice_text, current_answer):
        evidence_rule = (
            "Use only details directly visible in the image. Do not mention common usage, typical purpose, "
            "world knowledge, or what objects are usually for.\n"
            if self.args.reflect_evidence_mode == "visible_only" else ""
        )
        return (
            "The model has already chosen an answer for a visual question. Do not change the answer in this step.\n"
            "Your task is only to write the smallest necessary visual evidence that supports or fails to support it.\n"
            "Use the image and the brief context only. Do not invent details.\n"
            "%s"
            "Write at most 2 short evidence points.\n"
            "=== Brief Context:\n"
            "%s\n"
            "=== Question:\n"
            "Question: %s%s\n"
            "Current Answer: %s\n"
            "Output exactly in this format:\n"
            "Evidence:\n"
            "1. <short visual evidence>\n"
            "2. <short visual evidence>"
        ) % (evidence_rule, cur_caption, question, choice_text, current_answer)

    def _format_reflective_review_prompt(self, cur_caption, question, choice_text, current_answer, rationale):
        if self.args.reflect_review_format == "keep_revise":
            return (
                "Review the current answer conservatively. The evidence step is a check, not a chance to freely reason.\n"
                "Choose keep unless the image/context clearly contradicts the current answer.\n"
                "Only choose revise when the contradiction is direct and the corrected answer is a single word or short phrase.\n"
                "=== Brief Context:\n"
                "%s\n"
                "=== Question:\n"
                "Question: %s%s\n"
                "Current Answer: %s\n"
                "=== Evidence Notes:\n"
                "%s\n"
                "Output exactly in this format:\n"
                "Evidence Check: supported / contradicted / uncertain\n"
                "Decision: keep / revise\n"
                "Corrected Answer:"
            ) % (cur_caption, question, choice_text, current_answer, rationale)
        return (
            "Review the current answer conservatively. The evidence step is a check, not a chance to freely reason.\n"
            "Keep the current answer if the evidence supports it or is uncertain.\n"
            "Revise the answer only when the image/context clearly contradicts it.\n"
            "The final answer must be a single word or short phrase.\n"
            "=== Brief Context:\n"
            "%s\n"
            "=== Question:\n"
            "Question: %s%s\n"
            "Current Answer: %s\n"
            "=== Evidence Notes:\n"
            "%s\n"
            "Output exactly in this format:\n"
            "Evidence Check: supported / contradicted / uncertain\n"
            "Evidence: <at most 2 short points>\n"
            "Final Answer:"
        ) % (cur_caption, question, choice_text, current_answer, rationale)

    def _extract_reflective_review_answer(self, response, initial_answer):
        if self.args.reflect_review_format != "keep_revise":
            return self._extract_direct_verify_answer(response, initial_answer)

        import re

        response_clean = str(response).strip()
        first_lines = "\n".join(response_clean.splitlines()[:4]).lower()
        if "contradicted" not in first_lines or "decision: revise" not in first_lines:
            return self._clean_short_answer(initial_answer)

        matches = re.findall(r"corrected\s+answer\s*:\s*(.+)", response_clean, flags=re.IGNORECASE)
        if not matches:
            return self._clean_short_answer(initial_answer)
        corrected = self._clean_short_answer(matches[-1])
        if self._looks_like_visual_cue_list(corrected):
            return self._clean_short_answer(initial_answer)
        return corrected

    def _extract_reflective_confidence(self, response):
        import re

        response_clean = str(response).strip()
        match = re.search(r"(?:^|\n)\s*confidence\s*:\s*(high|medium|low)", response_clean, flags=re.IGNORECASE)
        return match.group(1).lower() if match else ""

    def _is_high_risk_question(self, question):
        text = str(question).lower()
        high_risk_cues = (
            "how many", "number", "count", "color", "colour", "text", "word", "letter", "sign",
            "read", "say", "left", "right", "behind", "front", "next to", "where", "wearing",
            "holding", "doing", "mouth", "hand", "what is in", "what are in"
        )
        return any(cue in text for cue in high_risk_cues)

    def _should_run_reflective_review(self, question, response, current_answer):
        trigger = self.args.reflect_trigger_mode
        if trigger == "always":
            return True
        high_risk = self._is_high_risk_question(question)
        confidence = self._extract_reflective_confidence(response)
        low_confidence = confidence in ("low", "medium") or self._looks_like_visual_cue_list(current_answer)
        if trigger == "high_risk":
            return high_risk
        if trigger == "low_confidence":
            return low_confidence
        if trigger == "high_risk_or_low_confidence":
            return high_risk or low_confidence
        return True

    def _extract_direct_verify_answer(self, response, initial_answer):
        if self.args.direct_verify_policy == "conflict_only":
            first_lines = "\n".join(str(response).strip().splitlines()[:3]).lower()
            if "contradicted" not in first_lines:
                return self._clean_short_answer(initial_answer)
        final_answer = self._extract_structured_cot_answer(response)
        if self.args.disable_direct_verify_fallback or self.args.direct_verify_policy == "no_fallback":
            return final_answer
        if self._looks_like_visual_cue_list(final_answer):
            return self._clean_short_answer(initial_answer)
        return final_answer

    def _format_cot_answer_prompt(self, prompt_before_answer):
        if self.args.cot_style in ("direct_verify", "reviewer_evidence", "candidate_judge", "rag_strategy_router",
                                   "protected_reflective", "multi_strategy_router",
                                   "direct_rephrase_consistency"):
            return (
                "=== Please answer directly with a single word or short phrase:\n"
                "%s" % (prompt_before_answer)
            )
        if self.args.cot_style == "adaptive_reflective_answer_first":
            return (
                "=== Please answer first with a single word or short phrase:\n"
                "Do not explain yet. Also give a coarse confidence label.\n"
                "Output exactly in this format:\n"
                "Answer: <short answer>\n"
                "Confidence: high / medium / low"
            )
        if self.args.cot_style == "reflective_answer_first":
            return (
                "=== Please answer first with a single word or short phrase:\n"
                "Do not explain yet. The first response should contain only the answer.\n"
                "Answer:"
            )
        if self.args.cot_style == "compact":
            return (
                "=== Please use compact visual cues, then give the final answer:\n"
                "Use at most 3 short visual cues. Do not write long reasoning.\n"
                "The final answer must be a single word or short phrase.\n"
                "Output exactly in this format:\n"
                "Visual Cues: <cue1>; <cue2>; <cue3>\n"
                "Final Answer:"
            )
        if self.args.cot_style == "answer_first":
            return (
                "=== Please answer first, then add very brief visual cues:\n"
                "The first line must be the final answer as a single word or short phrase.\n"
                "Use at most 3 short visual cues after that. Do not write long reasoning.\n"
                "Output exactly in this format:\n"
                "Final Answer:"
            )
        if self.args.cot_style == "answer_first_locked":
            return (
                "=== Please answer first, then give very brief visual reasons:\n"
                "Give the answer before any reasoning. Do not revise it after giving reasons.\n"
                "The answer must be a single word or short phrase.\n"
                "Output exactly in this format:\n"
                "Answer: <short answer>\n"
                "Reasons:\n"
                "1. <visible reason>\n"
                "2. <visible reason>"
            )
        if self.args.cot_style == "visual_facts":
            return (
                "=== Please ground the answer with minimal visible facts:\n"
                "List at most 2 visible facts from the image that are directly relevant to the question.\n"
                "Do not use long reasoning. Then answer with a single word or short phrase.\n"
                "Output exactly in this format:\n"
                "Visible Facts:\n"
                "1. <directly relevant visible fact>\n"
                "2. <directly relevant visible fact>\n"
                "Answer:"
            )
        return (
            "=== Please think step by step, then provide your final answer:\n"
            "Let's think step by step.\n"
            "%s" % (prompt_before_answer)
        )

    def _format_round_state_context(self, state_history, max_rounds=4):
        if not state_history:
            return ""

        lines = [
            "Previous rounds produced the following non-authoritative evidence. "
            "Use it only when it is relevant to the current question."
        ]
        for state in state_history[-max_rounds:]:
            round_id = state.get("round_id", "?")
            instruction = state.get("instruction", "unknown")
            objects = state.get("selected_objects", [])
            object_text = ", ".join(objects) if objects else "none"
            lines.append(f"Round {round_id}: requested {instruction} evidence for [{object_text}].")

            scores = state.get("onion_scores")
            if scores:
                score_text = ", ".join(f"{k}:{v:.2f}" for k, v in scores.items())
                lines.append(f"Routing scores: {score_text}.")

            image_path = state.get("enhanced_image_path")
            if image_path:
                lines.append("Visual evidence: an enhanced image view was generated and used.")

            dyfo_evidence = state.get("dyfo_visual_evidence")
            if dyfo_evidence:
                lines.append("DyFo visual evidence: %s" % self._truncate_text(dyfo_evidence))

            caption_evidence = state.get("enhanced_caption")
            if caption_evidence:
                lines.append("Caption evidence: %s" % self._truncate_text(caption_evidence))

            knowledge_evidence = state.get("enhanced_knowledge")
            if knowledge_evidence:
                lines.append("Knowledge evidence: %s" % self._truncate_text(knowledge_evidence))

        return "\n".join(lines)

    def _make_round_state(self, round_idx, onion_instruction, enhance_image_path,
                          enhance_caption, enhance_knowledge, dyfo_visual_evidence, pred_answer,
                          final_score, pred_candidates, dyfo_decision_trace=None):
        meta = onion_instruction[2] if len(onion_instruction) > 2 and isinstance(onion_instruction[2], dict) else {}
        selected_objects = onion_instruction[1] if len(onion_instruction) > 1 else []
        state = {
            "type": "round_state",
            "round_id": int(round_idx) + 1 if round_idx is not None else None,
            "instruction": onion_instruction[0] if onion_instruction else None,
            "selected_objects": selected_objects,
            "onion_scores": meta.get("scores", {}),
            "onion_threshold": meta.get("threshold"),
            "enhanced_image_path": enhance_image_path,
            "enhanced_caption": enhance_caption,
            "enhanced_knowledge": enhance_knowledge,
            "dyfo_visual_evidence": dyfo_visual_evidence,
            "dyfo_decision_trace": dyfo_decision_trace,
            "pred_answer": pred_answer,
            "final_score": final_score,
            "pred_candidates": pred_candidates,
        }
        evidence_bits = []
        if enhance_image_path:
            evidence_bits.append("image")
        if enhance_caption:
            evidence_bits.append("caption")
        if enhance_knowledge:
            evidence_bits.append("knowledge")
        if dyfo_visual_evidence:
            evidence_bits.append("dyfo_visual")
        state["executed_evidence"] = evidence_bits
        state["evidence_summary"] = (
            "Round %s: requested %s on [%s]; executed evidence: %s; answer hypothesis: %s"
            % (
                state["round_id"],
                state["instruction"],
                ", ".join(selected_objects) if selected_objects else "none",
                ", ".join(evidence_bits) if evidence_bits else "none",
                pred_answer,
            )
        )
        return state

    def _format_regional_context(self, region_items, max_regions=None):
        if not region_items:
            return ""
        max_regions = max_regions or self.args.max_regional_captions
        lines = []
        for item in region_items[:max_regions]:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, (list, tuple)):
                obj_name = str(item[1]) if len(item) > 1 else "object"
                if len(item) > 3 and item[3]:
                    text = f"{obj_name}: {item[3]}"
                elif len(item) > 2 and item[2]:
                    text = f"{obj_name}: {', '.join(item[2]) if isinstance(item[2], list) else item[2]}"
                else:
                    text = obj_name
                if len(item) > 4 and item[4]:
                    text += f" {item[4]}"
            else:
                text = str(item)
            text = self._truncate_text(text, max_chars=220)
            if text:
                lines.append(text)
        if not lines:
            return ""
        return "Regional visual context:\n" + "\n".join(f"- {line}" for line in lines)

    def _region_object_names(self, region_items):
        names = []
        for item in region_items:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                names.append(str(item[1]))
        return names

    def _get_ocr_context(self, image_key, object_list=None):
        ocr_dict = getattr(self.dataset, "val_ocr_text", {}).get(image_key, {})
        if not ocr_dict:
            return ""
        if object_list:
            selected = []
            object_set = set(object_list)
            for obj, text in ocr_dict.items():
                if obj in object_set:
                    selected.append(text)
            if selected:
                return "OCR context: " + " ".join(selected)
        return "OCR context: " + " ".join(ocr_dict.values())

    def _query_qwen_global_caption(self, image_key, image_path, question):
        if image_key in self.qwen_global_caption_cache:
            return self.qwen_global_caption_cache[image_key]
        prompt = (
            "Describe only the visible facts in this image that may help answer the question. "
            "Be concise and do not answer the question.\n"
            f"Question: {question}"
        )
        caption = self._call_llm(prompt, image_path=image_path, max_new_tokens=self.args.qwen_caption_max_tokens)
        self.qwen_global_caption_cache[image_key] = caption
        return caption

    def _query_qwen_local_caption(self, image_key, image_path, question, object_list):
        object_key = "|".join(object_list or [])
        cache_key = (image_key, object_key)
        if cache_key in self.qwen_local_caption_cache:
            return self.qwen_local_caption_cache[cache_key]
        if not object_list:
            self.qwen_local_caption_cache[cache_key] = ""
            return ""
        prompt = (
            "Describe the specified object(s) in the image only if they are visible. "
            "Focus on details useful for the question and do not answer it.\n"
            f"Question: {question}\nObjects: {object_list}"
        )
        caption = self._call_llm(prompt, image_path=image_path, max_new_tokens=self.args.qwen_caption_max_tokens)
        self.qwen_local_caption_cache[cache_key] = caption
        return caption

    def _filter_thoughts_with_clip(self, key, thought):
        if not thought:
            return "", ""
        if not hasattr(self.dataset, "image_val_feature"):
            return thought, thought
        with torch.no_grad():
            img_id = self.dataset.valkey2idx[key]
            img_emb = torch.from_numpy(self.dataset.image_val_feature[img_id]).cuda().float().unsqueeze(dim=0)
            parts = [x.strip() for x in thought.split(".") if x.strip()]
            if not parts:
                return "", ""
            inputs = self.clip_processor(
                text=parts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            clip_outputs = self.clip_model(**inputs)
            thought_emb = clip_outputs["pooler_output"]
            thought_emb /= thought_emb.norm(dim=-1, keepdim=True)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            sim_cands = img_emb @ thought_emb.T
            kept = [parts[i] for i in range(len(parts)) if sim_cands[0, i].item() > self.args.verify_threshold]
            return ".".join(kept).strip() + ("." if kept else ""), ".".join(parts).strip() + "."

    def _filter_thoughts_with_qwen_image_check(self, image_path, thought):
        if not thought:
            return "", ""
        parts = [x.strip() for x in thought.split(".") if x.strip()]
        kept = []
        for part in parts[:self.args.max_thought_verify_sentences]:
            prompt = (
                "Check whether the following statement is directly supported by the image. "
                "Reply with only 'yes' or 'no'.\n"
                f"Statement: {part}"
            )
            reply = self._call_llm(prompt, image_path=image_path, max_new_tokens=8).strip().lower()
            if reply.startswith("yes"):
                kept.append(part)
        return ".".join(kept).strip() + ("." if kept else ""), ".".join(parts).strip() + "."
    
    def inference(self, save_every_step):
        # 推理主流程,遍历验证集样本,调用sample_inference_interactive进行推理
        answers = []
        full_answers = []
        # 如果需要每步保存结果,则创建输出目录
        if save_every_step:
            os.system("mkdir -p %s" % self.args.output_path)
            os.system("mkdir -p %s/prompt_samples" % self.args.output_path)
            os.system("mkdir -p %s/format_samples" % self.args.output_path)

        # 交互模式:用户输入图片id和问题,针对指定样本推理
        if self.args.pick_example_with_question_mode:
            while True:
                image_id = input("Input one image id please")
                question = input("Input one question please")
                image_id = str(image_id)
                self.given_question = question
                # 遍历验证集,找到匹配的key进行推理
                for idx, key in enumerate(tqdm(self.dataset.val_keys)):
                    if image_id not in key:
                        continue
                    final_answer, answer_list = self.sample_inference_interactive(key)
                    print(final_answer)
                    print(answer_list)
                    pdb.set_trace()

        # 遍历所有验证集样本,批量推理

        # # 短测试代码
        # i = 0
        shard_keys = [
            key for idx, key in enumerate(self.dataset.val_keys)
            if idx % self.args.num_shards == self.args.shard_id
        ]
        print(
            "[shard] shard_id=%s num_shards=%s assigned_samples=%s total_samples=%s"
            % (
                self.args.shard_id,
                self.args.num_shards,
                len(shard_keys),
                len(self.dataset.val_keys),
            )
        )
        
        for idx, key in enumerate(tqdm(self.dataset.val_keys)):

            # 数据分片：只处理属于当前shard的样本
            if idx % self.args.num_shards != self.args.shard_id:
                continue

            if self.args.max_samples_per_shard > 0 and len(answers) >= self.args.max_samples_per_shard:
                break

            print('----------inference----------processing sample %s/%s----------for loop----------' % (str(idx), str(len(self.dataset.val_keys))))

            # 如果已保存该样本结果则跳过
            if save_every_step:
                # 这里没有修改关于时间戳的内容
                out_file_name = "%s/prompt_samples/sample_%s_shard%s_*.json" % (
                    self.args.output_path,
                    str(idx),
                    str(self.args.shard_id),
                )
                print(out_file_name)
                out_file_list = glob.glob(out_file_name)
                if len(out_file_list) > 0:
                    continue
            # pick_example_mode下只处理特定样本
            if self.args.pick_example_mode:
                if not self.pick_example(key):
                    continue
            
            # 推理得到答案和详细推理过程
            # 这里是推理的核心代码
            final_answer, answer_list = self.sample_inference_interactive(key)
            answers.append(final_answer)
            full_answers.append(answer_list)
            if self.args.strategy_profile_output:
                image_key = int(key.split('<->')[0]) if self.args.dataset_name!="fvqa" else self.image_dict[key]
                profile_record = {
                    "key": key,
                    "image_id": image_key,
                    "question": self.dataset.question_dict[key],
                    "question_type": self._classify_vqa_question_type(self.dataset.question_dict[key]),
                    "strategy": self.args.strategy_name,
                    "pred_answer": final_answer[1],
                    "score": float(final_answer[3]),
                    "split": self.args.split_name,
                }
                with open(self.args.strategy_profile_output, "a") as f:
                    f.write(json.dumps(profile_record, ensure_ascii=False) + "\n")
            print('-----inference-----processing-----answer-----beg')
            print(final_answer)
            print(answer_list)
            print('-----inference-----processing-----answer-----end')
            print()
            # 计算当前准确率
            acc = 0.
            for answer in answers:
                acc += float(answer[3])
            print(acc * 100. / len(answers), len(answers))
            # 保存最新推理结果到json文件
            if save_every_step:
                json.dump(answers[-1], open("%s/prompt_samples/sample_%s_shard%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(self.args.shard_id), str(float(answers[-1][3]))), 'w'))
                json.dump(full_answers[-1], open("%s/format_samples/sample_%s_shard%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(self.args.shard_id), str(float(answers[-1][3]))), 'w'))
            
            # # 短测试代码
            # i += 1
            # if i > 20:
            #     break

        # 返回所有推理结果
        return answers, full_answers
    
    # 单个样本推理代码-交互式
    def sample_inference_interactive(self, key):

        # 获取图片id（支持fvqa特殊处理）
        image_key = int(key.split('<->')[0]) if self.args.dataset_name!="fvqa" else self.image_dict[key] # for fvqa
        # 加载原始图片
        raw_image = self.dataset.find_image(image_key)

        self.current_blip2_image = raw_image
        # 这个位置获取了current_blip2_image,处理图片的信息

        # debug模式下记录时间
        if self.args.debug:
            t1=time.time()

        # 加载场景图（属性信息）
        scene_graph_path = os.path.join(self.dataset.sg_attr_dir, str(image_key).zfill(12) + ".json")

        # 这个位置获取scene_graph_attr,加载场景图信息
        if getattr(self.args, "dataset_name", "") == "mme":
            scene_graph_attr = [[]]
        elif os.path.isfile(scene_graph_path):
            scene_graph_attr = json.load(open(scene_graph_path))
        else:
            scene_graph_attr = [[]]
        
        # 如果采用caption策略，则加载或生成场景图对应的文本描述（caption）
        if self.args.iterative_strategy == "caption":
            # 构建场景图描述文件的存储目录路径
            self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)
            # 根据图像键值（image_key）构造完整的json文件路径（文件名补齐12位数字，不足前补零）
            scene_path = os.path.join(self.sg_cap_dir, str(image_key).zfill(12) + ".json")
            # 如果上述路径不存在对应的caption文件，则尝试从备份目录（_v2后缀）加载
            if not os.path.isfile(scene_path):
                scene_path = os.path.join(self.sg_cap_dir + "_v2", str(image_key).zfill(12) + ".json")
            # 若备份路径仍不存在文件，则根据传入的场景图属性（scene_graph_attr[0]）动态生成caption
            if not os.path.isfile(scene_path):
                # 生成格式：每个属性对象转为 "类别 is 属性1, 属性2, ..." 的句子
                scene_graph_caption = [f"{attr['class']} is {', '.join(attr['attr'])}." \
                                        for attr in scene_graph_attr[0]]
            else:
                # 如果找到了caption文件，则从json文件中加载预生成的描述
                scene_graph_caption = json.load(open(scene_path))

        # 构建属性列表（包含置信度、类别、属性、caption/ocr文本）
        attr_list = []
        # 场景图[0]号信息是场景中的所有物体及其关系属性
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if self.args.iterative_strategy == "caption":
                if isinstance(scene_graph_caption, list):
                    tmp_cap = scene_graph_caption[attr_id]
                else:
                    rect_str = str(attr['rect'])
                    try:
                        tmp_cap = scene_graph_caption[rect_str]
                    except:
                        tmp_cap = attr['class']
                        print("Fail to parse attr\n")
                tmp_attr = [attr['conf'], attr['class'], attr['attr'], tmp_cap]
            else:
                tmp_attr = [attr['conf'], attr['class'], attr['attr']]
            if self.args.caption_type == "vinvl_ocr":
                ocr_for_image = self.val_ocr_text.get(image_key, {})
                if attr['class'] in ocr_for_image:
                    tmp_attr.append(ocr_for_image[attr['class']])
                else:
                    tmp_attr.append("")
            attr_list.append(tmp_attr)
        # 按置信度降序排序属性
        # attr_list元素格式:[置信度, 类别, 属性, (可选)caption, (可选)ocr文本]
        attr_list.sort(key=lambda x: x[0], reverse=True)

        # 初始化推理相关变量
        answer_list = []
        noticed_caption_list = []
        thoughts = []

        # debug模式下记录准备时间
        if self.args.debug:
            t2=time.time()
            print("    PREPARE TIME", t2-t1)

        # 新增一个全局的对话模块，增加长上下文对话一体化

        # 初始化对话历史
        self.current_conversation = []

        # 对单个数据样本，初始化其关心个体
        self.attention_object = []
        round_state_history = []

        # all-regional模式一次性注入更多区域信息，避免多轮重复消耗同一批对象。
        rounds = 1 if self.args.use_all_regional_captions else self.args.rounds
        for i in range(rounds):
            # debug模式下记录时间
            if self.args.debug:
                t3=time.time()

            if self.args.use_all_regional_captions:
                idx_list = list(range(min(len(attr_list), self.args.max_regional_captions)))
                object_list = list(dict.fromkeys([attr[1] for attr in attr_list[:self.args.max_regional_captions]]))
            else:
                # 使用自己写的挑选关注目标的方法
                idx_list, object_list = self.init_attention_object(key, attr_list, self.dataset.find_image_path(image_key))
            # idx = idx_list[0]

            # 同步GPU
            torch.cuda.synchronize()
            # debug模式下记录时间
            if self.args.debug:
                t4=time.time()

            # # 补充所有的caption
            # for i in idx_list:
            #     noticed_caption_list.append(attr_list[i])
            # # BLIP2模式下生成局部caption
            # noticed_caption_list.append(attr_list[idx])

            if self.args.use_all_regional_captions:
                noticed_caption_list = attr_list[:self.args.max_regional_captions]
            else:
                for idx in idx_list:
                    if idx is not None and 0 <= idx < len(attr_list):
                        noticed_caption_list.append(attr_list[idx])
                # 保留现有Qwen caption作为全局描述。
                noticed_caption_list.append(self.caption_qwen.get(str(image_key), ""))

            # onion指令阶段
            self.messages = None
            onion_instruction, self.messages = self.onion_make_instruction(key, object_list)

            # 推理（传入历史思考链，让后续轮次有信息增量）
            current_answer = self.sample_inference(
                key, attr_list, noticed_caption_list,
                thoughts_list=thoughts,
                onion_instruction=onion_instruction,
                round_idx=i,
                state_history=round_state_history
            )
            answer_list.append(current_answer)

            round_state = current_answer[-1] if (
                isinstance(current_answer, list)
                and len(current_answer) > 0
                and isinstance(current_answer[-1], dict)
                and current_answer[-1].get("type") == "round_state"
            ) else None
            if round_state is not None:
                round_state_history.append(round_state)

            # debug模式下记录时间
            if self.args.debug:
                t5=time.time()
                print("    VISUAL LOOP TIME", t4-t3)
                print("    REASON LOOP TIME", t5-t4)
            # 同步GPU
            torch.cuda.synchronize()
            
            # 移除本轮已关注的物体，避免下一轮重复选择
            if not self.args.use_all_regional_captions:
                for idx in sorted(idx_list, reverse=True):
                    if idx is not None and 0 <= idx < len(attr_list):
                        attr_list.pop(idx)
            # 记录本轮增强信息，供后续轮次了解已做工作（不注入答案避免锚定偏差）
            if round_state is not None:
                enhancement_desc = round_state["evidence_summary"]
            else:
                enhanced_objects = ", ".join(object_list) if object_list else "none"
                enhancement_desc = f"Round {i+1}: requested {onion_instruction[0]} on [{enhanced_objects}]"
            thoughts.append(enhancement_desc)

        # 跨轮多数投票：取出现次数最多的答案作为最终结果
        answers_text = [ans[1] for ans in answer_list]
        majority_answer = max(set(answers_text), key=answers_text.count)
        for ans in answer_list:
            if ans[1] == majority_answer:
                final_answer = ans
                break
        return final_answer, answer_list

    def init_attention_object(self, key, attr_list, image_path, ban_option=[]):
        '''
        直接提出问题,询问
        "对于给定的问题和图像,下面哪些选项是你应该关注的?"
        '''

        # 补充内容：问图像的内容作为补充；告知‘不能选这个东西’

        # 1. 获取当前问题
        question = self.dataset.question_dict[key]

        # 2. 准备当前问题的候选对象列表 这里使用了去重
        obj_list = [obj[1] for obj in attr_list][:25]  # 从attr_list中提取对象名称，只要前25个
        unique_obj_list = list(dict.fromkeys(obj_list))
        if not unique_obj_list:
            return [], []

        # 设置要选择的实体数量
        n_select = min(3, len(unique_obj_list))  # 默认选择3个，但不超过选项总数

        # prompt
        # # 根据提供的图像和问题,从下面选项中选择n个最应该关注的实体.<图像和问题>
        # prompt = f"Based on the provided image and question, select {n_select} entities from the options below that should be the most focused on.\n"
        # 根据提供的问题,从下面选项中选择n个最应该关注的实体.<仅问题>
        # 为每个选项分配字母标签
        option_labels = [chr(65 + i) for i in range(len(unique_obj_list))]  # A, B, C, D, ...
        options_with_labels = [f"{label}. {option}" for label, option in zip(option_labels, unique_obj_list)]

        prompt = f"Based on the provided question, select {n_select} entities from the options below that should be the most focused on.\n"
        prompt += f"Question: {question}\n"
        prompt += f"Options: {options_with_labels}\n"  # 直接打印列表
        prompt += f"\nPlease select the top-{n_select} most relevant entities (output only the letters, e.g., ['A', 'C']):"

        if image_path:
            response = self._call_llm(prompt, image_path=image_path)
        else:
            response = self._call_llm(prompt)

        response_list = string_to_list_if_possible(response)

        # deepseek代码
        # 获取response_list（字母列表）
        response_list = string_to_list_if_possible(response)  # 例如 ['A', 'C']

        # 创建字母到对象的映射字典
        letter_to_object = {chr(65 + i): obj for i, obj in enumerate(unique_obj_list)}

        # 获取选中的对象
        selected_objects = [letter_to_object[letter] for letter in response_list if letter in letter_to_object]

        # 获取在原始obj_list中的索引（第一次出现的位置）
        original_indices = []
        for obj in selected_objects:
            try:
                idx = obj_list.index(obj)
                original_indices.append(idx)
            except ValueError:
                original_indices.append(-1)  # 如果找不到，返回-1

        # print(f"Selected objects: {selected_objects}")
        # print(f"Indices in original list: {original_indices}")
        # deepseek代码

        print()
        print('-----init_attention_object-----相关信息-----+++++-----beg')
        print('prompt:', prompt)
        print('unique_obj_list:', unique_obj_list)
        print('response_list:', response_list)
        print('selected_objects:', selected_objects)
        print('original_indices:', original_indices)
        print('-----init_attention_object-----相关信息-----+++++-----end')
        print()

        # # 检查回答，有额外选项的重新跑一遍
        # if self.check_answer(response_list, obj_list, ban_option) == False:
        #     response_list = self.init_attention_object(key, attr_list, image_path, ban_option=ban_option)

        # 返回的是初始选项，和选中的对象列表
        return original_indices, selected_objects
    

    # 检测init_attention_object回复是否规范的函数，主要检查是不是按要求选择的
    def check_answer(self, response_list, obj_list, ban_option):
        
        result = True
        for i in response_list:
            if i not in obj_list and i not in ban_option:
                result = False
                ban_option.append(i)
                break
        
        return result

    # 将选项列表转换为格式化的文本字符串，并返回正确选项的内容。
    def make_choices_text(self, choices, answer):
        """
        将选项列表转换为格式化的文本字符串，并返回正确选项的内容。
        
        该函数接收一个选项列表和正确答案的索引，生成两种输出：
        1. 所有选项拼接成的字符串，格式如："选项1, 选项2, 选项3."
        2. 根据索引从选项列表中提取的正确答案内容
        
        Args:
            choices (list): 包含所有选项的列表，每个元素为字符串形式的选项内容
            answer (int): 正确答案在choices列表中的索引位置（从0开始计数）
        
        Returns:
            tuple: 包含两个元素的元组
                - str: 所有选项用逗号拼接并末尾加点的字符串，如："A. 苹果, B. 香蕉, C. 橙子."
                - str: 正确答案对应的选项内容
                
        Example:
            >>> options = ['A. 苹果', 'B. 香蕉', 'C. 橙子']
            >>> make_choices_text(options, 1)
            ('A. 苹果, B. 香蕉, C. 橙子.', 'B. 香蕉')
        
        Note:
            - 函数使用f-string格式化输出,确保选项字符串后有一个点号结尾
            - choices[answer]直接返回原始选项内容，不会添加额外格式
        """
        return f"{', '.join(choices)}.", choices[answer]
    
    # 针对单样本的核心推理代码
    # key, 场景图属性, 思考历史
    def sample_inference(self, key, attr_list, scene_graph_attr, thoughts_list=None,
                         onion_instruction=[None, ], round_idx=None, state_history=None):

        # onion_instruction[0] 已经给出的下一步的指令
        # onion_instruction[1] 已经给出的下一步指令的对象

        # 补充：这段代码是 Chain-of-Thought (CoT) 推理步骤的后处理与验证模块，主要功能是用 CLIP 模型筛选高质量的推理步骤。

        # 获取图片id
        image_key = int(key.split('<->')[0]) if self.args.dataset_name!="fvqa" else self.image_dict[key] # for fvqa
        # 获取图片路径
        image_path = self.dataset.find_image_path(image_key)
        # 是否随机选择caption
        if self.args.random_caption:
            random.seed(image_key) # keep random context in every step of the same sample consistent
        # 获取问题、答案、caption
        question = self.dataset.question_dict[key]
        answer = self.dataset.answer_dict[key]
        caption = self.dataset.inputtext_dict.get(image_key, [""])[0]
        # caption += ' '
        # print(type(caption))
        # print(type(self.caption_qwen[str(image_key)]))
        qwen_caption = self.caption_qwen.get(str(image_key), caption)
        simplified_caption_prompt = 'Please organize the parts relevant to the question from the given description.\n'
        simplified_caption_prompt += 'If no valid relevant information is available, please reply with "None".'
        simplified_caption_prompt += 'Question: %s\n' % question
        simplified_caption_prompt += 'Description: %s\n' % qwen_caption
        simplified_qwen_caption = self._call_llm(
            simplified_caption_prompt
        )


        data_row = {
            'key' : key,
            'image_key' : image_key,
            'question' : question,
            'answer' : answer,
            'caption' : caption,
            'image_path' : image_path,
            'qwen_caption' : qwen_caption
        }
        regional_context = self._format_regional_context(scene_graph_attr)
        ocr_context = self._get_ocr_context(image_key, onion_instruction[1] if len(onion_instruction) > 1 else None) \
            if self.args.use_ocr_context else ""
        qwen_global_caption = ""
        qwen_local_caption = ""
        if self.args.use_qwen_blip2_caption:
            if self.args.qwen_caption_mode in ("both", "global"):
                qwen_global_caption = self._query_qwen_global_caption(image_key, image_path, question)
            if self.args.qwen_caption_mode in ("both", "local"):
                qwen_local_caption = self._query_qwen_local_caption(
                    image_key, image_path, question,
                    onion_instruction[1] if len(onion_instruction) > 1 else []
                )
        print('-----sample_inference-----样本相关信息-----+++++-----beg')
        print('image id:', image_key)
        print('question:', question)
        print('answer:', answer)
        print('caption:', caption)
        print('-----sample_inference-----样本相关信息-----+++++-----end')
        print()

        # 选择特定问题进行推理
        if self.args.pick_example_mode:
            question = self.temp_question
        if self.args.pick_example_with_question_mode:
            question = self.given_question
        if self.args.random_caption:
            caption = random.choice(list(self.dataset.traincontext_caption_dict.values()))

        # 推理相关变量初始化
        thought_list, all_thought_list = [], []
        # 检索构建few-shot prompt所需的训练示例 key（按相似度）
        context_key_list = self.get_context_keys(key, self.args.similarity_metric, self.args.n_shot * self.args.n_ensemble)

        # onion指示操作区
        enhance_image_path = None
        enhance_caption = None
        enhance_knowledge = None
        dyfo_visual_evidence = ""
        dyfo_focus_image_path = None
        dyfo_final_answer = ""
        dyfo_decision_trace = None
        dyfo_evidence_enabled = self.args.use_dyfo_visual_evidence or self.args.mcts_action_mode == "dyfo_evidence"
        question_type = self._classify_vqa_question_type(question)
        multi_strategy_route = None
        if self.args.cot_style == "multi_strategy_router":
            multi_strategy_route = self._route_with_multi_strategy_profile(key, question)
        selective_evidence_kinds = {"caption"}
        if self.args.cot_style == "reviewer_evidence" and self.args.reviewer_evidence_scope == "selective":
            selective_evidence_kinds = self._selective_reviewer_evidence_kinds(question)
            self._current_selective_evidence_kinds = selective_evidence_kinds
            print('-----selective_reviewer_evidence-----触发证据-----+++++-----beg')
            print('selective_evidence_kinds:', sorted(selective_evidence_kinds))
            print('-----selective_reviewer_evidence-----触发证据-----+++++-----end')
            print()

        effective_use_image_enhance = self.args.use_image_enhance
        effective_use_caption_enhance = self.args.use_caption_enhance
        effective_use_knowledge_enhance = self.args.use_knowledge_enhance
        if self.args.cot_style == "reviewer_evidence" and self.args.reviewer_evidence_scope == "selective":
            effective_use_image_enhance = self.args.use_image_enhance and "image" in selective_evidence_kinds
            effective_use_caption_enhance = "caption_enhance" in selective_evidence_kinds
            effective_use_knowledge_enhance = "knowledge" in selective_evidence_kinds
        if self.args.cot_style == "candidate_judge" and self.args.candidate_judge_route_evidence:
            effective_use_image_enhance = self.args.use_image_enhance and question_type in ("text_ocr", "visual_detail", "category")
            effective_use_caption_enhance = self.args.use_caption_enhance and question_type in ("visual_detail", "category", "general")
            effective_use_knowledge_enhance = self.args.use_knowledge_enhance and question_type in ("knowledge", "category")
        if self.args.cot_style == "multi_strategy_router":
            selected_strategy = (multi_strategy_route or {}).get("strategy", self.args.multi_strategy_default)
            effective_use_image_enhance = self.args.use_image_enhance and selected_strategy == "marker_mcts"
            effective_use_caption_enhance = False
            effective_use_knowledge_enhance = False
        selective_mode = self.args.cot_style == "reviewer_evidence" and self.args.reviewer_evidence_scope == "selective"
        knowledge_triggered = onion_instruction[0] == 'knowledge' or selective_mode
        if self.args.knowledge_enhance_trigger == "always":
            knowledge_triggered = True
        elif self.args.knowledge_enhance_trigger == "knowledge_qtype":
            knowledge_triggered = question_type in ("knowledge", "category")
        
        # ========== 三个核心增强模块（由args控制开关） ==========
        if effective_use_image_enhance and onion_instruction[0] == 'image':
            if self.args.mcts_action_mode == "dyfo_evidence":
                dyfo_result = self._run_dyfo_visual_evidence_search(data_row, onion_instruction[1], attr_list)
                dyfo_visual_evidence = dyfo_result.get("evidence", "")
                dyfo_focus_image_path = dyfo_result.get("focus_image_path")
                dyfo_final_answer = dyfo_result.get("final_answer", "")
                dyfo_decision_trace = dyfo_result.get("decision_trace")
                self.last_dyfo_visual_evidence = dyfo_visual_evidence
                self.last_dyfo_focus_image_path = dyfo_focus_image_path
                if self.args.dyfo_use_focus_image_as_answer and dyfo_focus_image_path:
                    enhance_image_path = dyfo_focus_image_path
                print('-----enhance_image-----DyFo visual evidence已生成-----')
                print('dyfo_visual_evidence:', dyfo_visual_evidence)
            else:
                enhance_image_path = self.enhance_image_object(data_row, onion_instruction[1], attr_list)
                print('-----enhance_image-----MCTS增强图像已生成-----')

        if effective_use_caption_enhance and (onion_instruction[0] == 'caption' or selective_mode):
            enhance_caption = self.enhance_caption_object(data_row, onion_instruction[1], attr_list)
            print('-----enhance_caption-----强化的针对目标描述-----+++++-----beg')
            print('enhance_caption:', enhance_caption)
            print('-----enhance_caption-----强化的针对目标描述-----+++++-----end')
            print()

        if effective_use_knowledge_enhance and knowledge_triggered:
            enhance_knowledge = self.enhance_knowledge_object(data_row, onion_instruction[1], attr_list)
            print('-----enhance_knowledge-----强化的针对目标知识-----+++++-----beg')
            print('enhance_knowledge:', enhance_knowledge)
            print('-----enhance_knowledge-----强化的针对目标知识-----+++++-----end')
            print()

        print('-----onion_instruction-----类别输出指示-----+++++-----beg')
        print('onion_instruction:', onion_instruction)
        if effective_use_caption_enhance and (onion_instruction[0] == 'caption' or selective_mode):
            print('enhance_caption:', enhance_caption)
        if effective_use_knowledge_enhance and knowledge_triggered:
            print('enhance_knowledge:', enhance_knowledge)
        print('-----onion_instruction-----类别输出指示-----+++++-----end')
        print()

        # 暂定修改：如何选择答案？如何收集答案？
        # 先在循环前建一个空列表，收集所有答案
        pred_candidates = []

        # 进行多次采样集成
        for repeat in range(self.args.n_ensemble):

            # 修改测试模式
            if self.args.debug:
                t1=time.time()

            # 根据引擎选择构建prompt
            prompt_before_answer = "Answer: The answer is"
            # prompt = 'Please answer the question based on the context, using a single word or short phrase. Below is an example for you:\n'
            prompt = 'Please answer the question based on the context, using a single word or short phrase. \n'

            ## prompt format following GPT-3 QA API
            # 根据GPT3的prompt格式构建提示语
            # 这一句有对提示语构建的更改
            if self.args.context_mode == "empty" or self.args.remove_caption:
                cur_caption = ""
            elif self.args.context_mode == "objects_only":
                object_names = onion_instruction[1] if len(onion_instruction) > 1 else []
                cur_caption = "Selected visual objects: " + ", ".join(object_names) if object_names else ""
            else:
                cur_caption = caption
            direct_answer_context = cur_caption

            # 获取上下文的训练示例
            for ni in range(self.args.n_shot):
                # 初始化上下文list
                if context_key_list is None:
                    if not self.train_keys:
                        raise ValueError("No train context keys are available for few-shot prompting.")
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                else:
                    context_key = context_key_list[ni + self.args.n_shot * repeat]

                # 确保获取有效的上下文，找出样例中所有内容都不为空的部分
                while True:  ## make sure get context with valid question and answer
                    if self.args.choice_only or (len(self.dataset.traincontext_question_dict[context_key]) != 0 and len(
                            self.dataset.traincontext_answer_dict[context_key][0]) != 0):
                        break
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                image_context_key = int(context_key.split('<->')[0]) if self.args.dataset_name!="fvqa" else self.image_dict[context_key] # for fvqa

                # 获取问题、答案、caption
                if self.args.random_caption:
                    context_caption = random.choice(list(self.dataset.traincontext_caption_dict.values()))
                    context_caption = random.choice(context_caption)
                elif self.args.remove_caption:
                    context_caption = ""
                else:
                    context_caption = self.dataset.traincontext_caption_dict[image_context_key][
                              random.randint(0, len(self.dataset.traincontext_caption_dict[image_context_key]) - 1)]

                # 组装
                # prompt += '===Example context:\n'
                # prompt += 'Context: %s\n' % (context_caption)

                if self.args.choice_only:
                    choice_text, answer_text = self.make_choices_text(self.traincontext_choices_dict[context_key],
                                                                      self.dataset.traincontext_answer_dict[context_key])
                    choice_text = f"\nChoices: {choice_text}"
                else:
                    choice_text = ""
                    answer_text = self.dataset.traincontext_answer_dict[context_key][0]
                    #if self.args.dataset_name !="fvqa" else self.dataset.traincontext_answer_dict[context_key]

                # if self.args.chain_of_thoughts:
                #     rationale_text = self.dataset.traincontext_rationale_dict[context_key][0]
                #     #if self.args.dataset_name !="fvqa" else self.dataset.traincontext_rationale_dict[context_key]
                #     prompt += 'Question: %s%s\n%s %s. %s\n\n===\n' % (self.dataset.traincontext_question_dict[context_key],
                #                                                              choice_text, prompt_before_answer, answer_text, rationale_text)
                # else:
                #     prompt += '===Example question and answer:\n'
                #     prompt += 'Question: %s%s\n%s %s\n' % (
                #     self.dataset.traincontext_question_dict[context_key], choice_text, prompt_before_answer, answer_text)

            # COT结合部分的内容
            state_context = "" if self.args.context_mode in ("caption_only", "objects_only", "empty", "no_round_state") \
                else self._format_round_state_context(state_history)
            if state_context:
                cur_caption += "\n"
                cur_caption += state_context
            elif self.args.context_mode not in ("caption_only", "objects_only", "empty", "no_round_state") \
                    and thoughts_list is not None and len(thoughts_list) > 0:
                cur_thoughts_list = [th for th in thoughts_list if th != '']
                if len(cur_thoughts_list) > 0:
                    cur_caption += "\n"
                    cur_caption += " ".join(cur_thoughts_list)

            if self.args.use_all_regional_captions and regional_context:
                cur_caption += "\n" + regional_context
            if self.args.use_ocr_context and ocr_context:
                cur_caption += "\n" + ocr_context
            if self.args.use_qwen_blip2_caption and not self.args.qwen_caption_no_final_context:
                if qwen_global_caption:
                    cur_caption += "\nQwen visual caption: " + self._truncate_text(qwen_global_caption, self.args.qwen_caption_final_max_chars)
                if qwen_local_caption:
                    cur_caption += "\nQwen local visual caption: " + self._truncate_text(qwen_local_caption, self.args.qwen_caption_final_max_chars)

            # 选择题
            if self.args.choice_only:
                choice_text, _ = self.make_choices_text(self.choices_dict[key], 0)
                choice_text = f"\nChoices: {choice_text}"
            else:
                choice_text = ""

            # 增强的caption和knowledge注入brief context
            if self.args.use_caption_enhance and enhance_caption:
                cur_caption += '\n' + enhance_caption
            if self.args.use_knowledge_enhance and enhance_knowledge:
                cur_caption += '\n' + enhance_knowledge
            if dyfo_evidence_enabled and dyfo_visual_evidence:
                cur_caption += '\nDyFo visual evidence: ' + self._truncate_text(
                    dyfo_visual_evidence, self.args.dyfo_evidence_context_max_chars
                )
            if self.args.direct_prompt_style == "context_gated":
                cur_caption = self._build_direct_context_for_style(
                    question, caption, regional_context, ocr_context
                )
                if dyfo_evidence_enabled and dyfo_visual_evidence:
                    cur_caption += '\nDyFo visual evidence: ' + self._truncate_text(
                        dyfo_visual_evidence, self.args.dyfo_evidence_context_max_chars
                    )
            prompt_context = direct_answer_context if self.args.cot_style == "reviewer_evidence" else cur_caption

            # 上下文参考
            prompt += '===The context you need to refer to:\n'
            prompt += 'Brief Context: %s\n' % prompt_context
            # # Detailed Context：优先使用caption增强结果，否则使用Qwen精简描述
            # detailed_context = enhance_caption if (self.args.use_caption_enhance and enhance_caption) else simplified_qwen_caption
            # prompt += 'Detailed Context: %s\n===\n' % detailed_context


            # 问题和答案
            prompt += '===The question you need to answer:\n'
            prompt += 'Question: %s%s\n' % (question, choice_text)
            if self.args.chain_of_thoughts:
                if self.args.cot_style == "complex_decompose":
                    prompt += '=== Please answer directly with a single word or short phrase:\n'
                    prompt += '%s' % (prompt_before_answer)
                else:
                    prompt += self._format_cot_answer_prompt(prompt_before_answer)
            else:
                prompt += self._format_direct_answer_instruction(question, prompt_before_answer)

            print('-----sample_inference-----n_shot prompt-----+++++-----beg')
            print(prompt)
            print('-----sample_inference-----n_shot prompt-----+++++-----end')
            print()

            # debug模式下记录时间
            if self.args.debug:
                t2=time.time()
            
            # Qwen模型的推理过程
            if 'qwen' in self.args.engine:

                # 增强图像判断
                answer_image_path = image_path
                if self.args.cot_style != "reviewer_evidence" and enhance_image_path:
                    answer_image_path = enhance_image_path
                reviewer_image_path = image_path
                if enhance_image_path and not self.args.reviewer_disable_enhanced_image:
                    reviewer_image_path = enhance_image_path

                # # 长上下文一体对话模块修改
                # response, self.messages = self._call_llm(prompt, image_path=image_path, history=self.messages, return_history=True)
                # 获取响应
                response = self._call_llm(prompt, image_path=answer_image_path)

                if self.args.chain_of_thoughts:
                    if self.args.cot_style == "multi_strategy_router":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        route = multi_strategy_route or self._route_with_multi_strategy_profile(key, question)
                        selected_strategy = route["strategy"]
                        runtime_image_path = answer_image_path
                        if selected_strategy == "marker_mcts" and enhance_image_path:
                            runtime_image_path = enhance_image_path

                        if selected_strategy == "direct":
                            extracted_answer = initial_answer
                            runtime_trace = "Direct Runtime\nInitial Response:\n%s\nFinal Answer: %s" % (
                                response, extracted_answer
                            )
                        elif selected_strategy == "reflective_r3":
                            extracted_answer, runtime_trace = self._run_reflective_r3_runtime(
                                question, choice_text, cur_caption, runtime_image_path
                            )
                        elif selected_strategy == "answer_first_no_caption":
                            extracted_answer, runtime_trace = self._run_answer_first_locked_runtime(
                                question, choice_text, runtime_image_path
                            )
                        elif selected_strategy == "marker_mcts":
                            marker_prompt = (
                                "Answer the visual question with a single word or short phrase.\n"
                                "Use the marked image if a marker is visible; the marker is only a visual hint, not an answer.\n"
                                "Brief Context: %s\n"
                                "Question: %s%s\n"
                                "Answer:"
                            ) % (cur_caption, question, choice_text)
                            marker_response = self._call_llm(marker_prompt, image_path=runtime_image_path)
                            extracted_answer = self._clean_short_answer(self._extract_answer_from_response(marker_response))
                            runtime_trace = (
                                "Marker MCTS Runtime\nEnhanced Image: %s\nPrompt:\n%s\nResponse:\n%s\nFinal Answer: %s"
                            ) % (enhance_image_path, marker_prompt, marker_response, extracted_answer)
                        else:
                            extracted_answer = initial_answer
                            runtime_trace = "Unknown selected strategy %s; fallback direct.\nFinal Answer: %s" % (
                                selected_strategy, extracted_answer
                            )

                        avg_text = ", ".join(
                            "%s:%.3f" % (name, val)
                            for name, val in sorted(route.get("strategy_avgs", {}).items())
                        )
                        response = (
                            "Multi Strategy Router: %s\n"
                            "Route Reason: %s\n"
                            "Route Averages: %s\n"
                            "Default Avg: %.3f\n"
                            "Best Avg: %.3f\n"
                            "Initial Direct Answer: %s\n"
                            "%s"
                        ) % (
                            selected_strategy,
                            route.get("reason", ""),
                            avg_text,
                            route.get("default_avg", 0.0),
                            route.get("best_avg", 0.0),
                            initial_answer,
                            runtime_trace,
                        )
                    elif self.args.cot_style == "protected_reflective":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        review_prompt = self._format_protected_review_prompt(
                            cur_caption, question, choice_text, initial_answer
                        )
                        review_response = self._call_llm(review_prompt, image_path=answer_image_path)
                        extracted_answer = self._extract_protected_review_answer(review_response, initial_answer)
                        response = (
                            "Initial Direct Answer: %s\n"
                            "Protected Review Prompt:\n%s\n"
                            "Protected Review Response:\n%s\n"
                            "Final Answer: %s"
                        ) % (initial_answer, review_prompt, review_response, extracted_answer)
                    elif self.args.cot_style == "rag_strategy_router":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        route = self._route_with_strategy_profile(key, question)
                        selected_strategy = route["strategy"]
                        if selected_strategy == self.args.strategy_cot_name:
                            if self.args.strategy_cot_runtime == "complex_decompose":
                                decomp = self._run_complex_decompose_from_direct(
                                    question, choice_text, cur_caption, initial_answer,
                                    question_type, answer_image_path
                                )
                                extracted_answer = decomp["final_answer"]
                                response = (
                                    "RAG Strategy Router: %s\n"
                                    "Router Mode: %s\n"
                                    "Route Stats: direct_avg=%.3f cot_avg=%.3f direct_hard_rate=%.3f "
                                    "complex_win_rate=%.3f rescue_rate=%.3f damage_rate=%.3f reason=%s\n"
                                    "Initial Direct Answer: %s\n"
                                    "Should Decompose: %s\n"
                                    "Decompose Prompt:\n%s\n"
                                    "Decompose Response:\n%s\n"
                                    "Decomposed Answer: %s\n"
                                    "Verify Enabled: %s\n"
                                    "Verify Prompt:\n%s\n"
                                    "Verify Response:\n%s\n"
                                    "Final Answer: %s"
                                ) % (
                                    selected_strategy, self.args.strategy_router_mode,
                                    route.get("direct_avg", 0.0), route.get("cot_avg", 0.0),
                                    route.get("direct_hard_rate", 0.0), route.get("complex_win_rate", 0.0),
                                    route.get("rescue_rate", 0.0), route.get("damage_rate", 0.0),
                                    route.get("reason", ""), initial_answer,
                                    decomp["should_decompose"], decomp["decompose_prompt"],
                                    decomp["decompose_response"], decomp["decomposed_answer"],
                                    self.args.decompose_verify, decomp["verify_prompt"],
                                    decomp["verify_response"], extracted_answer,
                                )
                            elif self.args.strategy_cot_runtime == "answer_first_locked":
                                cot_prompt = self._format_candidate_prompt(
                                    question, choice_text, cur_caption, "answer_first_locked", question_type
                                )
                                cot_response = self._call_llm(cot_prompt, image_path=answer_image_path)
                                extracted_answer = self._extract_first_answer_line(cot_response)
                                response = (
                                    "RAG Strategy Router: %s\n"
                                    "Router Mode: %s\n"
                                    "Route Stats: direct_avg=%.3f cot_avg=%.3f rescue_rate=%.3f damage_rate=%.3f reason=%s\n"
                                    "Initial Direct Answer: %s\n"
                                    "CoT Prompt:\n%s\n"
                                    "CoT Response:\n%s\n"
                                    "Final Answer: %s"
                                ) % (
                                    selected_strategy, self.args.strategy_router_mode,
                                    route.get("direct_avg", 0.0), route.get("cot_avg", 0.0),
                                    route.get("rescue_rate", 0.0), route.get("damage_rate", 0.0),
                                    route.get("reason", ""), initial_answer, cot_prompt, cot_response,
                                    extracted_answer,
                                )
                            else:
                                review_prompt = self._format_protected_review_prompt(
                                    cur_caption, question, choice_text, initial_answer
                                )
                                review_response = self._call_llm(review_prompt, image_path=answer_image_path)
                                extracted_answer = self._extract_protected_review_answer(review_response, initial_answer)
                                response = (
                                    "RAG Strategy Router: %s\n"
                                    "Router Mode: %s\n"
                                    "Route Stats: direct_avg=%.3f cot_avg=%.3f rescue_rate=%.3f damage_rate=%.3f reason=%s\n"
                                    "Initial Direct Answer: %s\n"
                                    "Protected Review Prompt:\n%s\n"
                                    "Protected Review Response:\n%s\n"
                                    "Final Answer: %s"
                                ) % (
                                    selected_strategy, self.args.strategy_router_mode,
                                    route.get("direct_avg", 0.0), route.get("cot_avg", 0.0),
                                    route.get("rescue_rate", 0.0), route.get("damage_rate", 0.0),
                                    route.get("reason", ""), initial_answer, review_prompt, review_response,
                                    extracted_answer,
                                )
                        else:
                            extracted_answer = initial_answer
                            response = (
                                "RAG Strategy Router: %s\n"
                                "Router Mode: %s\n"
                                "Route Stats: direct_avg=%.3f cot_avg=%.3f direct_hard_rate=%.3f "
                                "complex_win_rate=%.3f rescue_rate=%.3f damage_rate=%.3f reason=%s\n"
                                "Initial Direct Response:\n%s\n"
                                "Final Answer: %s"
                            ) % (
                                selected_strategy, self.args.strategy_router_mode,
                                route.get("direct_avg", 0.0), route.get("cot_avg", 0.0),
                                route.get("direct_hard_rate", 0.0), route.get("complex_win_rate", 0.0),
                                route.get("rescue_rate", 0.0), route.get("damage_rate", 0.0),
                                route.get("reason", ""), response, extracted_answer,
                            )
                    elif self.args.cot_style == "complex_decompose":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        should_decompose = self._is_complex_for_decomposition(question)
                        decompose_prompt = ""
                        decompose_response = ""
                        verify_prompt = ""
                        verify_response = ""
                        decomposed_answer = ""

                        if should_decompose:
                            decompose_prompt = self._format_decompose_prompt(
                                question, choice_text, cur_caption, initial_answer, question_type
                            )
                            decompose_response = self._call_llm(decompose_prompt, image_path=answer_image_path)
                            decomposed_answer = self._clean_short_answer(
                                self._extract_structured_cot_answer(decompose_response)
                            )
                            if self._looks_like_visual_cue_list(decomposed_answer) or not decomposed_answer:
                                decomposed_answer = initial_answer

                            if self.args.decompose_verify:
                                verify_prompt = self._format_decompose_verify_prompt(
                                    question, choice_text, cur_caption, initial_answer,
                                    decomposed_answer, decompose_response
                                )
                                verify_response = self._call_llm(verify_prompt, image_path=answer_image_path)
                                extracted_answer = self._extract_decompose_verify_answer(
                                    verify_response, initial_answer, decomposed_answer
                                )
                            else:
                                extracted_answer = decomposed_answer
                        else:
                            extracted_answer = initial_answer

                        response = (
                            "Complex Decompose Mode: %s\n"
                            "Should Decompose: %s\n"
                            "Initial Direct Answer: %s\n"
                            "Decompose Prompt:\n%s\n"
                            "Decompose Response:\n%s\n"
                            "Decomposed Answer: %s\n"
                            "Verify Enabled: %s\n"
                            "Verify Prompt:\n%s\n"
                            "Verify Response:\n%s\n"
                            "Final Answer: %s"
                        ) % (
                            self.args.decompose_complexity_mode,
                            should_decompose,
                            initial_answer,
                            decompose_prompt,
                            decompose_response,
                            decomposed_answer,
                            self.args.decompose_verify,
                            verify_prompt,
                            verify_response,
                            extracted_answer,
                        )
                    elif self.args.cot_style == "direct_rephrase_consistency":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        rephrase_result = self._run_rephrase_consistency(
                            question, choice_text, cur_caption, regional_context, ocr_context,
                            initial_answer, question_type, answer_image_path
                        )
                        extracted_answer = rephrase_result["final_answer"]
                        response = rephrase_result["trace"]
                    elif self.args.cot_style == "candidate_judge":
                        selected_objects = onion_instruction[1] if len(onion_instruction) > 1 else []
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        candidate_records = [{
                            "label": "direct_context",
                            "answer": initial_answer,
                            "prompt": prompt,
                            "response": response,
                        }]
                        self._current_initial_answer = initial_answer

                        image_only_prompt = self._format_candidate_prompt(
                            question, choice_text, "", "image_only", question_type
                        )
                        candidate_records.append(self._call_candidate_answer(
                            "direct_image_only", image_only_prompt, image_path
                        ))

                        answer_first_prompt = self._format_candidate_prompt(
                            question, choice_text, cur_caption, "answer_first_locked", question_type
                        )
                        candidate_records.append(self._call_candidate_answer(
                            "answer_first_locked", answer_first_prompt, image_path, extractor="first_answer"
                        ))

                        if self.args.candidate_judge_include_caption_candidate and caption:
                            caption_prompt = self._format_candidate_prompt(
                                question, choice_text, caption, "caption_only", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "caption_only", caption_prompt, image_path
                            ))

                        if question_type in ("text_ocr", "visual_detail", "category"):
                            visual_context = cur_caption
                            if regional_context:
                                visual_context += "\n" + regional_context
                            if ocr_context:
                                visual_context += "\n" + ocr_context
                            visual_prompt = self._format_candidate_prompt(
                                question, choice_text, visual_context, "visual_detail", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "visual_detail_guarded", visual_prompt, image_path
                            ))

                        if question_type in ("knowledge", "category"):
                            knowledge_context = cur_caption
                            if enhance_knowledge:
                                knowledge_context += "\n" + enhance_knowledge
                            knowledge_prompt = self._format_candidate_prompt(
                                question, choice_text, knowledge_context, "knowledge_guarded", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "knowledge_guarded", knowledge_prompt, image_path
                            ))

                        if self.args.candidate_judge_include_count_candidate and self._question_is_count(question):
                            count_context = cur_caption
                            if regional_context:
                                count_context += "\n" + regional_context
                            if enhance_caption:
                                count_context += "\n" + enhance_caption
                            count_prompt = self._format_candidate_prompt(
                                question, choice_text, count_context, "count_specialist", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "count_specialist", count_prompt, image_path
                            ))

                        if self.args.candidate_judge_include_ocr_candidate and self._question_is_ocr(question):
                            ocr_candidate_context = cur_caption
                            if ocr_context:
                                ocr_candidate_context += "\nOCR: " + ocr_context
                            if regional_context:
                                ocr_candidate_context += "\n" + regional_context
                            ocr_prompt = self._format_candidate_prompt(
                                question, choice_text, ocr_candidate_context, "ocr_specialist", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "ocr_specialist", ocr_prompt, image_path
                            ))

                        if self.args.candidate_judge_include_coverage_candidate:
                            coverage_context = cur_caption
                            coverage_parts = [regional_context, ocr_context, enhance_caption, enhance_knowledge]
                            if dyfo_evidence_enabled:
                                coverage_parts.append(dyfo_visual_evidence)
                            for part in coverage_parts:
                                if part:
                                    coverage_context += "\n" + part
                            coverage_prompt = self._format_candidate_prompt(
                                question, choice_text, coverage_context, "coverage_scan", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "coverage_scan", coverage_prompt, image_path
                            ))

                        if self.args.candidate_judge_include_contrast_candidate:
                            contrast_context = cur_caption
                            if regional_context:
                                contrast_context += "\n" + regional_context
                            if ocr_context:
                                contrast_context += "\n" + ocr_context
                            contrast_prompt = self._format_candidate_prompt(
                                question, choice_text, contrast_context, "contrastive", question_type
                            )
                            candidate_records.append(self._call_candidate_answer(
                                "contrastive_alternative", contrast_prompt, image_path
                            ))

                        for rec in candidate_records:
                            rec["answer"] = self._clean_short_answer(rec.get("answer", ""))
                            rec["normalized"] = self._normalize_candidate_answer(rec["answer"])

                        consensus_answer = self._candidate_consensus_answer(candidate_records)
                        unique_candidate_records = self._dedupe_candidate_records(candidate_records)
                        if consensus_answer and not self.args.candidate_judge_always_judge:
                            extracted_answer = consensus_answer
                            judge_prompt = ""
                            judge_response = "Skipped: candidate consensus."
                        elif len(unique_candidate_records) <= 1:
                            extracted_answer = unique_candidate_records[0]["answer"] if unique_candidate_records else initial_answer
                            judge_prompt = ""
                            judge_response = "Skipped: only one valid unique candidate."
                        else:
                            evidence_text = self._build_reviewer_evidence(
                                base_context=caption,
                                selected_objects=selected_objects,
                                regional_context=regional_context if self.args.use_all_regional_captions else regional_context,
                                ocr_context=ocr_context if self.args.use_ocr_context else "",
                                enhance_caption=enhance_caption,
                                enhance_knowledge=enhance_knowledge,
                                enhance_image_path=enhance_image_path,
                                qwen_global_caption=qwen_global_caption if self.args.use_qwen_blip2_caption else "",
                                qwen_local_caption=qwen_local_caption if self.args.use_qwen_blip2_caption else "",
                                dyfo_visual_evidence=dyfo_visual_evidence if dyfo_evidence_enabled else "",
                            )
                            judge_prompt = self._format_candidate_judge_prompt(
                                question, choice_text, question_type, evidence_text, unique_candidate_records
                            )
                            judge_image_path = image_path
                            if self.args.candidate_judge_use_enhanced_image and enhance_image_path:
                                judge_image_path = enhance_image_path
                            judge_response = self._call_llm(judge_prompt, image_path=judge_image_path)
                            extracted_answer = self._extract_candidate_judge_answer(
                                judge_response, unique_candidate_records, initial_answer
                            )

                        candidate_summary = "\n".join(
                            "[%s] answer=%s\nprompt:\n%s\nresponse:\n%s"
                            % (
                                rec.get("label", "candidate"),
                                rec.get("answer", ""),
                                rec.get("prompt", ""),
                                rec.get("response", ""),
                            )
                            for rec in candidate_records
                        )
                        response = (
                            "Question Type: %s\n"
                            "Candidate Answers:\n%s\n"
                            "Judge Prompt:\n%s\n"
                            "Judge Response:\n%s\n"
                            "Final Answer: %s"
                        ) % (question_type, candidate_summary, judge_prompt, judge_response, extracted_answer)
                    elif self.args.cot_style == "direct_verify":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        verify_prompt = self._format_direct_verify_prompt(cur_caption, question, choice_text, initial_answer)
                        verify_response = self._call_llm(verify_prompt, image_path=answer_image_path)
                        extracted_answer = self._extract_direct_verify_answer(verify_response, initial_answer)
                        response = (
                            "Initial Answer: %s\n"
                            "Verification Prompt:\n%s\n"
                            "Verification Response:\n%s"
                        ) % (initial_answer, verify_prompt, verify_response)
                    elif self.args.cot_style == "reviewer_evidence":
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        selected_objects = onion_instruction[1] if len(onion_instruction) > 1 else []
                        evidence_text = self._build_reviewer_evidence(
                            base_context=caption,
                            selected_objects=selected_objects,
                            regional_context=regional_context if self.args.use_all_regional_captions else "",
                            ocr_context=ocr_context if self.args.use_ocr_context else "",
                            enhance_caption=enhance_caption,
                            enhance_knowledge=enhance_knowledge,
                            enhance_image_path=enhance_image_path,
                            qwen_global_caption=qwen_global_caption if self.args.use_qwen_blip2_caption else "",
                            qwen_local_caption=qwen_local_caption if self.args.use_qwen_blip2_caption else "",
                            dyfo_visual_evidence=dyfo_visual_evidence if dyfo_evidence_enabled else "",
                        )
                        verify_prompt = self._format_reviewer_evidence_prompt(
                            question, choice_text, initial_answer, evidence_text
                        )
                        verify_response = self._call_llm(verify_prompt, image_path=reviewer_image_path)
                        extracted_answer = self._extract_direct_verify_answer(verify_response, initial_answer)
                        response = (
                            "Initial Answer: %s\n"
                            "Reviewer Evidence:\n%s\n"
                            "Reviewer Prompt:\n%s\n"
                            "Reviewer Response:\n%s"
                        ) % (initial_answer, evidence_text, verify_prompt, verify_response)
                    elif self.args.cot_style in ("reflective_answer_first", "adaptive_reflective_answer_first"):
                        initial_responses = [response]
                        initial_answers = [self._extract_first_answer_line(response)]
                        for ensemble_idx in range(1, max(1, self.args.reflect_initial_ensemble)):
                            extra_response = self._call_llm(prompt, image_path=answer_image_path)
                            initial_responses.append(extra_response)
                            initial_answers.append(self._extract_first_answer_line(extra_response))

                        if len(initial_answers) > 1:
                            normalized_counts = {}
                            normalized_to_answer = {}
                            for ans in initial_answers:
                                normalized = process_answer(ans)
                                normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
                                normalized_to_answer.setdefault(normalized, ans)
                            best_norm = max(normalized_counts, key=normalized_counts.get)
                            current_answer = normalized_to_answer[best_norm]
                        else:
                            current_answer = initial_answers[0]

                        transcript = ["Round 1 Answer: %s" % current_answer]
                        if len(initial_responses) > 1:
                            transcript.append("Initial Ensemble Responses:\n%s" % "\n---\n".join(initial_responses))
                            transcript.append("Initial Ensemble Answers: %s" % initial_answers)
                        reflect_rounds = max(1, self.args.reflect_rounds)
                        reflect_cycles = max(0, (reflect_rounds - 1) // 2)
                        should_review = self._should_run_reflective_review(question, response, current_answer)
                        if not should_review:
                            transcript.append("Reflective Review: skipped by trigger mode %s" % self.args.reflect_trigger_mode)
                            reflect_cycles = 0
                        review_context = "" if self.args.reflect_review_context == "empty" else cur_caption
                        for cycle in range(reflect_cycles):
                            rationale_prompt = self._format_reflective_rationale_prompt(
                                review_context, question, choice_text, current_answer
                            )
                            rationale_response = self._call_llm(rationale_prompt, image_path=answer_image_path)
                            review_prompt = self._format_reflective_review_prompt(
                                review_context, question, choice_text, current_answer, rationale_response
                            )
                            review_response = self._call_llm(review_prompt, image_path=answer_image_path)
                            revised_answer = self._extract_reflective_review_answer(review_response, current_answer)
                            transcript.extend([
                                "Round %d Evidence Prompt:\n%s" % (2 + cycle * 2, rationale_prompt),
                                "Round %d Evidence Response:\n%s" % (2 + cycle * 2, rationale_response),
                                "Round %d Review Prompt:\n%s" % (3 + cycle * 2, review_prompt),
                                "Round %d Review Response:\n%s" % (3 + cycle * 2, review_response),
                                "Round %d Answer: %s" % (3 + cycle * 2, revised_answer),
                            ])
                            current_answer = revised_answer
                        extracted_answer = current_answer
                        response = "\n".join(transcript)
                    elif self.args.cot_style == "answer_first_locked":
                        extracted_answer = self._extract_first_answer_line(response)
                    elif self.args.cot_style in ("compact", "answer_first", "visual_facts"):
                        extracted_answer = self._extract_structured_cot_answer(response)
                    else:
                        extracted_answer = self._extract_answer_from_response(response)
                    pred_candidates.append(extracted_answer)
                    filtered_thought, all_thought = response, response
                    if self.args.use_clip_thought_verify:
                        filtered_thought, all_thought = self._filter_thoughts_with_clip(key, response)
                    if self.args.use_qwen_blip2_thought_verify:
                        filtered_thought, all_thought = self._filter_thoughts_with_qwen_image_check(image_path, filtered_thought)
                    thought_list.append(filtered_thought)
                    all_thought_list.append(all_thought)
                else:
                    pred_candidates.append(response)

                print('-----sample_inference-----model generate response-----+++++-----beg')
                print(response)
                print('-----sample_inference-----model generate response-----+++++-----end')
                print()

            if self.args.debug:
                t3=time.time()

            # # 参数输出指示点 False True False
            # print('----------print(self.args.chain_of_thoughts)')
            # print(self.args.chain_of_thoughts)
            # print('----------print(self.args.with_clip_verify)')
            # print(self.args.with_clip_verify)
            # print('----------print(self.args.choice_only)')
            # print(self.args.choice_only)

            if self.args.debug:
                t4=time.time()
                print("    REASON PREPARE TIME", t2-t1)
                print("    REASON INF TIME", t3-t2)
                print("    REASON POST TIME", t4-t3)
        maxval = -999.
        pred_candidates = [self._postprocess_answer(candidate) for candidate in pred_candidates]
        if (
            self.args.mcts_action_mode == "dyfo_evidence"
            and self.args.dyfo_decision_mode in ("best_focus_answer", "weighted_vote")
            and dyfo_final_answer
        ):
            pred_candidates = [self._postprocess_answer(dyfo_final_answer)]
            print('-----dyfo_decision-----使用DyFo native final answer-----+++++-----beg')
            print('dyfo_decision_mode:', self.args.dyfo_decision_mode)
            print('dyfo_final_answer:', dyfo_final_answer)
            print('dyfo_decision_trace:', dyfo_decision_trace)
            print('-----dyfo_decision-----使用DyFo native final answer-----+++++-----end')
            print()

        if self.args.ensemble_strategy == "first":
            pred_answer = pred_candidates[0]
        elif self.args.ensemble_strategy == "normalized_majority":
            normalized_counts = {}
            original_by_norm = {}
            for candidate in pred_candidates:
                norm = process_answer(candidate)
                normalized_counts[norm] = normalized_counts.get(norm, 0) + 1
                original_by_norm.setdefault(norm, candidate)
            best_norm = max(normalized_counts, key=normalized_counts.get)
            pred_answer = original_by_norm[best_norm]
        else:
            # 集成投票：对所有n_ensemble次采样取多数答案
            pred_answer = max(set(pred_candidates), key=pred_candidates.count)

        ## a rough accuracy estimator for fast results check
        if self.args.choice_only:
            if pred_answer not in self.choices_dict[key]:
                choices_list = self.choices_dict[key] + [pred_answer]
                inputs = self.clip_processor(text=choices_list, return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                thought_emb = clip_outputs['pooler_output']
                thought_emb /= thought_emb.norm(dim=-1, keepdim=True)
                sim = thought_emb[-1].unsqueeze(0) @ thought_emb[:-1].T
                pred_answer = self.choices_dict[key][sim.argmax().item()]
            final_score = 1 if pred_answer == self.choices_dict[key][answer] else 0
        else:
            if self.args.dataset_name in ("pope", "mme"):
                final_score = yes_no_answer_score(pred_answer, answer)
            elif self.args.legacy_answer_normalization:
                final_score = legacy_normalized_direct_answer_score(pred_answer, answer)
            else:
                final_score = official_direct_answer_score(pred_answer, answer)
        if self.args.debug:
            print(prompt)
            print(pred_answer)
            print(answer)
            pdb.set_trace()
        round_state = self._make_round_state(
            round_idx=round_idx,
            onion_instruction=onion_instruction,
            enhance_image_path=enhance_image_path,
            enhance_caption=enhance_caption,
            enhance_knowledge=enhance_knowledge,
            dyfo_visual_evidence=dyfo_visual_evidence,
            pred_answer=pred_answer,
            final_score=final_score,
            pred_candidates=pred_candidates,
            dyfo_decision_trace=dyfo_decision_trace
        )
        if self.args.chain_of_thoughts:
            return [key, pred_answer, prompt, final_score, thought_list, all_thought_list, float(maxval),
                    self._region_object_names(scene_graph_attr), round_state]
        return [key, pred_answer, prompt, final_score, float(maxval), self._region_object_names(scene_graph_attr),
                round_state]
    
    def onion_make_instruction(self, key, object_list):
        onion_instruction = []
        image_key = int(key.split('<->')[0]) if self.args.dataset_name!="fvqa" else self.image_dict[key]
        image_path = self.dataset.find_image_path(image_key)
        question = self.dataset.question_dict[key]

        # ====================== 多点投票 + 三方向打分（A+B组合方案） ======================
        # 3次采样用不同措辞，每次对 A/B/C 各打1-5分，取平均值最高方向
        prompt_variants = [
            'Please rate each of the following on a scale of 1-5 based on how much additional '
            'information is needed to answer the question (1 = not needed, 5 = strongly needed).\n'
            'A. image, B. caption, C. knowledge\n'
            'Output format: "A:4, B:3, C:2"',
            'For each option, assess how much it would help improve the answer to the question '
            'on a 1-5 scale (1 = no help, 5 = significant help).\n'
            'A. image, B. caption, C. knowledge\n'
            'Output format: "A:4, B:3, C:2"',
            'Evaluate the potential contribution of each part to answering the question correctly '
            'on a 1-5 scale (1 = low contribution, 5 = high contribution).\n'
            'A. image, B. caption, C. knowledge\n'
            'Output format: "A:4, B:3, C:2"',
        ]

        base_prompt = ('I am giving you a question and an image, but you do not need to answer it.\n'
                       'Question: %s\n' % question)

        # 多轮采样
        all_scores = {"image": [], "caption": [], "knowledge": []}
        all_messages = None
        for variant_idx, variant_prompt in enumerate(prompt_variants):
            full_prompt = base_prompt + variant_prompt
            try:
                response, msgs = self._call_llm(
                    full_prompt,
                    image_path=image_path, return_history=True
                )
                if variant_idx == 0:
                    all_messages = msgs
                # 解析 "A:4, B:3, C:2" 格式
                scores = self._parse_onion_scores(response)
                for key_name in all_scores:
                    if key_name in scores:
                        all_scores[key_name].append(scores[key_name])
            except Exception as e:
                print(f"onion打分采样{variant_idx}失败: {e}")

        # 计算各方向平均分
        avg_scores = {}
        for key_name, score_list in all_scores.items():
            if score_list:
                avg_scores[key_name] = sum(score_list) / len(score_list)
            else:
                avg_scores[key_name] = 0.0

        # 选最高分方向；若最高分 < 3，本轮跳过增强
        SCORE_THRESHOLD = 3.0
        scored_dirs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_dir, top_score = scored_dirs[0]

        if top_score < SCORE_THRESHOLD:
            selected = "skip"
            print(f'onion决策：所有方向需求度<{SCORE_THRESHOLD}，跳过增强。scores={avg_scores}')
        else:
            selected = top_dir
            print(f'onion决策：选择"{selected}"方向 (avg_score={top_score:.1f})。scores={avg_scores}')

        # 如果selected=skip，后续sample_inference中不会触发任何增强模块
        # ======================================================================

        onion_instruction.append(selected)
        onion_instruction.append(object_list)
        onion_instruction.append({
            "scores": avg_scores,
            "threshold": SCORE_THRESHOLD,
        })
        return onion_instruction, all_messages

    def _parse_onion_scores(self, response):
        """从VLM回复中解析 A:4, B:3, C:2 格式的分数"""
        scores = {}
        if not isinstance(response, str) or len(response) == 0:
            return scores

        import re
        # 匹配 "A:4", "B:3" 等模式
        label_map = {"A": "image", "B": "caption", "C": "knowledge"}
        matches = re.findall(r'([A-C])\s*:\s*(\d)', response, re.IGNORECASE)
        for label, score_str in matches:
            label = label.upper()
            if label in label_map:
                score = int(score_str)
                score = max(1, min(5, score))  # 裁剪到 [1,5]
                scores[label_map[label]] = score
        return scores
    
    def _mcts_should_trigger(self, question):
        mode = getattr(self.args, "mcts_trigger_mode", "all")
        if mode == "all":
            return True

        q = question.lower()
        global_patterns = [
            "what city", "what country", "what place", "where", "what event",
            "what activity", "what period", "what time", "why", "used for",
            "most likely", "what institution", "what kind of resort"
        ]
        visual_patterns = [
            "what color", "how many", "what word", "what is written", "what sign",
            "what logo", "what brand", "what number", "what letter", "what item",
            "what object", "what animal", "what kind of animal", "what is on",
            "what is in", "what is behind", "what is holding", "what is wearing",
            "what is hanging", "what is made of", "what type of", "which"
        ]

        if any(pattern in q for pattern in global_patterns):
            return False

        if mode == "count_color_object_only":
            narrow_patterns = [
                "how many", "what color", "which color", "what object",
                "what item", "what animal", "what food", "what device",
                "what appliance", "what is on", "what is in", "what is behind",
                "what is holding", "what is wearing", "what is hanging"
            ]
            narrow_exclusions = [
                "what type", "what kind", "which", "what city", "what country",
                "what place", "where", "why", "used for", "most likely",
                "what event", "what activity", "what period", "what time"
            ]
            if any(pattern in q for pattern in narrow_exclusions):
                return False
            return any(pattern in q for pattern in narrow_patterns)

        return any(pattern in q for pattern in visual_patterns)

    def _dyfo_should_trigger(self, question, question_type):
        if getattr(self.args, "dataset_name", "") == "mme":
            return True
        mode = getattr(self.args, "dyfo_trigger_mode", "visual_detail")
        if mode == "always":
            return True
        if mode == "never":
            return False
        if mode == "mcts":
            return self._mcts_should_trigger(question)
        if mode == "visual_detail":
            return (
                question_type in ("visual_detail", "text_ocr", "category")
                or self._question_is_count(question)
                or "color" in str(question).lower()
            )
        return self._mcts_should_trigger(question)

    def _parse_dyfo_focus_text(self, response, fallback):
        text = str(response).strip()
        matches = re.findall(r"(?:focus|target|cue)\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if matches:
            text = matches[-1].strip()
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = lines[-1] if lines else str(fallback)
        text = re.sub(r"^\s*(?:[-*]|\d+[\).:])\s*", "", text).strip()
        text = text.strip(" \t\"'`.,;:!?")
        if not text or text.lower() in ("none", "n/a", "na", "unknown") or len(text.split()) > 12:
            return str(fallback).strip()
        return text

    def _dyfo_question_focus_fallback(self, question):
        focus = re.sub(r"\bplease answer yes or no\b", "", str(question), flags=re.IGNORECASE)
        focus = re.sub(r"\b(answer|reply) (with )?(only )?(yes|no|yes or no)\b", "", focus, flags=re.IGNORECASE)
        focus = focus.strip(" ?.")
        focus = re.sub(r"^(is|are|was|were|do|does|did|can|could|would|will|has|have|had)\s+", "", focus, flags=re.IGNORECASE)
        focus = re.sub(r"^(there|this|that|the image|the picture|a photo|this photo)\s+", "", focus, flags=re.IGNORECASE)
        focus = focus.strip(" ?.")
        words = focus.split()
        if len(words) > 8:
            focus = " ".join(words[:8])
        return focus or "the visual evidence needed by the question"

    def _dyfo_initial_focus(self, question, obj_list):
        fallback = obj_list[0] if obj_list else self._dyfo_question_focus_fallback(question)
        object_hint = ", ".join(obj_list[:8]) if obj_list else "No detector candidates are available; infer the cue from the question."
        prompt = (
            "Choose the most useful visual focus cue for answering the question.\n"
            "The focus should be a visible object, attribute, text area, relation, or small region that a visual expert can localize.\n"
            "Do not answer the question. Output exactly: Focus: <short visual cue>\n"
            "Question: %s\n"
            "Candidate objects: %s"
        ) % (question, object_hint)
        response = self._call_llm(prompt, image_path=None, max_new_tokens=self.args.dyfo_focus_max_tokens, use_images=False)
        return self._parse_dyfo_focus_text(response, fallback), response

    def _dyfo_refine_focus(self, question, current_focus, action, image_path):
        if action == "semantic_focus":
            instruction = (
                "Make the visual focus more specific and localizable. Prefer the object, attribute, text, or relation "
                "most directly needed by the question."
            )
        else:
            instruction = (
                "Broaden the visual focus just enough to include missing context around the current target. "
                "Keep it short and localizable."
            )
        prompt = (
            "You are updating a visual search focus for a VQA system.\n"
            "%s\n"
            "Do not answer the question. Output exactly: Focus: <short visual cue>\n"
            "Question: %s\n"
            "Current focus: %s"
        ) % (instruction, question, current_focus)
        response = self._call_llm(
            prompt, image_path=image_path if self.args.dyfo_text_focus_use_image else None,
            max_new_tokens=self.args.dyfo_focus_max_tokens,
            use_images=self.args.dyfo_text_focus_use_image,
        )
        return self._parse_dyfo_focus_text(response, current_focus), response

    def _dyfo_locate_focus(self, image_pil, focus_text):
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        try:
            self.ensure_lang_sam()
            with torch.no_grad():
                result = self.sam.predict([image_pil], [focus_text])
        except Exception as exc:
            print(f"[dyfo] visual expert failed for focus={focus_text}: {exc}")
            return None
        if not result:
            return None
        masks = result[0].get("masks", None)
        if masks is None or len(masks) == 0:
            return None
        boxes = []
        for mask in masks:
            mask_array = mask if isinstance(mask, np.ndarray) else np.array(mask)
            if mask_array.dtype != bool:
                mask_array = mask_array > 0
            y_idx, x_idx = np.where(mask_array)
            if len(y_idx) == 0 or len(x_idx) == 0:
                continue
            boxes.append([
                int(np.min(x_idx)), int(np.min(y_idx)),
                int(np.max(x_idx)), int(np.max(y_idx))
            ])
        if not boxes:
            return None
        boxes = np.array(boxes)
        return (
            int(np.min(boxes[:, 0])),
            int(np.min(boxes[:, 1])),
            int(np.max(boxes[:, 2])),
            int(np.max(boxes[:, 3])),
        )

    def _dyfo_expand_box(self, box, image_size, scale):
        w, h = image_size
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(1.0, (x2 - x1) * scale)
        bh = max(1.0, (y2 - y1) * scale)
        return (
            int(max(0, cx - bw / 2.0)),
            int(max(0, cy - bh / 2.0)),
            int(min(w, cx + bw / 2.0)),
            int(min(h, cy + bh / 2.0)),
        )

    def _dyfo_crop_for_node(self, image_pil, box):
        if not box:
            return image_pil
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return image_pil
        return image_pil.crop((x1, y1, x2, y2))

    def _dyfo_consistency_check(self, crop, focus_text, question):
        temp_path = None
        try:
            temp_path = os.path.join(
                self.args.cache_path,
                "dyfo_check_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
            )
            os.makedirs(self.args.cache_path, exist_ok=True)
            crop.save(temp_path)
            prompt = (
                "Check whether the image crop clearly contains the visual focus needed for the question.\n"
                "Reply with only yes or no.\n"
                "Question: %s\n"
                "Visual focus: %s"
            ) % (question, focus_text)
            reply = self._call_llm(prompt, image_path=temp_path, max_new_tokens=8)
            return 1.0 if str(reply).strip().lower().startswith("yes") else 0.0, reply
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _dyfo_answer_from_crop(self, crop, focus_text, question):
        temp_path = None
        try:
            temp_path = os.path.join(
                self.args.cache_path,
                "dyfo_answer_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
            )
            os.makedirs(self.args.cache_path, exist_ok=True)
            crop.save(temp_path)
            if getattr(self.args, "dataset_name", "") in ("pope", "mme"):
                prompt = (
                    "Answer the visual question using this focused image region.\n"
                    "Use the visual focus as a hint, but answer the original question.\n"
                    "Return only yes or no.\n"
                    "Question: %s\n"
                    "Visual focus: %s\n"
                    "Answer:"
                ) % (question, focus_text)
            else:
                prompt = (
                    "Answer the visual question using this focused image region.\n"
                    "Use the visual focus as a hint, but answer the original question.\n"
                    "If the crop is insufficient, make the best concise answer from visible evidence.\n"
                    "Return only one word or a short phrase.\n"
                    "Question: %s\n"
                    "Visual focus: %s\n"
                    "Answer:"
                ) % (question, focus_text)
            response = self._call_llm(
                prompt,
                image_path=temp_path,
                max_new_tokens=getattr(self.args, "dyfo_answer_max_tokens", 32),
            )
            answer = self._clean_short_answer(self._extract_answer_from_response(response))
            return answer, response, prompt
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _dyfo_node_reward(self, visual_hit, lmm_consistent, area_ratio):
        if not visual_hit:
            return 0.0
        consistency = 1.0 if lmm_consistent > 0 else 0.0
        if getattr(self.args, "dyfo_area_reward", "compact") == "paper":
            area_score = area_ratio
        else:
            area_score = 1.0 - area_ratio
        return max(0.0, min(1.0, consistency * area_score))

    def _dyfo_weighted_vote(self, nodes):
        vote_scores = defaultdict(float)
        norm_to_answer = {}
        vote_items = []
        for node in nodes:
            answer = self._clean_short_answer(getattr(node, "local_answer", ""))
            if not answer:
                continue
            norm = normalize_vqa_answer(answer)
            if not norm:
                continue
            mean_value = node.value / max(1, node.visits)
            weight = max(float(getattr(node, "reward", 0.0)), float(mean_value), 1e-6)
            vote_scores[norm] += weight
            norm_to_answer.setdefault(norm, answer)
            vote_items.append({
                "focus": node.focus,
                "action": node.action,
                "depth": node.depth,
                "answer": answer,
                "normalized_answer": norm,
                "weight": weight,
                "reward": node.reward,
                "visits": node.visits,
            })
        if not vote_scores:
            return "", {"vote_scores": {}, "vote_items": vote_items}
        best_norm = max(vote_scores, key=vote_scores.get)
        return norm_to_answer[best_norm], {
            "best_normalized_answer": best_norm,
            "vote_scores": dict(vote_scores),
            "vote_items": vote_items,
        }

    def _run_dyfo_visual_evidence_search(self, data_row, obj_list, attr_list):
        key = data_row["key"]
        image_path = data_row["image_path"]
        question = data_row["question"]
        question_type = self._classify_vqa_question_type(question)
        if not self._dyfo_should_trigger(question, question_type):
            return {"evidence": "", "focus_image_path": None, "trace": "skipped_by_trigger"}

        _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
        obj_list = list(dict.fromkeys((obj_list or []) + selected_objects))
        if not obj_list:
            obj_list = selected_objects

        original = Image.open(image_path).convert("RGB")
        image_area = max(1, original.width * original.height)
        initial_focus, initial_focus_response = self._dyfo_initial_focus(question, obj_list)

        class _Node:
            def __init__(self, focus, box, depth, parent=None, action="root"):
                self.focus = focus
                self.box = box
                self.image_region = box
                self.textual_cue = focus
                self.depth = depth
                self.parent = parent
                self.action = action
                self.children = []
                self.untried = ["semantic_focus", "semantic_scatter"]
                self.visits = 0
                self.value = 0.0
                self.reward = 0.0
                self.visual_hit = False
                self.lmm_reply = ""
                self.focus_response = ""
                self.local_answer = ""
                self.local_answer_response = ""
                self.local_answer_prompt = ""

        root = _Node(initial_focus, (0, 0, original.width, original.height), 0)
        nodes = [root]

        def select_node(node):
            while node.children and not node.untried and node.depth < self.args.dyfo_max_depth:
                total = max(1, sum(child.visits for child in node.children))
                def ucb(child):
                    exploit = child.value / max(1, child.visits)
                    explore = self.args.dyfo_exploration_weight * math.sqrt(math.log(total + 1) / max(1, child.visits))
                    return exploit + explore
                node = max(node.children, key=ucb)
            return node

        for _ in range(max(1, self.args.dyfo_n_simulations)):
            leaf = select_node(root)
            if leaf.depth >= self.args.dyfo_max_depth:
                target = leaf
            else:
                action = leaf.untried.pop(0) if leaf.untried else random.choice(["semantic_focus", "semantic_scatter"])
                parent_crop = self._dyfo_crop_for_node(original, leaf.box)
                parent_crop_path = None
                try:
                    parent_crop_path = os.path.join(
                        self.args.cache_path,
                        "dyfo_parent_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
                    )
                    os.makedirs(self.args.cache_path, exist_ok=True)
                    parent_crop.save(parent_crop_path)
                    focus, focus_response = self._dyfo_refine_focus(question, leaf.focus, action, parent_crop_path)
                finally:
                    if parent_crop_path and os.path.exists(parent_crop_path):
                        os.remove(parent_crop_path)

                if action == "semantic_focus":
                    base_crop = parent_crop
                    local_box = self._dyfo_locate_focus(base_crop, focus)
                    if local_box:
                        lx1, ly1, lx2, ly2 = local_box
                        px1, py1, _, _ = leaf.box
                        box = (px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2)
                    else:
                        box = leaf.box
                else:
                    located = self._dyfo_locate_focus(parent_crop, focus)
                    if located:
                        lx1, ly1, lx2, ly2 = located
                        px1, py1, _, _ = leaf.box
                        local_abs = (px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2)
                        box = self._dyfo_expand_box(local_abs, original.size, self.args.dyfo_scatter_scale)
                    else:
                        box = self._dyfo_expand_box(leaf.box, original.size, self.args.dyfo_scatter_scale)
                box = self._dyfo_expand_box(box, original.size, self.args.dyfo_focus_padding)
                target = _Node(focus, box, leaf.depth + 1, parent=leaf, action=action)
                target.focus_response = focus_response
                leaf.children.append(target)
                nodes.append(target)

            crop = self._dyfo_crop_for_node(original, target.box)
            visual_hit = self._dyfo_locate_focus(crop, target.focus) is not None
            lmm_consistent, lmm_reply = self._dyfo_consistency_check(crop, target.focus, question)
            x1, y1, x2, y2 = target.box
            area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / image_area))
            reward = self._dyfo_node_reward(visual_hit, lmm_consistent, area_ratio)
            target.reward = reward
            target.visual_hit = visual_hit
            target.lmm_reply = lmm_reply
            if self.args.dyfo_decision_mode in ("best_focus_answer", "weighted_vote") and not target.local_answer:
                local_answer, local_response, local_prompt = self._dyfo_answer_from_crop(
                    crop, target.focus, question
                )
                target.local_answer = local_answer
                target.local_answer_response = local_response
                target.local_answer_prompt = local_prompt
            node = target
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

        best_node = max(nodes, key=lambda n: (n.reward, n.value / max(1, n.visits), -n.depth))
        best_crop = self._dyfo_crop_for_node(original, best_node.box)
        if self.args.dyfo_decision_mode == "best_focus_answer" and not best_node.local_answer:
            local_answer, local_response, local_prompt = self._dyfo_answer_from_crop(
                best_crop, best_node.focus, question
            )
            best_node.local_answer = local_answer
            best_node.local_answer_response = local_response
            best_node.local_answer_prompt = local_prompt
        image_filename = os.path.basename(image_path)
        focus_image_path = os.path.join(self.args.cache_path, "dyfo_focus_%s" % image_filename)
        os.makedirs(self.args.cache_path, exist_ok=True)
        best_crop.save(focus_image_path)

        evidence_prompt = (
            "Write concise visual evidence from this focused image crop for answering the question.\n"
            "Do not answer the question unless the evidence directly determines it.\n"
            "Use at most two short bullet points. Mention uncertainty if the crop is unclear.\n"
            "Question: %s\n"
            "Visual focus: %s"
        ) % (question, best_node.focus)
        evidence_response = self._call_llm(
            evidence_prompt, image_path=focus_image_path, max_new_tokens=self.args.dyfo_evidence_max_tokens
        )
        evidence = (
            "Focus: %s. Search action: %s. Reward: %.3f. Evidence: %s"
            % (best_node.focus, best_node.action, best_node.reward, self._truncate_text(evidence_response, 700))
        )
        final_answer = ""
        decision_trace = {
            "mode": self.args.dyfo_decision_mode,
            "best_focus_answer": best_node.local_answer,
        }
        if self.args.dyfo_decision_mode == "best_focus_answer":
            final_answer = best_node.local_answer
        elif self.args.dyfo_decision_mode == "weighted_vote":
            final_answer, vote_trace = self._dyfo_weighted_vote(nodes)
            decision_trace.update(vote_trace)
        trace = {
            "initial_focus": initial_focus,
            "initial_focus_response": initial_focus_response,
            "best_focus": best_node.focus,
            "best_action": best_node.action,
            "best_box": best_node.box,
            "best_reward": best_node.reward,
            "best_focus_answer": best_node.local_answer,
            "dyfo_decision_mode": self.args.dyfo_decision_mode,
            "dyfo_final_answer": final_answer,
            "dyfo_decision_trace": decision_trace,
            "nodes": [
                {
                    "focus": node.focus,
                    "textual_cue": node.textual_cue,
                    "action": node.action,
                    "depth": node.depth,
                    "box": node.box,
                    "image_region": node.image_region,
                    "reward": node.reward,
                    "visits": node.visits,
                    "visual_hit": node.visual_hit,
                    "lmm_reply": node.lmm_reply,
                    "local_answer": node.local_answer,
                }
                for node in nodes
            ],
        }
        print("[dyfo] visual evidence:", evidence)
        if final_answer:
            print("[dyfo] final answer:", final_answer)
        return {
            "evidence": evidence,
            "focus_image_path": focus_image_path,
            "final_answer": final_answer,
            "decision_trace": decision_trace,
            "trace": trace,
        }

    def enhance_image_object(self, data_row, obj_list, attr_list):

        # 获取样本信息
        key = data_row['key']
        image_path = data_row['image_path']
        question = data_row['question']

        # 如果是补充list，则补充并且去重（保留原逻辑用于物体候选）
        _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
        obj_list = list(dict.fromkeys(obj_list + selected_objects))
        self.attention_object = list(dict.fromkeys(self.attention_object + obj_list))
        obj_list = list(dict.fromkeys(self.attention_object + obj_list))

        if not self._mcts_should_trigger(question):
            print(f"MCTS跳过：问题不属于触发模式 {self.args.mcts_trigger_mode}: {question}")
            return None

        self.ensure_lang_sam()

        # ========== MCTS搜索最优物体增强 ==========
        # 将图像转为base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode()

        mcts_row = {
            'image': image_base64,
            'image_path': image_path,
            'question': question,
            'answer': data_row['answer'],
            'index': str(data_row['image_key']),
            'candidate_objects': obj_list
        }

        class _TempArgs:
            model_path = self.args.engine
            image_size = 1024
            temperature = 0.0

        temp_args = _TempArgs()
        temp_args.mcts_action_mode = self.args.mcts_action_mode
        temp_args.mcts_filter_objects = self.args.mcts_filter_objects

        mcts_sample = MCTSQuestionSample(
            row=mcts_row,
            args=temp_args,
            llm_model=self.model,
            llm_processor=self.processor,
            sam_model=self.sam,
            clip_model=self.clip_full_model,
            clip_processor=self.clip_full_processor,
            use_vllm=self.use_vllm,
            vllm_client=self.vllm_client,
            vllm_model_name=self.vllm_model_name
        )
        mcts_sample.n_simulations = self.args.mcts_n_simulations  # MCTS模拟次数

        # 提取关键物体并构建动作空间
        mcts_sample.key_objects = mcts_sample.extract_key_objects_sync()
        mcts_sample._setup_actions()

        if len(mcts_sample.actions) == 0:
            # 未检测到任何物体，回退到原图
            print("MCTS未检测到可增强的物体，使用原图")
            return None

        # 运行MCTS搜索最优增强图像
        final_answer, prompt, full_answer, best_image_b64, best_node, root_node = mcts_sample.get_final_answer()

        print(f"MCTS增强完成，最终答案: {final_answer}")

        # 保存最优增强图像到缓存
        best_image = mcts_sample._base64_to_image(best_image_b64)
        image_filename = os.path.basename(image_path)
        cache_path = os.path.join(self.args.cache_path, f"mcts_{image_filename}")
        os.makedirs(self.args.cache_path, exist_ok=True)
        best_image.save(cache_path)

        return cache_path
    
    def enhance_caption_object(self, data_row, obj_list, attr_list):

        # 增强物体描述

        # 获取样本信息
        image_path = data_row['image_path']
        key = data_row['key']
        image_key = data_row['image_key']

        # 如果是补充list，则补充并且去重
        _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
        obj_list = list(dict.fromkeys(obj_list + selected_objects))
        # 在多轮活动中交叠补充
        self.attention_object = list(dict.fromkeys(self.attention_object + obj_list))
        obj_list = list(dict.fromkeys(self.attention_object + obj_list))

        # 获取问题、答案、caption
        question, caption = self.dataset.question_dict[key], self.dataset.inputtext_dict[image_key]

        prompt = 'I am giving you a question, an image, and some supplementary information, but you do not need to answer it.\n'
        # # 详细描述
        # prompt += 'Please provide a detailed description of the specified object in the image based on the question I give you.\n'
        # 简要描述
        prompt += 'Please provide a concise description of the specified object in the image based on the question I give you.\n'
        prompt += 'Object: %s\n' % str(obj_list)

        response = self._call_llm(prompt, image_path=image_path)

        print('-----enhance_caption_object-----相关信息-----+++++-----beg')
        print('prompt:', prompt)
        print('response:', response)
        print('-----enhance_caption_object-----相关信息-----+++++-----end')

        return response
    
    def enhance_knowledge_object(self, data_row, obj_list, attr_list):

        # 补充物体相关知识

        # 获取样本信息
        image_path = data_row['image_path']
        key = data_row['key']
        question = data_row['question']

        # 如果是补充list，则补充并且去重
        _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
        obj_list = list(dict.fromkeys(obj_list + selected_objects))
        # 在多轮活动中交叠补充
        self.attention_object = list(dict.fromkeys(self.attention_object + obj_list))
        obj_list = list(dict.fromkeys(self.attention_object + obj_list))

        mode = self.args.knowledge_notes_mode
        if mode == "legacy":
            return self._legacy_generate_knowledge(data_row, obj_list, image_path)

        retrieved_items = []
        if mode in ("raw_retrieved", "notes", "hybrid"):
            retrieved_items = self.retrieve_knowledge_notes_candidates(question, obj_list)

        if mode == "raw_retrieved":
            knowledge = self._format_retrieved_knowledge(retrieved_items, self.args.knowledge_raw_max_chars)
            if not knowledge and self.args.knowledge_notes_fallback_legacy:
                knowledge = self._legacy_generate_knowledge(data_row, obj_list, image_path)
            return knowledge

        if mode == "retrieval_free":
            return self.generate_knowledge_notes(
                question=question,
                image_path=image_path,
                obj_list=obj_list,
                retrieved_knowledge="",
                retrieval_free=True,
            )

        retrieved_text = self._format_retrieved_knowledge(retrieved_items, self.args.knowledge_raw_max_chars)
        if not retrieved_text and self.args.knowledge_notes_fallback_legacy:
            return self._legacy_generate_knowledge(data_row, obj_list, image_path)

        notes = self.generate_knowledge_notes(
            question=question,
            image_path=image_path,
            obj_list=obj_list,
            retrieved_knowledge=retrieved_text,
            retrieval_free=False,
        )

        if mode == "hybrid" and retrieved_text:
            return "Knowledge Notes: %s\nRetrieved Knowledge: %s" % (
                self._truncate_text(notes, self.args.knowledge_notes_max_chars),
                self._truncate_text(retrieved_text, self.args.knowledge_raw_max_chars),
            )
        return notes

    def _legacy_generate_knowledge(self, data_row, obj_list, image_path):
        # 模型补充知识：保留原始 onion 行为，作为兼容 baseline。
        prompt = 'I am giving you a question, an image, and some supplementary information, but you do not need to answer it.\n'
        prompt += 'Please supplement additional knowledge about the specified target based on the question and image I provide, rather than information already present in the image.\n'
        prompt += 'Object: %s\n' % str(obj_list)

        response = self._call_llm(prompt, image_path=image_path)
        return response

    def _knowledge_tokenize(self, text):
        return set(re.findall(r"[a-z0-9]+", str(text).lower()))

    def _knowledge_record_text(self, record):
        if isinstance(record, str):
            return record
        if isinstance(record, dict):
            title = str(record.get("title") or record.get("name") or record.get("key") or "").strip()
            text = str(record.get("text") or record.get("contents") or record.get("description") or record.get("passage") or "").strip()
            if title and text:
                return "%s: %s" % (title, text)
            return title or text
        if isinstance(record, (list, tuple)):
            return " ".join(str(x) for x in record)
        return str(record)

    def _load_external_knowledge_corpus(self):
        if self.external_knowledge_corpus is not None:
            return self.external_knowledge_corpus

        corpus = []
        path = self.args.knowledge_corpus_file
        if not path:
            self.external_knowledge_corpus = corpus
            self.external_knowledge_index = []
            return corpus

        if not os.path.isfile(path):
            print(f"[knowledge_notes] missing external corpus file, skip: {path}")
            self.external_knowledge_corpus = corpus
            self.external_knowledge_index = []
            return corpus

        def add_record(key, value):
            if isinstance(value, list):
                text = " ".join(str(x) for x in value)
            else:
                text = self._knowledge_record_text(value)
            text = text.strip()
            if text:
                corpus.append({"title": str(key), "text": text})

        try:
            if path.endswith(".jsonl"):
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        text = self._knowledge_record_text(item).strip()
                        if text:
                            corpus.append(item if isinstance(item, dict) else {"text": text})
            elif path.endswith(".json"):
                data = json.load(open(path, "r"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        add_record(key, value)
                elif isinstance(data, list):
                    for item in data:
                        text = self._knowledge_record_text(item).strip()
                        if text:
                            corpus.append(item if isinstance(item, dict) else {"text": text})
            else:
                with open(path, "r") as f:
                    for line_id, line in enumerate(f):
                        text = line.strip()
                        if text:
                            corpus.append({"title": str(line_id), "text": text})
        except Exception as e:
            print(f"[knowledge_notes] failed to load corpus {path}: {e}")
            corpus = []

        self.external_knowledge_corpus = corpus
        self.external_knowledge_index = [
            self._knowledge_tokenize(self._knowledge_record_text(record)) for record in corpus
        ]
        print(f"[knowledge_notes] loaded external corpus: {path}, records={len(corpus)}")
        return corpus

    def retrieve_knowledge_notes_candidates(self, question, obj_list):
        candidates = []
        seen = set()
        query_terms = self._knowledge_tokenize(question)
        object_terms = []
        for obj in obj_list:
            object_terms.extend(sorted(self._knowledge_tokenize(obj)))
        query_terms.update(object_terms)

        if self.args.knowledge_use_wit:
            for obj in obj_list:
                obj_key = str(obj).strip()
                values = self.wit_knowkedge.get(obj_key) or self.wit_knowkedge.get(obj_key.lower())
                if not values:
                    continue
                text = " ".join(str(x) for x in values) if isinstance(values, list) else str(values)
                text = text.strip()
                if text and text not in seen:
                    seen.add(text)
                    candidates.append({"source": "wit", "title": obj_key, "text": text, "score": 999})

        corpus = self._load_external_knowledge_corpus()
        scored = []
        if corpus and self.args.knowledge_retrieval_mode in ("lexical", "hybrid"):
            for idx, record in enumerate(corpus):
                record_text = self._knowledge_record_text(record)
                terms = self.external_knowledge_index[idx] if self.external_knowledge_index else self._knowledge_tokenize(record_text)
                overlap = len(query_terms & terms)
                object_overlap = sum(1 for term in object_terms if term in terms)
                score = overlap + object_overlap * 2
                if score <= 0:
                    continue
                scored.append((score, idx, record_text))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, idx, text in scored[: self.args.knowledge_top_k]:
                if text and text not in seen:
                    seen.add(text)
                    title = corpus[idx].get("title", str(idx)) if isinstance(corpus[idx], dict) else str(idx)
                    candidates.append({"source": "corpus", "title": title, "text": text, "score": score})

        return candidates[: self.args.knowledge_top_k]

    def _format_retrieved_knowledge(self, retrieved_items, max_chars):
        lines = []
        for i, item in enumerate(retrieved_items, start=1):
            title = item.get("title", "")
            text = item.get("text", "")
            source = item.get("source", "knowledge")
            prefix = "Knowledge %d" % i
            if title:
                prefix += " (%s:%s)" % (source, title)
            lines.append("%s: %s" % (prefix, self._truncate_text(text, 500)))
        return self._truncate_text("\n".join(lines), max_chars)

    def generate_knowledge_notes(self, question, image_path, obj_list, retrieved_knowledge, retrieval_free=False):
        if retrieval_free:
            prompt = (
                "You are generating Knowledge Notes for a visual question answering system.\n"
                "Look at the image and question, but do not answer the question directly.\n"
                "Write concise background knowledge, typical-use knowledge, category knowledge, or commonsense "
                "that would help answer the question. If no extra knowledge is needed, write a short visual note.\n"
                "Question: %s\n"
                "Objects of interest: %s\n"
                "Return Knowledge Notes in no more than %d words."
                % (question, str(obj_list), self.args.knowledge_notes_max_words)
            )
        else:
            prompt = (
                "You are generating Knowledge Notes for a visual question answering system.\n"
                "Use the image and question to filter the retrieved knowledge. Keep only knowledge that is relevant "
                "to the image-question pair, and ignore misleading or unrelated passages.\n"
                "Do not answer the question directly. Produce concise notes that can help a later model answer.\n"
                "Question: %s\n"
                "Objects of interest: %s\n"
                "Retrieved knowledge:\n%s\n"
                "If the retrieved knowledge is not relevant, write a short image-grounded note instead.\n"
                "Return Knowledge Notes in no more than %d words."
                % (question, str(obj_list), retrieved_knowledge, self.args.knowledge_notes_max_words)
            )

        response = self._call_llm(
            prompt,
            image_path=image_path if self.args.knowledge_notes_use_image else None,
            max_new_tokens=self.args.knowledge_notes_max_tokens,
        )
        notes = self._truncate_text(response, self.args.knowledge_notes_max_chars)

        print('-----knowledge_notes-----相关信息-----+++++-----beg')
        print('mode:', self.args.knowledge_notes_mode)
        print('retrieval_free:', retrieval_free)
        print('objects:', obj_list)
        if retrieved_knowledge:
            print('retrieved_knowledge:', retrieved_knowledge)
        print('notes:', notes)
        print('-----knowledge_notes-----相关信息-----+++++-----end')

        return notes

    def load_wit_knowkedge(self):
        file_path = '/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/knowledge/deduplicated_merged_by_title.json'
        wit_knowkedge = json.load(open(file_path, 'r'))
        return wit_knowkedge

    # 从场景图中提取关键信息并生成标准化的问题
    def pick_example(self, key):
        image_key = int(key.split('<->')[0]) if self.args.dataset_name != "fvqa" else self.image_dict[key]  # for fvqa
        scene_graph_path = os.path.join(self.dataset.sg_attr_dir, str(image_key).zfill(12) + ".json")
        scene_graph_attr = json.load(open(scene_graph_path))
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if attr['class'] in ['girl', 'boy', 'man', 'woman'] and len(attr['attr']) > 0:
                description = attr['attr'][0]
                self.temp_question = f"What is the {description} {attr['class']} doing?"
                return True
        return False
    
    def load_caption_qwen(self):
        if getattr(self.args, "dataset_name", "") == "mme":
            return {}
        file_path = '/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/caption_onion/aokvqa_val_caption_8b_256.json'
        if not os.path.isfile(file_path):
            print(f"[caption_qwen] missing caption file, continue with empty captions: {file_path}")
            return {}
        caption_dict = json.load(open(file_path, 'r'))
        return caption_dict

    def get_context_keys(self, key, metric, n):
        """
        根据指定的度量方式获取最相似的n个上下文样本的键值。
        
        该函数用于检索与当前验证样本最相似的训练样本，以便为模型提供上下文示例。
        支持基于问题相似度和问题+图像联合相似度两种检索方式。
        
        Args:
            key: 当前验证样本的唯一标识符
            metric: 相似度计算方式，可选 'question'（仅问题相似度）或 'imagequestion'（问题+图像联合相似度）
            n: 需要返回的最相似样本数量
        
        Returns:
            list: 包含n个最相似训练样本键值的列表，如果metric参数无效则返回None
        """
        if n <= 0 or not getattr(self.dataset, "valkey2idx", None):
            return []
        
        if metric == 'question':
            # 仅基于问题相似度检索上下文样本
            
            # 将验证样本的键转换为索引ID
            lineid = self.dataset.valkey2idx[key]
            
            if self.args.pick_example_mode:
                # 动态计算模式：使用CLIP模型实时计算验证样本的问题特征
                # 对问题模板进行编码
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 通过CLIP模型获取问题特征
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                
                # 归一化特征向量
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                
                # 计算验证样本问题与所有训练样本问题的相似度
                similarity = np.matmul(self.dataset.train_feature, val_feature.detach()[0].numpy())
            else:
                # 预计算模式：使用预先计算好的验证集特征
                similarity = np.matmul(self.dataset.train_feature, self.dataset.val_feature[lineid, :])
            
            # 获取相似度最高的n个样本的索引（降序排列）
            index = similarity.argsort()[-n:][::-1]
            
            # 将索引转换回样本键值并返回
            return [self.dataset.train_idx[str(x)] for x in index]
        
        elif metric == 'imagequestion':
            # 基于问题+图像联合相似度检索上下文样本
            # 同时考虑问题语义相似度和图像特征相似度
            
            # 将验证样本的键转换为索引ID
            lineid = self.dataset.valkey2idx[key]
            
            # 计算问题相似度部分
            if self.args.pick_example_mode:
                # 动态计算模式：使用CLIP模型实时计算验证样本的问题特征
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                
                # 计算问题相似度
                question_similarity = np.matmul(self.dataset.train_feature, val_feature.detach()[0].numpy())
            else:
                # 预计算模式：使用预先计算好的验证集问题特征
                question_similarity = np.matmul(self.dataset.train_feature, self.dataset.val_feature[lineid, :])
            
            # 计算图像相似度部分，并与问题相似度相加得到联合相似度
            # 注意：这里将问题相似度和图像相似度简单相加，可以根据需要调整权重
            similarity = question_similarity + np.matmul(self.dataset.image_train_feature, self.dataset.image_val_feature[lineid, :])
            
            # 获取联合相似度最高的n个样本的索引（降序排列）
            index = similarity.argsort()[-n:][::-1]
            
            # 将索引转换回样本键值并返回
            return [self.dataset.train_idx[str(x)] for x in index]
        
        else:
            # 不支持的metric参数
            return None

    def get_related_obj_dict(self, key):
        if self.args.train_sim_metric == "rationale":
            return self.get_related_obj_dict_rationale(key)
        elif self.args.train_sim_metric == "answer":
            if not hasattr(self, "train_object_select"):
                self.train_object_select = pickle.load(open(self.args.train_sim_file, "rb"))
            return self.train_object_select[key]

    def get_related_obj_dict_rationale(self, key):
        image_context_key = int(key.split('<->')[0])
        context_scene_graph = json.load(open(os.path.join(self.dataset.sg_dir, str(image_context_key).zfill(12) + ".json")))
        context_scene_graph_attr = json.load(
            open(os.path.join(self.dataset.sg_attr_dir, str(image_context_key).zfill(12) + ".json")))

        obj_list = []
        for obj in context_scene_graph[0]:
            if obj['class'] not in obj_list:
                obj_list.append(obj['class'])
        for obj in context_scene_graph_attr[0]:
            if obj['class'] not in obj_list:
                obj_list.append(obj['class'])

        related_obj_dict = {}
        rationale = self.dataset.traincontext_rationale_dict[key]
        for obj in obj_list:
            for r in rationale:
                if obj in r:
                    if obj not in related_obj_dict:
                        related_obj_dict[obj] = 1
                    else:
                        related_obj_dict[obj] += 1
        return related_obj_dict

    def get_interactive_context_keys(self, key, metric, n):
        if metric == 'question':
            assert False
        elif metric == 'imagequestion':
            ## combined with Q-similairty (image+question)
            lineid = self.dataset.valkey2idx[key]
            if self.args.pick_example_mode:
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                question_similarity = np.matmul(self.dataset.train_feature, val_feature.detach()[0].numpy())
            else:
                question_similarity = np.matmul(self.dataset.train_feature, self.dataset.val_feature[lineid, :])
            ## end of Q-similairty
            similarity = question_similarity + np.matmul(self.dataset.image_train_feature, self.dataset.image_val_feature[lineid, :])
            similarity = similarity.argsort()
            idx_list = []
            rel_obj_list = []
            for i in range(len(similarity)):
                context_key = self.dataset.train_idx[str(similarity[-1 - i])]
                rel_obj_dict = self.get_related_obj_dict(context_key)
                if len(rel_obj_dict) > 0:
                    idx_list.append(context_key)
                    rel_obj_list.append(rel_obj_dict)
                if len(idx_list) >= n:
                    break
            return idx_list, rel_obj_list
        else:
            return None
    
    #qwen测试
    def initialize_qwen(self, model_name):
        # vLLM API模式：不加载本地模型，创建OpenAI client
        if self.args.use_vllm:
            from openai import OpenAI
            self.use_vllm = True
            self.vllm_client = OpenAI(base_url=self.args.vllm_url, api_key="not-needed")
            self.vllm_model_name = model_name
            self.model = None
            self.processor = None
            self.tokenizer = None
            return

        self.use_vllm = False
        self.vllm_client = None
        self.vllm_model_name = None

        if model_name == "qwen3-VL-2B":
            self.qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-2B-Instruct"
        elif model_name == "qwen3-VL-4B":
            self.qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-4B-Instruct"
        elif model_name == "qwen3-VL-8B":
            self.qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-8B-Instruct"
        elif model_name == "qwen3-VL-27B":
            self.qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3.6-27B"
        elif model_name == "qwen3-VL-30B":
            self.qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-30B-A3B-Instruct"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.qwen_path,
            dtype="auto",
            device_map="cuda")
        self.processor = AutoProcessor.from_pretrained(self.qwen_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.qwen_path)

    def _call_llm(self, prompt, image_path=None, max_new_tokens=512, use_images=True, history=None, return_history=False):
        if self.use_vllm:
            return chat_with_qwen_vllm(
                self.vllm_client, self.vllm_model_name, prompt,
                image_path=image_path, max_new_tokens=max_new_tokens,
                use_images=use_images, history=history, return_history=return_history
            )
        else:
            return chat_with_qwen_vl(
                self.model, self.processor, prompt,
                image_path=image_path, max_new_tokens=max_new_tokens,
                use_images=use_images, history=history, return_history=return_history
            )

    def initialize_lang_sam(self):
        with torch.no_grad():
            self.sam = LangSAM(
                gdino_model_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base", 
                gdino_processor_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base")

    def ensure_lang_sam(self):
        if self.sam is None:
            self.initialize_lang_sam()
            


# 参数解析器函数
def parser_args():

    parser = argparse.ArgumentParser()

    # LLM
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--use_vllm', action='store_true', help='use vLLM OpenAI-compatible API instead of local model loading')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1', help='vLLM server URL')
    # 实验超参数
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=5, help="number of ensemble (self-consistency samples)")
    parser.add_argument('--rounds', type=int, default=3, help="number of interactive rounds")
    # 单样本-调试
    parser.add_argument('--image_id', type=int, default=-1, help="selected image id pick example only")
    parser.add_argument('--pick_example_with_question_mode', action='store_true')
    parser.add_argument('--pick_example_mode', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # 数据分片（单卡多进程并行）
    parser.add_argument('--shard_id', type=int, default=0, help="shard id (0-indexed)")
    parser.add_argument('--num_shards', type=int, default=1, help="total number of shards")
    parser.add_argument('--max_samples_per_shard', type=int, default=-1,
                        help='optional cap on processed samples per shard for smoke/profile runs')
    # 汇总模式：不推理，只读取prompt_samples目录计算全量准确率
    parser.add_argument('--merge_only', action='store_true', help="merge shard results and compute accuracy")
    parser.add_argument('--summary_log', type=str, default='', help="path to write accuracy summary line")
    # 实验类型-模型结构
    parser.add_argument('--choice_only', action='store_true')
    parser.add_argument('--eval_all_direct_answers', action='store_true',
                        help='internal analysis only: include difficult_direct_answer=True samples in DA aggregate')
    parser.add_argument('--legacy_answer_normalization', action='store_true',
                        help='internal analysis only: use the old normalized 0.3*match direct-answer score')
    parser.add_argument('--answer_postprocess', type=str, default='none',
                        choices=['none', 'safe_rules', 'legacy_visualcot'],
                        help='optional prediction post-processing before scoring/voting')
    parser.add_argument('--direct_prompt_style', type=str, default='default',
                        choices=['default', 'answer_first_strict', 'type_specialist', 'context_gated'],
                        help='direct-answer prompt/context variant used when --chain_of_thoughts is off')
    parser.add_argument('--chain_of_thoughts', action='store_true')
    parser.add_argument('--with_clip_verify', action='store_true')
    parser.add_argument('--use_clip_thought_verify', action='store_true',
                        help='filter chain-of-thought sentences by CLIP image/text similarity')
    parser.add_argument('--verify_threshold', type=float, default=0.2,
                        help='CLIP similarity threshold for thought verification')
    parser.add_argument('--use_qwen_blip2_caption', action='store_true',
                        help='use Qwen-VL as a BLIP2-style global/local visual captioner')
    parser.add_argument('--qwen_caption_mode', type=str, default='both', choices=['both', 'global', 'local'],
                        help='which Qwen caption helper to use when --use_qwen_blip2_caption is enabled')
    parser.add_argument('--qwen_caption_no_final_context', action='store_true',
                        help='query Qwen captions but do not inject them into the final answer prompt')
    parser.add_argument('--use_qwen_blip2_thought_verify', action='store_true',
                        help='use Qwen-VL as a BLIP2-style image support checker for thoughts')
    parser.add_argument('--qwen_caption_max_tokens', type=int, default=160,
                        help='max tokens for Qwen visual caption helper')
    parser.add_argument('--qwen_caption_final_max_chars', type=int, default=700,
                        help='max characters of each Qwen caption injected into final prompt')
    parser.add_argument('--max_thought_verify_sentences', type=int, default=8,
                        help='max CoT sentences checked by Qwen visual verifier')
    # ----交互策略
    parser.add_argument('--iterative_strategy', type=str, default="caption", help="caption or sg")
    # ----三大增强模块开关（消融实验用）
    parser.add_argument('--use_image_enhance', action='store_true', help="enable image enhancement module")
    parser.add_argument('--use_caption_enhance', action='store_true', help="enable caption enhancement module")
    parser.add_argument('--use_knowledge_enhance', action='store_true', help="enable knowledge enhancement module")
    parser.add_argument('--knowledge_notes_mode', type=str, default='legacy',
                        choices=['legacy', 'retrieval_free', 'raw_retrieved', 'notes', 'hybrid'],
                        help='knowledge enhancement mode: legacy Qwen commonsense, retrieval-free notes, raw retrieved knowledge, NoteMR-style notes, or notes+raw hybrid')
    parser.add_argument('--knowledge_enhance_trigger', type=str, default='routed',
                        choices=['routed', 'always', 'knowledge_qtype'],
                        help='when --use_knowledge_enhance should run: onion-routed only, every sample, or knowledge/category question types')
    parser.add_argument('--knowledge_corpus_file', type=str, default='',
                        help='optional JSON/JSONL/TXT external knowledge corpus for Knowledge Notes retrieval')
    parser.add_argument('--knowledge_retrieval_mode', type=str, default='hybrid',
                        choices=['lexical', 'hybrid'],
                        help='retrieval strategy for external knowledge corpus')
    parser.add_argument('--knowledge_top_k', type=int, default=5,
                        help='maximum retrieved knowledge passages used for Knowledge Notes')
    parser.add_argument('--knowledge_use_wit', action='store_true',
                        help='include local WIT/object-title knowledge as retrieval candidates')
    parser.add_argument('--knowledge_notes_use_image', action='store_true',
                        help='let the Knowledge Notes generator inspect the image')
    parser.add_argument('--knowledge_notes_fallback_legacy', action='store_true',
                        help='fall back to legacy Qwen knowledge generation when no retrieved knowledge is found')
    parser.add_argument('--knowledge_notes_max_words', type=int, default=80,
                        help='word budget requested from the Knowledge Notes generator')
    parser.add_argument('--knowledge_notes_max_tokens', type=int, default=128,
                        help='max new tokens for Knowledge Notes generation')
    parser.add_argument('--knowledge_notes_max_chars', type=int, default=700,
                        help='max Knowledge Notes characters injected into the final context')
    parser.add_argument('--knowledge_raw_max_chars', type=int, default=1200,
                        help='max raw retrieved knowledge characters injected or shown to notes generator')
    parser.add_argument('--mcts_n_simulations', type=int, default=20, help="number of MCTS simulations for image enhancement")
    parser.add_argument('--mcts_trigger_mode', type=str, default='all',
                        choices=['all', 'visual_detail_only', 'count_color_object_only'],
                        help='controls which questions can trigger MCTS image enhancement')
    parser.add_argument('--mcts_action_mode', type=str, default='all',
                        choices=['all', 'outline_only', 'marker_only', 'no_crop', 'dyfo_evidence'],
                        help='controls the MCTS image operation set')
    parser.add_argument('--mcts_filter_objects', action='store_true',
                        help='filter generic MCTS key objects and align them to selected scene-graph objects')
    parser.add_argument('--use_dyfo_visual_evidence', action='store_true',
                        help='inject DyFo-style visual evidence into final answer/reviewer prompts')
    parser.add_argument('--dyfo_trigger_mode', type=str, default='visual_detail',
                        choices=['always', 'never', 'visual_detail', 'mcts'],
                        help='which questions can trigger DyFo-style visual evidence search')
    parser.add_argument('--dyfo_n_simulations', type=int, default=6,
                        help='number of DyFo-style focus-tree simulations')
    parser.add_argument('--dyfo_max_depth', type=int, default=3,
                        help='maximum depth for DyFo-style focus-tree search')
    parser.add_argument('--dyfo_exploration_weight', type=float, default=1.0,
                        help='UCT exploration weight for DyFo-style focus-tree search')
    parser.add_argument('--dyfo_scatter_scale', type=float, default=1.6,
                        help='semantic scatter expansion scale')
    parser.add_argument('--dyfo_focus_padding', type=float, default=1.2,
                        help='padding scale around localized focus boxes')
    parser.add_argument('--dyfo_area_reward', type=str, default='compact',
                        choices=['compact', 'paper'],
                        help='compact rewards small consistent regions; paper uses the raw area ratio')
    parser.add_argument('--dyfo_text_focus_use_image', action='store_true',
                        help='let Qwen see the current crop while updating the textual focus')
    parser.add_argument('--dyfo_use_focus_image_as_answer', action='store_true',
                        help='answer on the best DyFo focus crop instead of only injecting evidence')
    parser.add_argument('--dyfo_decision_mode', type=str, default='evidence_inject',
                        choices=['evidence_inject', 'best_focus_answer', 'weighted_vote'],
                        help='DyFo final decision: inject evidence into the normal answer path, answer from the best focus node, or reward-weighted vote over focus nodes')
    parser.add_argument('--dyfo_focus_max_tokens', type=int, default=32,
                        help='max tokens for textual focus generation')
    parser.add_argument('--dyfo_answer_max_tokens', type=int, default=32,
                        help='max tokens for each DyFo focus-node local answer')
    parser.add_argument('--dyfo_evidence_max_tokens', type=int, default=96,
                        help='max tokens for final DyFo visual evidence generation')
    parser.add_argument('--dyfo_evidence_context_max_chars', type=int, default=700,
                        help='max characters of DyFo visual evidence injected into answer context')
    parser.add_argument('--use_all_regional_captions', action='store_true',
                        help='inject top regional captions instead of selecting a few objects over multiple rounds')
    parser.add_argument('--max_regional_captions', type=int, default=25,
                        help='maximum number of regional captions injected by all-regional mode')
    parser.add_argument('--use_ocr_context', action='store_true',
                        help='load OCR text and inject OCR context when available')
    parser.add_argument('--ocr_train_file', type=str, default='',
                        help='optional path to coco17_ocr_train.json')
    parser.add_argument('--ocr_val_file', type=str, default='',
                        help='optional path to coco17_ocr_val/test.json')
    parser.add_argument('--ocr_conf_threshold', type=float, default=0.2,
                        help='minimum OCR confidence')
    parser.add_argument('--ensemble_strategy', type=str, default='majority',
                        choices=['majority', 'normalized_majority', 'first'],
                        help='how to select final answer from n_ensemble candidates')
    parser.add_argument('--context_mode', type=str, default='full',
                        choices=['full', 'empty', 'caption_only', 'objects_only', 'no_round_state'],
                        help='controls current-sample brief context injected into final prompt')
    parser.add_argument('--answer_extraction_strategy', type=str, default='current',
                        choices=['current', 'strict_final', 'last_line', 'raw'],
                        help='how to extract a short answer from CoT responses before voting')
    parser.add_argument('--cot_style', type=str, default='step_by_step',
                        choices=['step_by_step', 'compact', 'answer_first', 'answer_first_locked',
                                 'visual_facts', 'direct_verify', 'reviewer_evidence',
                                 'reflective_answer_first', 'adaptive_reflective_answer_first',
                                 'candidate_judge', 'protected_reflective', 'rag_strategy_router',
                                 'multi_strategy_router', 'complex_decompose',
                                 'direct_rephrase_consistency'],
                        help='prompt style used when --chain_of_thoughts is enabled')
    parser.add_argument('--rephrase_num_questions', type=int, default=3,
                        help='number of semantically equivalent questions generated by direct_rephrase_consistency')
    parser.add_argument('--rephrase_generation_mode', type=str, default='mixed',
                        choices=['simple', 'visual_focus', 'answer_type', 'mixed'],
                        help='how rephrased questions should vary')
    parser.add_argument('--rephrase_trigger', type=str, default='always',
                        choices=['always', 'risky_qtype', 'complex_qtype'],
                        help='which questions trigger direct_rephrase_consistency')
    parser.add_argument('--rephrase_arbitration', type=str, default='conservative_review',
                        choices=['keep_baseline', 'majority_if_consensus', 'all_agree', 'conservative_review'],
                        help='how rephrase answers are allowed to override initial direct answer')
    parser.add_argument('--rephrase_consensus_threshold', type=int, default=2,
                        help='minimum rephrased answer votes needed to propose a non-baseline answer')
    parser.add_argument('--rephrase_answer_context', type=str, default='same',
                        choices=['same', 'empty', 'regional', 'ocr_regional'],
                        help='context visible when answering rephrased questions')
    parser.add_argument('--rephrase_context_max_chars', type=int, default=900,
                        help='maximum context characters visible to rephrase answer/review prompts')
    parser.add_argument('--rephrase_generation_max_tokens', type=int, default=128,
                        help='max tokens for generating rephrased questions')
    parser.add_argument('--rephrase_answer_max_tokens', type=int, default=16,
                        help='max tokens for answering each rephrased question')
    parser.add_argument('--rephrase_review_max_tokens', type=int, default=96,
                        help='max tokens for conservative rephrase reviewer')
    parser.add_argument('--decompose_complexity_mode', type=str, default='adaptive',
                        choices=['always', 'adaptive', 'conservative', 'never'],
                        help='which questions are decomposed by --cot_style complex_decompose')
    parser.add_argument('--decompose_verify', action='store_true',
                        help='conservatively verify decomposed answer against direct answer')
    parser.add_argument('--decompose_context_max_chars', type=int, default=1400,
                        help='maximum context characters visible to decomposition prompts')
    parser.add_argument('--reflect_rounds', type=int, default=3,
                        help='number of answer/evidence/review stages for --cot_style reflective_answer_first')
    parser.add_argument('--reflect_trigger_mode', type=str, default='always',
                        choices=['always', 'high_risk', 'low_confidence', 'high_risk_or_low_confidence'],
                        help='when adaptive_reflective_answer_first should run evidence/review')
    parser.add_argument('--reflect_evidence_mode', type=str, default='default',
                        choices=['default', 'visible_only'],
                        help='controls whether reflective evidence can include commonsense/typical-use statements')
    parser.add_argument('--reflect_review_format', type=str, default='final_answer',
                        choices=['final_answer', 'keep_revise'],
                        help='controls reflective reviewer output format and extraction')
    parser.add_argument('--reflect_review_context', type=str, default='same',
                        choices=['same', 'empty'],
                        help='whether reflective evidence/review sees the same context as round 1 or no text context')
    parser.add_argument('--reflect_initial_ensemble', type=int, default=1,
                        help='number of direct first-answer calls before a single reflective review')
    parser.add_argument('--direct_verify_policy', type=str, default='balanced',
                        choices=['balanced', 'keep_stronger', 'conflict_only', 'revise_freely', 'no_fallback'],
                        help='revision policy used by --cot_style direct_verify')
    parser.add_argument('--disable_direct_verify_fallback', action='store_true',
                        help='do not fall back to the initial answer when direct_verify returns a cue-like answer')
    parser.add_argument('--reviewer_evidence_scope', type=str, default='all',
                        choices=['all', 'caption_object', 'caption_only', 'object_only', 'enhance_only',
                                 'no_caption', 'no_objects', 'selective'],
                        help='which evidence providers are visible to --cot_style reviewer_evidence')
    parser.add_argument('--reviewer_disable_enhanced_image', action='store_true',
                        help='for --cot_style reviewer_evidence, keep reviewer on the original image even when MCTS creates an enhanced image')
    parser.add_argument('--candidate_judge_consensus_votes', type=int, default=2,
                        help='minimum matching candidate answers needed to skip the judge in --cot_style candidate_judge')
    parser.add_argument('--candidate_judge_always_judge', action='store_true',
                        help='always run the final candidate judge even when multiple candidates agree')
    parser.add_argument('--candidate_judge_allow_new_answer', action='store_true',
                        help='allow candidate judge to output an answer not present in the candidate set')
    parser.add_argument('--candidate_judge_include_caption_candidate', action='store_true',
                        help='add an extra caption-only candidate answer in --cot_style candidate_judge')
    parser.add_argument('--candidate_judge_route_evidence', action='store_true',
                        help='route image/caption/knowledge enhancement by question type in --cot_style candidate_judge')
    parser.add_argument('--candidate_judge_use_enhanced_image', action='store_true',
                        help='let the candidate judge inspect the enhanced image instead of the original image when available')
    parser.add_argument('--candidate_judge_include_count_candidate', action='store_true',
                        help='add a counting-specialist candidate for count questions')
    parser.add_argument('--candidate_judge_include_ocr_candidate', action='store_true',
                        help='add an OCR/text-specialist candidate for text-reading questions')
    parser.add_argument('--candidate_judge_include_coverage_candidate', action='store_true',
                        help='add a full coverage scan candidate using regional/OCR/enhanced evidence')
    parser.add_argument('--candidate_judge_include_contrast_candidate', action='store_true',
                        help='add a contrastive alternative candidate to fight wrong consensus')
    parser.add_argument('--strategy_name', type=str, default='default',
                        help='strategy label written to --strategy_profile_output')
    parser.add_argument('--strategy_profile_output', type=str, default='',
                        help='append per-sample strategy correctness records to this JSONL file')
    parser.add_argument('--strategy_profile_path', type=str, default='',
                        help='combined JSONL strategy profile used by --cot_style rag_strategy_router')
    parser.add_argument('--strategy_direct_name', type=str, default='direct',
                        help='strategy-profile key for the direct baseline')
    parser.add_argument('--strategy_cot_name', type=str, default='protected_reflective',
                        help='strategy-profile key for the CoT/protected strategy')
    parser.add_argument('--strategy_router_default', type=str, default='direct',
                        help='fallback strategy when RAG evidence is weak')
    parser.add_argument('--strategy_cot_runtime', type=str, default='protected_reflective',
                        choices=['protected_reflective', 'answer_first_locked', 'complex_decompose'],
                        help='runtime behavior when rag_strategy_router selects the CoT strategy')
    parser.add_argument('--strategy_router_mode', type=str, default='conservative_risk',
                        choices=['direct_failure', 'direct_vs_complex', 'qtype_conditional',
                                 'conservative_risk', 'legacy'],
                        help='train-profile routing rule used by --cot_style rag_strategy_router')
    parser.add_argument('--strategy_retrieval_metric', type=str, default='imagequestion',
                        choices=['question', 'imagequestion'],
                        help='retrieval metric for strategy RAG router')
    parser.add_argument('--strategy_topk', type=int, default=20,
                        help='number of strategy-profile neighbors used by RAG router')
    parser.add_argument('--strategy_min_neighbors', type=int, default=5,
                        help='minimum available profiled neighbors before routing away from default')
    parser.add_argument('--strategy_margin', type=float, default=0.12,
                        help='minimum cot_avg - direct_avg needed to select CoT')
    parser.add_argument('--strategy_direct_hard_threshold', type=float, default=0.0,
                        help='neighbor score at or below this is treated as direct-hard')
    parser.add_argument('--strategy_direct_safe_threshold', type=float, default=0.6,
                        help='neighbor score at or above this is treated as direct-safe / complex-win')
    parser.add_argument('--strategy_min_direct_hard_rate', type=float, default=0.55,
                        help='minimum direct-hard neighbor rate for direct_failure routing')
    parser.add_argument('--strategy_min_complex_win_rate', type=float, default=0.20,
                        help='minimum neighbor rate where complex clearly beats failed direct')
    parser.add_argument('--strategy_min_rescue_rate', type=float, default=0.15,
                        help='minimum neighbor rate where direct is wrong and CoT is right')
    parser.add_argument('--strategy_max_damage_rate', type=float, default=0.10,
                        help='maximum neighbor rate where direct is right and CoT is wrong')
    parser.add_argument('--strategy_min_net_gain', type=float, default=0.08,
                        help='minimum rescue_rate - damage_rate for conservative_risk routing')
    parser.add_argument('--multi_strategy_names', type=str,
                        default='direct,reflective_r3,answer_first_no_caption,marker_mcts',
                        help='comma-separated strategy names available to --cot_style multi_strategy_router')
    parser.add_argument('--multi_strategy_default', type=str, default='direct',
                        help='default strategy for --cot_style multi_strategy_router')
    parser.add_argument('--multi_strategy_margin', type=float, default=0.08,
                        help='minimum best_strategy_avg - default_avg needed to route away from default')
    # ----caption策略
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    # 数据集选择-验证测试
    parser.add_argument('--dataset_name', type=str, default='aokvqa', help='aokvqa, okvqa, pope, mme')
    parser.add_argument('--split_name', type=str, default='val', help='train, val, test')
    # 描述文本选择
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg, vinvl_ocr')
    # 路径相关
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--cache_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/cache')
    # 不确定要不要修改的路径
    parser.add_argument('--raw_image_dir', type=str, default="/data2/lizhengxue/datasets/coco17")
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--concept_caption_path', type=str, default='scene_graph_coco17_caption')
    parser.add_argument('--sg_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text')
    parser.add_argument('--similarity_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_clip_new')
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--train_sim_metric', type=str, default='rationale')
    parser.add_argument('--train_sim_file', type=str, default='')
    parser.add_argument('--val_sim_file', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='/data2/lizhengxue/datasets/aokvqa')
    parser.add_argument('--coco_annotation_path', type=str, default='/data2/lizhengxue/datasets/coco17/annotations',
                        help='COCO caption/instance annotation directory, separated from VQA annotation directories')
    parser.add_argument('--aokvqa_context_path', type=str,
                        default='/data2/lizhengxue/datasets/aokvqa',
                        help='A-OKVQA annotation directory reused as few-shot context for datasets without train annotations')
    parser.add_argument('--mme_manifest_file', type=str, default='',
                        help='Prepared MME jsonl manifest with materialized image paths.')
    parser.add_argument('--valcaption_file', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/vinvl_caption/VinVL_base_val2014.tsv')

    args = parser.parse_args()

    return args


def load_official_da_eval_keys(args):
    if args.dataset_name == "mme":
        answer_by_key, official_keys = load_mme_answer_annotations(args)
        return official_keys
    if args.dataset_name == "pope":
        answer_by_key, official_keys = load_direct_answer_annotations(args)
        return official_keys
    if args.dataset_name == "okvqa":
        question_file = os.path.join(args.coco_path, f"OpenEnded_mscoco_{args.split_name}2014_questions.json")
        try:
            questions = json.load(open(question_file, "r"))["questions"]
        except FileNotFoundError:
            return set()
        return {
            str(sample["image_id"]) + "<->" + str(sample["question_id"])
            for sample in questions
        }

    anno_file = os.path.join(args.coco_path, f"aokvqa_v1p0_{args.split_name}.json")
    try:
        annotations = json.load(open(anno_file, "r"))
    except FileNotFoundError:
        return set()
    return {
        str(sample["image_id"]) + "<->" + str(sample["question_id"])
        for sample in annotations
        if sample.get("difficult_direct_answer") is False
    }


def load_direct_answer_annotations(args):
    if args.dataset_name == "mme":
        return load_mme_answer_annotations(args)

    if args.dataset_name == "pope":
        subsets = ["random", "popular", "adversarial"] if args.split_name == "all" else [args.split_name]
        answer_by_key = {}
        official_keys = set()
        for subset in subsets:
            anno_file = os.path.join(args.coco_path, f"coco_pope_{subset}.json")
            try:
                f = open(anno_file, "r")
            except FileNotFoundError:
                continue
            with f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    image_id = int(sample["image"].split("_")[-1].split(".")[0])
                    key = f"{image_id}<->{subset}_{sample['question_id']}"
                    answer_by_key[key] = [str(sample["label"]).lower()]
                    official_keys.add(key)
        return answer_by_key, official_keys

    if args.dataset_name == "okvqa":
        answer_file = os.path.join(args.coco_path, f"mscoco_{args.split_name}2014_annotations.json")
        try:
            annotations = json.load(open(answer_file, "r"))["annotations"]
        except FileNotFoundError:
            return {}, set()
        answer_by_key = {}
        official_keys = set()
        for sample in annotations:
            key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            answer_by_key[key] = [ans["answer"] for ans in sample.get("answers", [])]
            official_keys.add(key)
        return answer_by_key, official_keys

    anno_file = os.path.join(args.coco_path, f"aokvqa_v1p0_{args.split_name}.json")
    try:
        annotations = json.load(open(anno_file, "r"))
    except FileNotFoundError:
        return {}, set()
    answer_by_key = {}
    official_keys = set()
    for sample in annotations:
        key = str(sample["image_id"]) + "<->" + str(sample["question_id"])
        answer_by_key[key] = sample.get("direct_answers", [])
        if sample.get("difficult_direct_answer") is False:
            official_keys.add(key)
    return answer_by_key, official_keys


def direct_answer_eval_report(args, answers):
    if args.choice_only:
        acc = sum(float(a[3]) for a in answers) if answers else 0.0
        total = len(answers)
        pct = acc * 100.0 / total if total else 0.0
        return {
            "primary_label": "MC准确率",
            "primary_pct": pct,
            "primary_sum": acc,
            "primary_total": total,
            "lines": [f"MC准确率: {pct:.2f}% ({acc:.2f}/{total})"],
        }

    answer_by_key, official_keys = load_direct_answer_annotations(args)
    official_scores = []
    legacy_official_scores = []
    official_full_scores = []
    legacy_all_scores = []

    for a in answers:
        key = a[0]
        pred = a[1]
        gold = answer_by_key.get(key)
        if gold is None:
            continue
        if args.dataset_name in ("pope", "mme"):
            official_score = yes_no_answer_score(pred, gold)
            legacy_score = official_score
        else:
            official_score = official_direct_answer_score(pred, gold)
            legacy_score = legacy_normalized_direct_answer_score(pred, gold)
        official_full_scores.append(official_score)
        legacy_all_scores.append(legacy_score)
        if key in official_keys:
            official_scores.append(official_score)
            legacy_official_scores.append(legacy_score)

    def _summarize(scores):
        total = len(scores)
        score_sum = sum(scores)
        pct = score_sum * 100.0 / total if total else 0.0
        return pct, score_sum, total

    official_pct, official_sum, official_total = _summarize(official_scores)
    legacy_official_pct, legacy_official_sum, legacy_official_total = _summarize(legacy_official_scores)
    official_full_pct, official_full_sum, official_full_total = _summarize(official_full_scores)
    legacy_full_pct, legacy_full_sum, legacy_full_total = _summarize(legacy_all_scores)

    if args.dataset_name == "pope":
        primary_label = "POPE准确率"
    elif args.dataset_name == "mme":
        primary_label = "MME准确率"
    else:
        primary_label = "OK-VQA准确率" if args.dataset_name == "okvqa" else "官方DA准确率"
    if args.eval_all_direct_answers:
        if args.dataset_name == "pope":
            primary_label = "全量POPE诊断"
        elif args.dataset_name == "mme":
            primary_label = "全量MME诊断"
        else:
            primary_label = "全量OK-VQA诊断" if args.dataset_name == "okvqa" else "全量官方DA诊断"
        primary_pct, primary_sum, primary_total = official_full_pct, official_full_sum, official_full_total
    else:
        primary_pct, primary_sum, primary_total = official_pct, official_sum, official_total

    return {
        "primary_label": primary_label,
        "primary_pct": primary_pct,
        "primary_sum": primary_sum,
        "primary_total": primary_total,
        "official_pct": official_pct,
        "official_sum": official_sum,
        "official_total": official_total,
        "legacy_official_pct": legacy_official_pct,
        "legacy_official_sum": legacy_official_sum,
        "legacy_official_total": legacy_official_total,
        "official_full_pct": official_full_pct,
        "official_full_sum": official_full_sum,
        "official_full_total": official_full_total,
        "legacy_full_pct": legacy_full_pct,
        "legacy_full_sum": legacy_full_sum,
        "legacy_full_total": legacy_full_total,
        "lines": [
            f"{'POPE准确率' if args.dataset_name == 'pope' else ('MME准确率' if args.dataset_name == 'mme' else ('OK-VQA准确率' if args.dataset_name == 'okvqa' else '官方DA准确率'))}: {official_pct:.2f}% ({official_sum:.2f}/{official_total})",
            f"旧指标@{'POPE' if args.dataset_name == 'pope' else ('MME' if args.dataset_name == 'mme' else ('OK-VQA' if args.dataset_name == 'okvqa' else '官方DA子集'))}: {legacy_official_pct:.2f}% ({legacy_official_sum:.2f}/{legacy_official_total})",
            f"{'全量POPE诊断' if args.dataset_name == 'pope' else ('全量MME诊断' if args.dataset_name == 'mme' else ('全量OK-VQA诊断' if args.dataset_name == 'okvqa' else '全量官方DA诊断'))}: {official_full_pct:.2f}% ({official_full_sum:.2f}/{official_full_total})",
            f"旧指标@全量诊断: {legacy_full_pct:.2f}% ({legacy_full_sum:.2f}/{legacy_full_total})",
        ],
    }


def official_da_eval_answers(args, answers):
    if args.choice_only or args.eval_all_direct_answers:
        eval_answers = answers
        label = "全量准确率"
    else:
        eval_keys = load_official_da_eval_keys(args)
        eval_answers = [a for a in answers if a[0] in eval_keys]
        if args.dataset_name == "pope":
            label = "POPE准确率"
        elif args.dataset_name == "mme":
            label = "MME准确率"
        else:
            label = "OK-VQA准确率" if args.dataset_name == "okvqa" else "官方DA准确率"
    if not eval_answers:
        return 0.0, 0.0, 0, label
    acc = sum(float(a[3]) for a in eval_answers)
    return acc * 100.0 / len(eval_answers), acc, len(eval_answers), label


def write_official_prediction_file(args, answers, output_dir, output_name):
    predictions = {}
    for a in answers:
        qid = a[0].split('<->')[1] if '<->' in a[0] else a[0]
        if args.choice_only:
            predictions[qid] = {"multiple_choice": a[1]}
        else:
            predictions[qid] = {"direct_answer": a[1]}
    out_path = os.path.join(output_dir, f"predictions_{args.split_name}_{output_name}")
    json.dump(predictions, open(out_path, "w"))
    print(f"[merge] official prediction 已保存: {out_path}")


def merge_results(args):
    """汇总多shard的逐样本推理结果，计算全量准确率并生成最终JSON。"""
    import glob

    prompt_dir = os.path.join(args.output_path, "prompt_samples")
    format_dir = os.path.join(args.output_path, "format_samples")

    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "sample_*.json")))
    if not prompt_files:
        print(f"[merge] 错误: {prompt_dir} 中没有找到 sample_*.json 文件")
        return

    print(f"[merge] 找到 {len(prompt_files)} 个样本文件")

    answers = []
    full_answers = []

    for fpath in prompt_files:
        with open(fpath) as f:
            entry = json.load(f)
        answers.append(entry)

        basename = os.path.basename(fpath)
        format_fpath = os.path.join(format_dir, basename)
        if os.path.isfile(format_fpath):
            with open(format_fpath) as f:
                full_answers.append(json.load(f))

    report = direct_answer_eval_report(args, answers)
    acc_pct = report["primary_pct"]

    print(f"\n{'='*50}")
    for line in report["lines"]:
        print(line)
    print(f"{'='*50}\n")

    # 如果指定了summary_log，将准确率写入汇总日志
    if args.summary_log:
        with open(args.summary_log, 'a') as f:
            for line in report["lines"]:
                f.write(line + "\n")

    # 生成合并后的最终JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_answer_dir = os.path.join(args.output_path, f"prompt_answer_{timestamp}")
    format_answer_dir = os.path.join(args.output_path, f"format_answer_{timestamp}")
    os.makedirs(prompt_answer_dir, exist_ok=True)
    os.makedirs(format_answer_dir, exist_ok=True)

    output_name = f"VisualCOT_{args.caption_type}_n{args.n_shot}_repeat{args.n_ensemble}_{args.similarity_metric}_{acc_pct:.2f}.json"
    json.dump(full_answers, open(os.path.join(prompt_answer_dir, output_name), 'w'))
    print(f"[merge] prompt_answer 已保存: {prompt_answer_dir}/{output_name}")

    format_prediction = []
    for a in answers:
        rec = {
            "answer": a[1],
            "question_id": a[0].split('<->')[1] if '<->' in a[0] else a[0],
        }
        if args.chain_of_thoughts and len(a) > 5:
            rec["thoughts"] = a[5]
        format_prediction.append(rec)

    json.dump(format_prediction, open(os.path.join(format_answer_dir, output_name), 'w'))
    print(f"[merge] format_answer 已保存: {format_answer_dir}/{output_name}")
    write_official_prediction_file(args, answers, format_answer_dir, output_name)


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    args = parser_args()

    # 汇总模式：不加载模型，直接从prompt_samples目录计算准确率
    if args.merge_only:
        merge_results(args)
        return

    # 数据集准备
    if args.dataset_name == "okvqa":
        aokvqa_data = okvqa_dataset(args)
    elif args.dataset_name == "pope":
        aokvqa_data = pope_dataset(args)
    elif args.dataset_name == "mme":
        aokvqa_data = mme_dataset(args)
    else:
        aokvqa_data = aokvqa_dataset(args)

    aokvqa_onion = onion(args, dataset=aokvqa_data)

    # 生成推理结果
    # answers是所有问题的答案列表,full_answers是包含更多信息的完整答案列表
    answers, full_answers = aokvqa_onion.inference(save_every_step = True)

    prediction = {}
    for answer in answers:
        prediction[answer[0]] = [answer[1], answer[2]]

    format_prediction = []
    for answer in answers:
        if args.chain_of_thoughts:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1],
                                      "thoughts": answer[5]})
        else:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1]})

    report = direct_answer_eval_report(args, answers)
    acc = report["primary_pct"]
    for line in report["lines"]:
        print(line)

    ## if save final predictions
    # 获取当前日期时间戳
    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y%m%d_%H%M%S")
    print(f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")  # 输出可读时间
    # 创建带日期戳的文件夹
    os.system("mkdir -p %s/prompt_answer_%s" % (args.output_path, date_str))
    os.system("mkdir -p %s/format_answer_%s" % (args.output_path, date_str))
    output_name = 'VisualCOT_%s_n%d_repeat%d_%s_%f.json' % (args.caption_type, args.n_shot, args.n_ensemble, args.similarity_metric, acc)
    json.dump(full_answers, open("%s/prompt_answer_%s/%s" % (args.output_path, date_str, output_name), 'w'))
    json.dump(format_prediction, open("%s/format_answer_%s/%s" % (args.output_path, date_str, output_name), 'w'))
    write_official_prediction_file(args, answers, "%s/format_answer_%s" % (args.output_path, date_str), output_name)

if __name__ == '__main__':
    main()
