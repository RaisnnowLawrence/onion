"""VisualCoT/Onion inference entry point.

The file is intentionally kept executable for compatibility with existing
experiment scripts.  Its contents are grouped by responsibility: scoring,
inference orchestration, enhancement strategies, CLI configuration, and
evaluation/output handling.
"""

import base64
import csv
import datetime
import glob
import gzip
import heapq
import json
import math
import os
import pdb
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)

from lang_sam import LangSAM

from qwen_utils import chat_with_qwen_vl, chat_with_qwen_vllm, string_to_list_if_possible
from mcts import MCTSQuestionSample
from official_vqa_answer_processor import normalize_vqa_answer


from onion_evaluation import (
    direct_answer_eval_report,
    legacy_normalized_direct_answer_score,
    merge_results,
    official_direct_answer_score,
    process_answer,
    write_official_prediction_file,
    yes_no_answer_score,
)
from onion_cli import parser_args
from dataset_utils import (
    build_dataset,
    can_skip_scene_graph,
    image_key_from_sample_key,
    is_dataset,
    uses_yes_no_prompt,
    uses_yes_no_scoring,
)


class Onion:
    """Run Onion inference and optional visual/knowledge enhancements."""

    def __init__(self, args, dataset):

        self.dataset = dataset
        self.args = args
        self.messages = None
        self.attention_object = []
        self.qwen_global_caption_cache = {}
        self.qwen_local_caption_cache = {}
        self.external_knowledge_corpus = None
        self.external_knowledge_index = None
        self.external_knowledge_source_counts = {}
        self.sample_knowledge_cache = None
        self.wikidata_kat_cache = None
        self.strategy_profile = {}
        self.val_ocr_text = getattr(dataset, "val_ocr_text", {})
        self.train_ocr_text = getattr(dataset, "train_ocr_text", {})
        self.last_dyfo_visual_evidence = ""
        self.last_dyfo_focus_image_path = None
        self.train_keys = getattr(dataset, "train_keys", [])
        
        # Core model and optional enhancement models are loaded lazily where
        # possible to keep multi-shard startup memory under control.
        self.initialize_qwen(self.args.engine)

        # 图像处理部分按需初始化。Direct/非视觉增强实验不需要加载
        # GroundingDINO + SAM，否则多 shard 同时启动时容易产生很高的显存峰值。
        self.sam = None
        self.owlv2_model = None
        self.owlv2_processor = None

        self.caption_qwen = self.load_caption_qwen()

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
        if (
            args.use_image_enhance
            and (
                getattr(args, "mcts_action_mode", "all") != "dyfo_evidence"
                or getattr(args, "dyfo_decision_mode", "") == "clip_statement_override"
            )
        ):
            self.clip_full_model = CLIPModel.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            self.clip_full_model = self.clip_full_model.cuda()
            self.clip_full_processor = CLIPProcessor.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")

        self.temp_question = "What is the person doing?"

    def _image_key_from_sample_key(self, key):
        return image_key_from_sample_key(key, self.args, getattr(self, "image_dict", None))

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
        return self._safe_rule_postprocess_answer(answer)

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
        if uses_yes_no_prompt(self.args):
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
                if "pure_score" in rec and "dyfo_score" in rec:
                    profile[key] = {
                        "key": key,
                        "image_id": rec.get("image_id"),
                        "question": rec.get("question", ""),
                        "question_type": rec.get("question_type", ""),
                        "scores": {
                            "pure": float(rec.get("pure_score", 0.0)),
                            "dyfo": float(rec.get("dyfo_score", 0.0)),
                        },
                        "answers": {
                            "pure": rec.get("pure_answer", ""),
                            "dyfo": rec.get("dyfo_answer", ""),
                        },
                        "router_label": rec.get("router_label", ""),
                        "recommended_route": rec.get("recommended_route", ""),
                    }
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

    def _strategy_profile_question_tokens(self, text):
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
            "there", "any", "of", "to", "in", "on", "at", "for", "with", "and",
            "or", "that", "this", "these", "those", "what", "which", "who", "where",
            "how", "many", "much", "kind", "type", "color", "colour", "image", "photo",
            "picture", "left", "right", "top", "bottom", "front", "behind", "near",
        }
        return {tok for tok in re.findall(r"[a-z0-9]+", str(text).lower()) if tok not in stop and len(tok) > 1}

    def _strategy_profile_fallback_neighbors(self, question, question_type, limit):
        query_tokens = self._strategy_profile_question_tokens(question)
        if not query_tokens:
            return []
        scored = []
        for ctx_key, rec in self.strategy_profile.items():
            rec_question = rec.get("question", "")
            rec_tokens = rec.get("_question_tokens")
            if rec_tokens is None:
                rec_tokens = self._strategy_profile_question_tokens(rec_question)
                rec["_question_tokens"] = rec_tokens
            if not rec_tokens:
                continue
            overlap = len(query_tokens & rec_tokens)
            if overlap == 0:
                continue
            union = len(query_tokens | rec_tokens) or 1
            score = overlap / union
            if rec.get("question_type", "") == question_type:
                score += 0.05
            scored.append((score, ctx_key))
        scored.sort(reverse=True)
        return [ctx_key for _, ctx_key in scored[:limit]]

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
        profile_context_keys = [ctx_key for ctx_key in context_keys if ctx_key in self.strategy_profile]
        if len(profile_context_keys) < self.args.strategy_min_neighbors:
            fallback_keys = self._strategy_profile_fallback_neighbors(
                question, question_type, topk * context_multiplier
            )
            seen = set(profile_context_keys)
            profile_context_keys.extend([ctx_key for ctx_key in fallback_keys if ctx_key not in seen])
            context_keys = profile_context_keys

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
        if getattr(self.args, "multi_strategy_router_source", "profile") == "mllm":
            return self._route_with_multi_strategy_mllm(question)
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

    def _route_with_multi_strategy_mllm(self, question):
        strategies = [s.strip() for s in self.args.multi_strategy_names.split(",") if s.strip()]
        default_strategy = self.args.multi_strategy_default
        if default_strategy not in strategies:
            strategies.insert(0, default_strategy)
        prompt = (
            "You are routing a visual question answering sample.\n"
            "Choose exactly one strategy from: %s.\n"
            "direct means answer with the original multimodal model.\n"
            "dyfo means first search for focused visual evidence when small objects, attributes, text, counting, or spatial relations matter.\n"
            "Prefer direct when the question is global, commonsense-heavy, or the visual focus is unlikely to help.\n"
            "Output exactly: Strategy: <name>\n"
            "Question: %s"
        ) % (", ".join(strategies), question)
        try:
            response = self._call_llm(
                prompt,
                image_path=None,
                max_new_tokens=getattr(self.args, "dyfo_focus_max_tokens", 32),
                use_images=False,
            )
        except Exception as exc:
            response = "router failed: %s" % exc
        text = str(response).strip().lower()
        selected = default_strategy
        for strategy in strategies:
            if re.search(r"\b%s\b" % re.escape(strategy.lower()), text):
                selected = strategy
                break
        if selected not in strategies:
            selected = default_strategy
        return {
            "strategy": selected,
            "reason": "mllm_router",
            "router_prompt": prompt,
            "router_response": response,
            "neighbors": [],
            "strategy_avgs": {},
            "best_avg": 0.0,
            "default_avg": 0.0,
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

    def _knowledge_direct_is_weak(self, answer):
        cleaned = self._clean_short_answer(answer)
        norm = self._normalize_candidate_answer(cleaned)
        weak_answers = {
            "", "unknown", "unsure", "not sure", "cannot tell", "cant tell", "i dont know",
            "object", "thing", "person", "people", "animal", "food", "vehicle", "item",
        }
        return norm in weak_answers or self._looks_like_visual_cue_list(cleaned)

    def _should_run_notemr_candidate(self, question, question_type, direct_answer):
        mode = getattr(self.args, "notemr_candidate_trigger", "knowledge_qtype_or_weak")
        if mode == "always":
            return True, "always"
        if mode == "never":
            return False, "never"
        is_knowledge_like = question_type in ("knowledge", "category")
        direct_weak = self._knowledge_direct_is_weak(direct_answer)
        if mode == "knowledge_qtype":
            return is_knowledge_like, "knowledge_qtype" if is_knowledge_like else "not_knowledge_qtype"
        if mode == "weak_direct":
            return direct_weak, "weak_direct" if direct_weak else "direct_not_weak"
        if is_knowledge_like or direct_weak:
            reason = "knowledge_qtype" if is_knowledge_like else "weak_direct"
            return True, reason
        return False, "direct_safe_nonknowledge"

    def _format_notemr_note_relevance_prompt(self, question, choice_text, direct_answer, knowledge_notes):
        return (
            "You are checking whether Knowledge Notes should be allowed to influence a VQA answer.\n"
            "Be strict. The original image is the primary evidence. Knowledge Notes are only helpful if they are "
            "clearly relevant to both the image-question pair and the expected answer.\n"
            "Reject notes that are generic, off-topic, merely caption-like, or introduce unsupported entities.\n"
            "Question: %s%s\n"
            "Current direct answer: %s\n"
            "Knowledge Notes:\n%s\n"
            "Output exactly:\n"
            "Relevant: yes / no\n"
            "Reason: <short reason>\n"
        ) % (
            question,
            choice_text,
            self._clean_short_answer(direct_answer),
            self._truncate_text(knowledge_notes, self.args.knowledge_notes_max_chars),
        )

    def _extract_notemr_relevance(self, response, knowledge_notes):
        text = str(response or "").strip()
        first_lines = "\n".join(text.splitlines()[:3]).lower()
        notes_l = str(knowledge_notes or "").lower()
        negative_note_markers = (
            "no useful external knowledge", "not relevant", "irrelevant", "unrelated",
            "no relevant", "no extra knowledge"
        )
        if any(marker in notes_l for marker in negative_note_markers):
            return False
        if re.search(r"relevant\s*:\s*yes", first_lines, flags=re.IGNORECASE):
            return True
        return False

    def _format_notemr_candidate_prompt(self, question, choice_text, direct_answer, knowledge_notes):
        return (
            "Answer the visual question with a single word or short phrase.\n"
            "Use the original image as the main evidence. Use the Knowledge Notes only as background when they "
            "directly clarify the visible scene or object. Do not answer from knowledge alone.\n"
            "If the Knowledge Notes are not useful, repeat the direct answer.\n"
            "Direct answer: %s\n"
            "Knowledge Notes:\n%s\n"
            "Question: %s%s\n"
            "Answer:"
        ) % (
            self._clean_short_answer(direct_answer),
            self._truncate_text(knowledge_notes, self.args.knowledge_notes_max_chars),
            question,
            choice_text,
        )

    def _format_notemr_conservative_judge_prompt(self, question, choice_text, direct_answer,
                                                 knowledge_answer, knowledge_notes, relevance_response):
        return (
            "You are a conservative VQA answer arbiter.\n"
            "Default decision: keep the direct answer.\n"
            "Use the knowledge candidate only when all conditions are true:\n"
            "1. The Knowledge Notes are clearly relevant to the image and question.\n"
            "2. The direct answer is weak, generic, or misses the asked concept.\n"
            "3. The knowledge candidate is a short answer and is better supported by the image-question pair.\n"
            "4. The knowledge candidate does not introduce an unsupported specific entity.\n"
            "If uncertain, keep direct.\n"
            "Question: %s%s\n"
            "Direct answer: %s\n"
            "Knowledge candidate answer: %s\n"
            "Knowledge Notes:\n%s\n"
            "Relevance check:\n%s\n"
            "Output exactly:\n"
            "Decision: keep_direct / use_knowledge_candidate\n"
            "Reason: <short reason>\n"
            "Final Answer:"
        ) % (
            question,
            choice_text,
            self._clean_short_answer(direct_answer),
            self._clean_short_answer(knowledge_answer),
            self._truncate_text(knowledge_notes, self.args.knowledge_notes_max_chars),
            self._truncate_text(relevance_response, 500),
        )

    def _extract_notemr_judge_answer(self, response, direct_answer, knowledge_answer):
        text = str(response or "").strip()
        first_lines = "\n".join(text.splitlines()[:4]).lower()
        direct_answer = self._clean_short_answer(direct_answer)
        knowledge_answer = self._clean_short_answer(knowledge_answer)
        if "decision: use_knowledge_candidate" not in first_lines:
            return direct_answer
        final_answer = self._extract_structured_cot_answer(text)
        final_norm = self._normalize_candidate_answer(final_answer)
        knowledge_norm = self._normalize_candidate_answer(knowledge_answer)
        if knowledge_norm and final_norm == knowledge_norm and not self._looks_like_visual_cue_list(knowledge_answer):
            return knowledge_answer
        return direct_answer

    def _run_notemr_conservative_candidate(self, question, choice_text, direct_answer, question_type,
                                           image_path, data_row, object_list, attr_list):
        direct_answer = self._clean_short_answer(direct_answer)
        should_run, trigger_reason = self._should_run_notemr_candidate(question, question_type, direct_answer)
        trace = {
            "triggered": should_run,
            "trigger_reason": trigger_reason,
            "direct_answer": direct_answer,
            "knowledge_notes": "",
            "relevance_prompt": "",
            "relevance_response": "",
            "knowledge_candidate_prompt": "",
            "knowledge_candidate_response": "",
            "knowledge_candidate_answer": "",
            "judge_prompt": "",
            "judge_response": "",
            "final_answer": direct_answer,
        }
        if not should_run:
            return trace

        notes = self.enhance_knowledge_object(data_row, object_list, attr_list)
        notes = self._truncate_text(notes, self.args.knowledge_notes_max_chars)
        trace["knowledge_notes"] = notes
        if not notes:
            return trace

        relevance_prompt = self._format_notemr_note_relevance_prompt(
            question, choice_text, direct_answer, notes
        )
        relevance_response = self._call_llm(
            relevance_prompt, image_path=image_path, max_new_tokens=self.args.notemr_relevance_max_tokens
        )
        trace["relevance_prompt"] = relevance_prompt
        trace["relevance_response"] = relevance_response
        relevant = self._extract_notemr_relevance(relevance_response, notes)
        if not relevant:
            trace["trigger_reason"] += "|notes_rejected"
            return trace

        candidate_prompt = self._format_notemr_candidate_prompt(
            question, choice_text, direct_answer, notes
        )
        candidate_response = self._call_llm(
            candidate_prompt, image_path=image_path, max_new_tokens=self.args.notemr_candidate_max_tokens
        )
        knowledge_answer = self._clean_short_answer(self._extract_answer_from_response(candidate_response))
        trace["knowledge_candidate_prompt"] = candidate_prompt
        trace["knowledge_candidate_response"] = candidate_response
        trace["knowledge_candidate_answer"] = knowledge_answer
        if not knowledge_answer or self._looks_like_visual_cue_list(knowledge_answer):
            return trace
        if self._normalize_candidate_answer(knowledge_answer) == self._normalize_candidate_answer(direct_answer):
            trace["final_answer"] = direct_answer
            return trace

        judge_prompt = self._format_notemr_conservative_judge_prompt(
            question, choice_text, direct_answer, knowledge_answer, notes, relevance_response
        )
        judge_response = self._call_llm(
            judge_prompt, image_path=image_path, max_new_tokens=self.args.notemr_judge_max_tokens
        )
        final_answer = self._extract_notemr_judge_answer(judge_response, direct_answer, knowledge_answer)
        trace["judge_prompt"] = judge_prompt
        trace["judge_response"] = judge_response
        trace["final_answer"] = final_answer
        return trace

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
                                   "direct_rephrase_consistency", "notemr_conservative_candidate"):
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
    
    # ------------------------------------------------------------------
    # Inference orchestration
    # ------------------------------------------------------------------
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
                image_key = self._image_key_from_sample_key(key)
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
        image_key = self._image_key_from_sample_key(key)
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
        if can_skip_scene_graph(self.args):
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

    def init_attention_object(self, key, attr_list, image_path, ban_option=None):
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
                         onion_instruction=None, round_idx=None, state_history=None):

        if onion_instruction is None:
            onion_instruction = [None]

        # onion_instruction[0] 已经给出的下一步的指令
        # onion_instruction[1] 已经给出的下一步指令的对象

        # 补充：这段代码是 Chain-of-Thought (CoT) 推理步骤的后处理与验证模块，主要功能是用 CLIP 模型筛选高质量的推理步骤。

        # 获取图片id
        image_key = self._image_key_from_sample_key(key)
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
        rag_strategy_route = None
        rag_selected_strategy = None
        if self.args.cot_style == "multi_strategy_router":
            multi_strategy_route = self._route_with_multi_strategy_profile(key, question)
        if self.args.cot_style == "rag_strategy_router":
            rag_strategy_route = self._route_with_strategy_profile(key, question)
            rag_selected_strategy = rag_strategy_route.get("strategy")
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
            effective_use_image_enhance = self.args.use_image_enhance and selected_strategy in ("marker_mcts", "dyfo")
            effective_use_caption_enhance = False
            effective_use_knowledge_enhance = False
        if self.args.cot_style == "rag_strategy_router" and self.args.strategy_cot_runtime == "dyfo_evidence":
            effective_use_image_enhance = (
                self.args.use_image_enhance
                and rag_selected_strategy == self.args.strategy_cot_name
            )
            effective_use_caption_enhance = False
            effective_use_knowledge_enhance = False
        selective_mode = self.args.cot_style == "reviewer_evidence" and self.args.reviewer_evidence_scope == "selective"
        knowledge_triggered = onion_instruction[0] == 'knowledge' or selective_mode
        if self.args.knowledge_enhance_trigger == "always":
            knowledge_triggered = True
        elif self.args.knowledge_enhance_trigger == "knowledge_qtype":
            knowledge_triggered = question_type in ("knowledge", "category")
        
        # ========== 三个核心增强模块（由args控制开关） ==========
        dyfo_force_run = (
            self.args.mcts_action_mode == "dyfo_evidence"
            and getattr(self.args, "dyfo_force_run_all_samples", False)
        )
        if effective_use_image_enhance and (onion_instruction[0] == 'image' or dyfo_force_run):
            if self.args.mcts_action_mode == "dyfo_evidence":
                if dyfo_force_run and onion_instruction[0] != 'image':
                    print("[dyfo] force-run enabled; overriding ONION instruction '%s'" % onion_instruction[0])
                dyfo_result = self._run_dyfo_visual_evidence_search(data_row, onion_instruction[1], attr_list)
                dyfo_visual_evidence = dyfo_result.get("evidence", "")
                dyfo_focus_image_path = dyfo_result.get("focus_image_path")
                dyfo_answer_image_path = dyfo_result.get("answer_image_path")
                dyfo_final_answer = dyfo_result.get("final_answer", "")
                dyfo_decision_trace = dyfo_result.get("decision_trace")
                self.last_dyfo_visual_evidence = dyfo_visual_evidence
                self.last_dyfo_focus_image_path = dyfo_focus_image_path
                if self.args.dyfo_use_focus_image_as_answer:
                    enhance_image_path = dyfo_answer_image_path or dyfo_focus_image_path
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
                image_context_key = image_key_from_sample_key(
                    context_key, self.args, getattr(self, "image_dict", None)
                )

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
            if (
                enhance_image_path
                and self.args.mcts_action_mode == "dyfo_evidence"
                and self.args.dyfo_use_focus_image_as_answer
                and self.args.dyfo_answer_image_mode.startswith("concat")
            ):
                cur_caption += (
                    "\nDyFo image layout: the provided image concatenates the original image with "
                    "a DyFo focused crop resized to the original size. Use the original view for "
                    "global context and the focused view for local detail."
                )
            if self.args.direct_prompt_style == "context_gated":
                cur_caption = self._build_direct_context_for_style(
                    question, caption, regional_context, ocr_context
                )
                if dyfo_evidence_enabled and dyfo_visual_evidence:
                    cur_caption += '\nDyFo visual evidence: ' + self._truncate_text(
                        dyfo_visual_evidence, self.args.dyfo_evidence_context_max_chars
                    )
                if (
                    enhance_image_path
                    and self.args.mcts_action_mode == "dyfo_evidence"
                    and self.args.dyfo_use_focus_image_as_answer
                    and self.args.dyfo_answer_image_mode.startswith("concat")
                ):
                    cur_caption += (
                        "\nDyFo image layout: the provided image concatenates the original image "
                        "with a DyFo focused crop resized to the original size."
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
                        if selected_strategy == "dyfo" and enhance_image_path:
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
                        elif selected_strategy == "dyfo":
                            extracted_answer = self._clean_short_answer(dyfo_final_answer or initial_answer)
                            runtime_trace = (
                                "DyFo Runtime\nDyFo Evidence:\n%s\nDyFo Decision Trace:\n%s\nFinal Answer: %s"
                            ) % (
                                dyfo_visual_evidence,
                                json.dumps(dyfo_decision_trace, ensure_ascii=False),
                                extracted_answer,
                            )
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
                        route = rag_strategy_route or self._route_with_strategy_profile(key, question)
                        selected_strategy = route["strategy"]
                        rag_selected_strategy = selected_strategy
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
                            elif self.args.strategy_cot_runtime == "dyfo_evidence":
                                extracted_answer = self._clean_short_answer(dyfo_final_answer or initial_answer)
                                response = (
                                    "RAG Strategy Router: %s\n"
                                    "Router Mode: %s\n"
                                    "Route Stats: direct_avg=%.3f cot_avg=%.3f rescue_rate=%.3f damage_rate=%.3f reason=%s\n"
                                    "Initial Direct Answer: %s\n"
                                    "DyFo Visual Evidence:\n%s\n"
                                    "DyFo Decision Trace:\n%s\n"
                                    "Final Answer: %s"
                                ) % (
                                    selected_strategy, self.args.strategy_router_mode,
                                    route.get("direct_avg", 0.0), route.get("cot_avg", 0.0),
                                    route.get("rescue_rate", 0.0), route.get("damage_rate", 0.0),
                                    route.get("reason", ""), initial_answer, dyfo_visual_evidence,
                                    json.dumps(dyfo_decision_trace, ensure_ascii=False), extracted_answer,
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
                    elif self.args.cot_style == "notemr_conservative_candidate":
                        selected_objects = onion_instruction[1] if len(onion_instruction) > 1 else []
                        initial_answer = self._clean_short_answer(self._extract_answer_from_response(response))
                        notemr_trace = self._run_notemr_conservative_candidate(
                            question=question,
                            choice_text=choice_text,
                            direct_answer=initial_answer,
                            question_type=question_type,
                            image_path=image_path,
                            data_row=data_row,
                            object_list=selected_objects,
                            attr_list=attr_list,
                        )
                        extracted_answer = notemr_trace["final_answer"]
                        response = (
                            "NoteMR Conservative Candidate\n"
                            "Question Type: %s\n"
                            "Triggered: %s\n"
                            "Trigger Reason: %s\n"
                            "Initial Direct Answer: %s\n"
                            "Knowledge Notes:\n%s\n"
                            "Relevance Prompt:\n%s\n"
                            "Relevance Response:\n%s\n"
                            "Knowledge Candidate Prompt:\n%s\n"
                            "Knowledge Candidate Response:\n%s\n"
                            "Knowledge Candidate Answer: %s\n"
                            "Judge Prompt:\n%s\n"
                            "Judge Response:\n%s\n"
                            "Final Answer: %s"
                        ) % (
                            question_type,
                            notemr_trace.get("triggered"),
                            notemr_trace.get("trigger_reason", ""),
                            initial_answer,
                            notemr_trace.get("knowledge_notes", ""),
                            notemr_trace.get("relevance_prompt", ""),
                            notemr_trace.get("relevance_response", ""),
                            notemr_trace.get("knowledge_candidate_prompt", ""),
                            notemr_trace.get("knowledge_candidate_response", ""),
                            notemr_trace.get("knowledge_candidate_answer", ""),
                            notemr_trace.get("judge_prompt", ""),
                            notemr_trace.get("judge_response", ""),
                            extracted_answer,
                        )
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
            and self.args.dyfo_decision_mode in ("best_focus_answer", "weighted_vote", "conservative_override", "token_confidence_override", "node_confidence_override", "clip_statement_override")
            and dyfo_final_answer
            and (
                self.args.cot_style != "rag_strategy_router"
                or rag_selected_strategy == self.args.strategy_cot_name
            )
            and (
                self.args.cot_style != "multi_strategy_router"
                or (multi_strategy_route or {}).get("strategy") == "dyfo"
            )
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
            if uses_yes_no_scoring(self.args):
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
    
    # ------------------------------------------------------------------
    # Routing and visual evidence enhancement
    # ------------------------------------------------------------------
    def onion_make_instruction(self, key, object_list):
        onion_instruction = []
        image_key = self._image_key_from_sample_key(key)
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
        if is_dataset(self.args, "mme"):
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

    def _parse_dyfo_key_objects(self, response):
        text = str(response).strip()
        if not text:
            return []

        candidates = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("key_objects", "objects", "targets"):
                    if isinstance(parsed.get(key), list):
                        candidates = parsed[key]
                        break
            elif isinstance(parsed, list):
                candidates = parsed
        except Exception:
            candidates = []

        if not candidates:
            match = re.search(r"\[[^\]]+\]", text)
            if match:
                try:
                    parsed = json.loads(match.group(0).replace("'", '"'))
                    if isinstance(parsed, list):
                        candidates = parsed
                except Exception:
                    candidates = []

        if not candidates:
            text = re.sub(r"(?is)^.*?(?:key objects?|objects?|targets?)\s*(?:are|:)\s*", "", text)
            parts = re.split(r"[,;\n]+", text)
            candidates = [part.strip() for part in parts]

        key_objects = []
        generic = {
            "image", "picture", "photo", "scene", "object", "objects", "thing",
            "things", "area", "region", "evidence", "unknown", "none", "n/a", "na"
        }
        for item in candidates:
            phrase = str(item).strip()
            phrase = re.sub(r"^\s*(?:[-*]|\d+[\).:])\s*", "", phrase).strip()
            phrase = phrase.strip(" \t\"'`.,;:!?")
            phrase = re.sub(r"\s+", " ", phrase)
            if not phrase:
                continue
            if phrase.lower() in generic:
                continue
            if len(phrase.split()) > 8:
                continue
            if phrase.lower() not in [obj.lower() for obj in key_objects]:
                key_objects.append(phrase)
            if len(key_objects) >= 5:
                break
        return key_objects

    def extract_key_objects_from_question(self, question):
        prompt = (
            "Extract the key visible object phrases needed to answer this visual question.\n"
            "Use the question text freely; do not restrict yourself to detector labels.\n"
            "Include relational phrases when helpful, such as 'person with white trousers' or 'person in blue'.\n"
            "Do not answer the question. Return a JSON list of 1 to 5 short phrases only.\n"
            "Question: %s"
        ) % question
        try:
            response = self._call_llm(
                prompt,
                image_path=None,
                max_new_tokens=getattr(self.args, "dyfo_focus_max_tokens", 32),
                use_images=False,
            )
        except Exception as exc:
            print(f"[dyfo] free key-object extraction failed: {exc}")
            return [], ""
        key_objects = self._parse_dyfo_key_objects(response)
        print("[dyfo] free-extracted key_objects:", key_objects)
        print("[dyfo] key_object_extraction_response:", response)
        return key_objects, response

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

    def _dyfo_initial_focus(self, question, obj_list=None, key_objects=None):
        obj_list = obj_list or []
        key_objects = key_objects or []
        fallback_pool = key_objects or obj_list
        fallback = fallback_pool[0] if fallback_pool else self._dyfo_question_focus_fallback(question)
        object_hint = ", ".join(key_objects[:8]) if key_objects else "No free key objects were extracted; infer the cue from the question."
        candidate_hint = ", ".join(obj_list[:8]) if obj_list else "No fallback detector candidates are available."
        prompt = (
            "Choose the most useful visual focus cue for answering the question.\n"
            "The focus should be a visible object, attribute, text area, relation, or small region that a visual expert can localize.\n"
            "Do not answer the question. Output exactly: Focus: <short visual cue>\n"
            "Question: %s\n"
            "Key objects: %s\n"
            "Fallback detector candidates: %s"
        ) % (question, object_hint, candidate_hint)
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

    def _dyfo_langsam_detect_boxes(self, image_pil, target):
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        try:
            self.ensure_lang_sam()
            with torch.no_grad():
                results = self.sam.predict([image_pil], [target])
        except Exception as exc:
            print("[dyfo dual] GroundingDINO failed target=%s error=%s" % (target, exc))
            return []
        if not results:
            return []
        result = results[0]
        raw_boxes = result.get("boxes", [])
        raw_scores = result.get("scores", [])
        detections = []
        for idx, raw_box in enumerate(raw_boxes):
            box = tuple(int(round(float(value))) for value in raw_box)
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            score = float(raw_scores[idx]) if idx < len(raw_scores) else 0.0
            detections.append({"box": box, "score": score, "expert": "grounding_dino"})
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[:max(1, getattr(self.args, "dyfo_dual_max_boxes_per_target", 3))]

    def initialize_owlv2(self):
        model_path = getattr(self.args, "dyfo_owlv2_model_path", "")
        if not model_path:
            raise ValueError("--dyfo_owlv2_model_path is required for dual visual experts")
        self.owlv2_processor = Owlv2Processor.from_pretrained(model_path, local_files_only=True)
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).cuda().eval()
        print("[dyfo dual] loaded OWLv2 from %s" % model_path)

    def ensure_owlv2(self):
        if self.owlv2_model is None or self.owlv2_processor is None:
            self.initialize_owlv2()

    def _dyfo_owlv2_detect_targets(self, image_pil, targets):
        targets = self._dyfo_unique_targets(targets)
        detections = {target: [] for target in targets}
        if not targets:
            return detections
        try:
            self.ensure_owlv2()
            inputs = self.owlv2_processor(text=[targets], images=[image_pil], return_tensors="pt")
            model_device = next(self.owlv2_model.parameters()).device
            model_dtype = next(self.owlv2_model.parameters()).dtype
            inputs = {
                key: value.to(model_device, dtype=model_dtype) if value.is_floating_point() else value.to(model_device)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self.owlv2_model(**inputs)
            target_sizes = torch.tensor([image_pil.size[::-1]], device=model_device)
            processed = self.owlv2_processor.post_process_grounded_object_detection(
                outputs,
                threshold=float(getattr(self.args, "dyfo_owlv2_threshold", 0.10)),
                target_sizes=target_sizes,
                text_labels=[targets],
            )[0]
            boxes = processed.get("boxes", []).detach().float().cpu().tolist()
            scores = processed.get("scores", []).detach().float().cpu().tolist()
            labels = processed.get("text_labels", [])
            if not labels:
                label_ids = processed.get("labels", []).detach().cpu().tolist()
                labels = [targets[int(idx)] for idx in label_ids]
            for raw_box, score, label in zip(boxes, scores, labels):
                target = str(label)
                if target not in detections:
                    continue
                box = tuple(int(round(float(value))) for value in raw_box)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections[target].append({"box": box, "score": float(score), "expert": "owlv2"})
        except Exception as exc:
            print("[dyfo dual] OWLv2 failed targets=%s error=%s" % (targets, exc))
            return detections
        max_boxes = max(1, getattr(self.args, "dyfo_dual_max_boxes_per_target", 3))
        for target in targets:
            detections[target].sort(key=lambda item: item["score"], reverse=True)
            detections[target] = detections[target][:max_boxes]
        return detections

    def _dyfo_box_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return float(intersection) / float(union) if union > 0 else 0.0

    def _dyfo_match_expert_boxes(self, boxes_a, boxes_b, iou_threshold):
        candidates = []
        for idx_a, detection_a in enumerate(boxes_a):
            for idx_b, detection_b in enumerate(boxes_b):
                iou = self._dyfo_box_iou(detection_a["box"], detection_b["box"])
                if iou >= iou_threshold:
                    candidates.append((iou, idx_a, idx_b))
        candidates.sort(reverse=True)
        used_a, used_b, matches = set(), set(), []
        for iou, idx_a, idx_b in candidates:
            if idx_a in used_a or idx_b in used_b:
                continue
            used_a.add(idx_a)
            used_b.add(idx_b)
            matches.append({"grounding_dino": boxes_a[idx_a], "owlv2": boxes_b[idx_b], "iou": iou})
        unmatched_a = [item for idx, item in enumerate(boxes_a) if idx not in used_a]
        unmatched_b = [item for idx, item in enumerate(boxes_b) if idx not in used_b]
        return matches, unmatched_a, unmatched_b

    def _dyfo_dual_expert_locate_required_targets(self, image_pil, required_targets):
        required_targets = self._dyfo_unique_targets(required_targets)
        owlv2_by_target = self._dyfo_owlv2_detect_targets(image_pil, required_targets)
        gdino_by_target = {
            target: self._dyfo_langsam_detect_boxes(image_pil, target)
            for target in required_targets
        }
        base_iou = float(getattr(self.args, "dyfo_dual_iou_threshold", 0.60))
        delta = float(getattr(self.args, "dyfo_dual_iou_delta", 0.10))
        initial_confirmed = 0
        initial_suspicious = 0
        for target in required_targets:
            matches, unmatched_a, unmatched_b = self._dyfo_match_expert_boxes(
                gdino_by_target[target], owlv2_by_target[target], base_iou
            )
            initial_confirmed += len(matches)
            initial_suspicious += len(unmatched_a) + len(unmatched_b)
        denominator = initial_confirmed + initial_suspicious
        conflict_rate = float(initial_suspicious) / denominator if denominator else 1.0
        low = float(getattr(self.args, "dyfo_dual_conflict_low", 0.50))
        high = float(getattr(self.args, "dyfo_dual_conflict_high", 0.70))
        if conflict_rate < low:
            effective_iou = max(0.0, base_iou - delta)
            look_mode = "glance"
        elif conflict_rate > high:
            effective_iou = min(1.0, base_iou + delta)
            look_mode = "stare"
        else:
            effective_iou = base_iou
            look_mode = "balanced"

        confirmed, suspicious = [], []
        target_boxes, support_boxes, missing_targets = {}, [], []
        for target in required_targets:
            boxes_a = gdino_by_target[target]
            boxes_b = owlv2_by_target[target]
            matches, unmatched_a, unmatched_b = self._dyfo_match_expert_boxes(
                boxes_a, boxes_b, effective_iou
            )
            target_support = []
            for match in matches:
                union_box = self._dyfo_union_boxes(
                    [match["grounding_dino"]["box"], match["owlv2"]["box"]],
                    image_pil.size,
                    1.0,
                )
                item = {"target": target, "box": union_box, **match}
                confirmed.append(item)
                target_support.append(union_box)
            for detection in unmatched_a + unmatched_b:
                item = {"target": target, **detection}
                suspicious.append(item)
                target_support.append(detection["box"])
            if target_support:
                target_boxes[target] = self._dyfo_union_boxes(target_support, image_pil.size, 1.0)
                support_boxes.extend(target_support)
            else:
                missing_targets.append(target)

        union_box = self._dyfo_union_boxes(
            support_boxes,
            image_pil.size,
            getattr(self.args, "dyfo_focus_padding", 1.2),
        )
        agreement_denominator = len(confirmed) + len(suspicious)
        agreement_score = (
            float(len(confirmed)) / agreement_denominator if agreement_denominator else 0.0
        )
        result = {
            "required_targets": required_targets,
            "target_boxes": target_boxes,
            "missing_targets": missing_targets,
            "all_targets_located": bool(required_targets) and not missing_targets,
            "support_boxes": support_boxes,
            "joined_box": None,
            "union_box": union_box,
            "query_log": [],
            "grounding_dino": gdino_by_target,
            "owlv2": owlv2_by_target,
            "confirmed_regions": confirmed,
            "suspicious_regions": suspicious,
            "conflict_rate": conflict_rate,
            "agreement_score": agreement_score,
            "base_iou_threshold": base_iou,
            "effective_iou_threshold": effective_iou,
            "look_mode": look_mode,
        }
        print(
            "[dyfo dual] mode=%s conflict=%.3f iou=%.3f confirmed=%d suspicious=%d missing=%s"
            % (look_mode, conflict_rate, effective_iou, len(confirmed), len(suspicious), missing_targets)
        )
        return result

    def _dyfo_build_active_look_highlight(self, original, dual_result):
        highlighted = original.copy().convert("RGB")
        draw = ImageDraw.Draw(highlighted)
        line_width = max(2, int(round(min(original.size) / 180.0)))
        for region in dual_result.get("confirmed_regions", []):
            box = tuple(region["box"])
            draw.rectangle(box, outline=(32, 190, 80), width=line_width)
            draw.text((box[0] + 2, box[1] + 2), str(region["target"]), fill=(32, 190, 80))
        for region in dual_result.get("suspicious_regions", []):
            box = tuple(region["box"])
            draw.rectangle(box, outline=(225, 45, 45), width=line_width)
            draw.text((box[0] + 2, box[1] + 2), str(region["target"]), fill=(225, 45, 45))
        return highlighted

    def _dyfo_locate_with_fallbacks(self, image_pil, focus_text, key_objects=None, selected_objects=None):
        key_objects = key_objects or []
        selected_objects = selected_objects or []
        queries = []

        if focus_text:
            queries.append(("focus_cue", focus_text))
        if key_objects:
            queries.append(("key_objects", ", ".join(key_objects[:5])))
        if selected_objects:
            queries.append(("selected_objects", ", ".join(selected_objects[:5])))

        seen = set()
        for source, query in queries:
            query = str(query).strip()
            if not query or query.lower() in seen:
                continue
            seen.add(query.lower())
            box = self._dyfo_locate_focus(image_pil, query)
            if box is not None:
                fallback_triggered = source != "focus_cue"
                print(
                    "[dyfo] LangSAM query source=%s fallback=%s query=%s box=%s"
                    % (source, fallback_triggered, query, box)
                )
                return box, query, source, fallback_triggered
            print("[dyfo] LangSAM query failed source=%s query=%s" % (source, query))

        print("[dyfo] LangSAM all queries failed for focus=%s" % focus_text)
        return None, "", "none", True

    def _dyfo_unique_targets(self, targets):
        unique = []
        seen = set()
        for target in targets or []:
            text = re.sub(r"\s+", " ", str(target).strip())
            text = text.strip(" \t\"'`.,;:!?")
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
            if len(unique) >= 5:
                break
        return unique

    def _dyfo_union_boxes(self, boxes, image_size, padding_scale=1.0):
        valid_boxes = []
        w, h = image_size
        for box in boxes or []:
            if not box:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(w, int(x1)))
            y1 = max(0, min(h, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 > x1 and y2 > y1:
                valid_boxes.append((x1, y1, x2, y2))
        if not valid_boxes:
            return None
        arr = np.array(valid_boxes)
        union_box = (
            int(np.min(arr[:, 0])),
            int(np.min(arr[:, 1])),
            int(np.max(arr[:, 2])),
            int(np.max(arr[:, 3])),
        )
        return self._dyfo_expand_box(union_box, image_size, padding_scale)

    def _dyfo_is_relation_question(self, question):
        q = str(question).lower()
        relation_terms = [
            "left", "right", "behind", "in front of", "next to", "beside",
            "between", "near", "on top of", "under", "below", "above",
            "around", "across", "closest", "farther", "same", "different",
            "than", "facing", "holding", "wearing", "with"
        ]
        return any(term in q for term in relation_terms)

    def _dyfo_abs_box_from_local(self, local_box, parent_box):
        if not local_box:
            return None
        px1, py1, _, _ = parent_box
        lx1, ly1, lx2, ly2 = local_box
        return (px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2)

    def _dyfo_locate_required_targets(self, image_pil, required_targets, allow_joined_query=True):
        required_targets = self._dyfo_unique_targets(required_targets)
        target_boxes = {}
        missing_targets = []
        query_log = []
        support_boxes = []

        for target in required_targets:
            box = self._dyfo_locate_focus(image_pil, target)
            query_log.append({
                "target": target,
                "query": target,
                "source": "required_target",
                "box": box,
            })
            if box is None:
                missing_targets.append(target)
            else:
                target_boxes[target] = box
                support_boxes.append(box)

        joined_box = None
        if allow_joined_query and missing_targets and len(required_targets) > 1:
            joined_query = ", ".join(required_targets)
            joined_box = self._dyfo_locate_focus(image_pil, joined_query)
            query_log.append({
                "target": "__joined_required_targets__",
                "query": joined_query,
                "source": "joined_required_targets",
                "box": joined_box,
            })
            if joined_box is not None:
                support_boxes.append(joined_box)

        union_box = self._dyfo_union_boxes(
            support_boxes, image_pil.size, getattr(self.args, "dyfo_focus_padding", 1.2)
        )
        result = {
            "required_targets": required_targets,
            "target_boxes": target_boxes,
            "missing_targets": missing_targets,
            "all_targets_located": bool(required_targets) and not missing_targets,
            "support_boxes": support_boxes,
            "joined_box": joined_box,
            "union_box": union_box,
            "query_log": query_log,
        }
        print(
            "[dyfo] required-target locate all_located=%s missing=%s union=%s"
            % (result["all_targets_located"], missing_targets, union_box)
        )
        return result

    def _dyfo_qwen_target_presence(self, crop, target, question):
        temp_path = None
        try:
            temp_path = os.path.join(
                self.args.cache_path,
                "dyfo_target_check_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
            )
            os.makedirs(self.args.cache_path, exist_ok=True)
            crop.save(temp_path)
            prompt = (
                "Is there a %s in this image crop? Answer yes or no.\n"
                "Question context: %s"
            ) % (target, question)
            reply = self._call_llm(prompt, image_path=temp_path, max_new_tokens=8)
            return str(reply).strip().lower().startswith("yes"), reply
        except Exception as exc:
            return False, "target presence check failed: %s" % exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _dyfo_spatial_context_check(self, crop, question):
        temp_path = None
        try:
            temp_path = os.path.join(
                self.args.cache_path,
                "dyfo_spatial_check_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
            )
            os.makedirs(self.args.cache_path, exist_ok=True)
            crop.save(temp_path)
            prompt = (
                "Does this image crop preserve enough spatial context to answer the spatial or relational question? "
                "Answer yes or no.\n"
                "Question: %s"
            ) % question
            reply = self._call_llm(prompt, image_path=temp_path, max_new_tokens=8)
            return str(reply).strip().lower().startswith("yes"), reply
        except Exception as exc:
            return False, "spatial context check failed: %s" % exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _dyfo_check_all_targets_present(self, crop, required_targets, question):
        required_targets = self._dyfo_unique_targets(required_targets)
        if not required_targets:
            return {
                "all_targets_present": True,
                "present_targets": [],
                "missing_targets": [],
                "target_boxes": {},
                "presence_replies": {},
                "locate_trace": [],
                "spatial_context_ok": True,
                "spatial_context_reply": "",
            }

        locate_result = self._dyfo_locate_required_targets(crop, required_targets, allow_joined_query=False)
        present_targets = set(locate_result["target_boxes"].keys())
        missing_targets = list(locate_result["missing_targets"])
        presence_replies = {}

        # LangSAM can fail on descriptive phrases; use Qwen as a conservative
        # yes/no retention fallback for targets that were not re-localized.
        for target in list(missing_targets):
            present, reply = self._dyfo_qwen_target_presence(crop, target, question)
            presence_replies[target] = reply
            if present:
                present_targets.add(target)
                missing_targets.remove(target)

        result = {
            "all_targets_present": len(missing_targets) == 0,
            "present_targets": list(present_targets),
            "missing_targets": missing_targets,
            "target_boxes": locate_result["target_boxes"],
            "presence_replies": presence_replies,
            "locate_trace": locate_result["query_log"],
            "spatial_context_ok": True,
            "spatial_context_reply": "",
        }
        if result["all_targets_present"] and self._dyfo_is_relation_question(question):
            spatial_ok, spatial_reply = self._dyfo_spatial_context_check(crop, question)
            result["spatial_context_ok"] = spatial_ok
            result["spatial_context_reply"] = spatial_reply
            if not spatial_ok:
                result["all_targets_present"] = False
                result["missing_targets"] = ["spatial_context"]
        print(
            "[dyfo] all-target retention all_present=%s present=%s missing=%s"
            % (result["all_targets_present"], result["present_targets"], result["missing_targets"])
        )
        return result

    def _dyfo_recover_missing_targets(self, original, current_box, required_targets, missing_targets, question):
        missing_targets = self._dyfo_unique_targets(missing_targets)
        if not missing_targets:
            return current_box, {
                "recovered": False,
                "reason": "no_missing_targets",
                "missing_targets": [],
                "recovery_boxes": [],
            }

        if missing_targets == ["spatial_context"]:
            recovered_box = self._dyfo_expand_box(
                current_box, original.size, getattr(self.args, "dyfo_scatter_scale", 1.6)
            ) if current_box else (0, 0, original.width, original.height)
            crop = self._dyfo_crop_for_node(original, recovered_box)
            retention = self._dyfo_check_all_targets_present(crop, required_targets, question)
            trace = {
                "recovered": retention["all_targets_present"],
                "reason": "zoom_out_for_spatial_context",
                "missing_targets": missing_targets,
                "recovery_boxes": [current_box] if current_box else [],
                "recovered_box": recovered_box,
                "retention": retention,
            }
            print(
                "[dyfo] spatial-context recovery recovered=%s box=%s"
                % (trace["recovered"], recovered_box)
            )
            return recovered_box, trace

        recovery_result = self._dyfo_locate_required_targets(original, missing_targets, allow_joined_query=True)
        recovery_boxes = list(recovery_result.get("support_boxes", []))
        if current_box:
            recovery_boxes.append(current_box)
        recovered_box = self._dyfo_union_boxes(
            recovery_boxes, original.size, getattr(self.args, "dyfo_focus_padding", 1.2)
        )
        if recovered_box is None:
            recovered_box = self._dyfo_expand_box(
                current_box, original.size, getattr(self.args, "dyfo_scatter_scale", 1.6)
            ) if current_box else (0, 0, original.width, original.height)

        crop = self._dyfo_crop_for_node(original, recovered_box)
        retention = self._dyfo_check_all_targets_present(crop, required_targets, question)
        trace = {
            "recovered": retention["all_targets_present"],
            "missing_targets": missing_targets,
            "recovery_boxes": recovery_boxes,
            "recovered_box": recovered_box,
            "retention": retention,
        }
        print(
            "[dyfo] missing-target recovery recovered=%s box=%s"
            % (trace["recovered"], recovered_box)
        )
        return recovered_box, trace

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

    def _dyfo_resize_like_original(self, crop, original):
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return crop.resize(original.size, resampling)

    def _dyfo_build_answer_image(self, original, focus_crop, image_filename):
        mode = getattr(self.args, "dyfo_answer_image_mode", "crop")
        os.makedirs(self.args.cache_path, exist_ok=True)

        if mode == "crop":
            answer_image = focus_crop
        elif mode == "resized_crop":
            answer_image = self._dyfo_resize_like_original(focus_crop, original)
        else:
            resized_focus = self._dyfo_resize_like_original(focus_crop, original)
            if mode == "concat_vertical":
                answer_image = Image.new("RGB", (original.width, original.height * 2), "white")
                answer_image.paste(original, (0, 0))
                answer_image.paste(resized_focus, (0, original.height))
            else:
                answer_image = Image.new("RGB", (original.width * 2, original.height), "white")
                answer_image.paste(original, (0, 0))
                answer_image.paste(resized_focus, (original.width, 0))

        answer_image_path = os.path.join(
            self.args.cache_path,
            "dyfo_answer_%s_%s" % (mode, image_filename)
        )
        answer_image.save(answer_image_path)
        return answer_image_path

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

    def _dyfo_answer_from_crop(self, crop, focus_text, question, original=None, highlight=None):
        temp_path = None
        try:
            mode = getattr(self.args, "dyfo_node_answer_image_mode", "concat_horizontal")
            if original is None:
                mode = "crop"
            if mode == "active_look_horizontal" and original is not None and highlight is not None:
                resized_highlight = self._dyfo_resize_like_original(highlight, original)
                resized_crop = self._dyfo_resize_like_original(crop, original)
                answer_image = Image.new("RGB", (original.width * 3, original.height), "white")
                answer_image.paste(original, (0, 0))
                answer_image.paste(resized_highlight, (original.width, 0))
                answer_image.paste(resized_crop, (original.width * 2, 0))
                image_description = (
                    "the original image on the left, a detector-agreement highlight in the middle "
                    "(green confirmed, red suspicious), and the focused region on the right"
                )
            elif mode == "crop":
                answer_image = crop
                image_description = "the focused image region"
            elif mode == "resized_crop":
                answer_image = self._dyfo_resize_like_original(crop, original)
                image_description = "the resized focused image region"
            else:
                resized_crop = self._dyfo_resize_like_original(crop, original)
                if mode == "concat_vertical":
                    answer_image = Image.new("RGB", (original.width, original.height * 2), "white")
                    answer_image.paste(original, (0, 0))
                    answer_image.paste(resized_crop, (0, original.height))
                    image_description = "the original image on top and its focused region below"
                else:
                    answer_image = Image.new("RGB", (original.width * 2, original.height), "white")
                    answer_image.paste(original, (0, 0))
                    answer_image.paste(resized_crop, (original.width, 0))
                    image_description = "the original image on the left and its focused region on the right"
            temp_path = os.path.join(
                self.args.cache_path,
                "dyfo_answer_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
            )
            os.makedirs(self.args.cache_path, exist_ok=True)
            answer_image.save(temp_path)
            if uses_yes_no_scoring(self.args):
                prompt = (
                    "Answer the visual question using %s.\n"
                    "Use the original image for global context and the focused region for detail.\n"
                    "Return only yes or no.\n"
                    "Question: %s\n"
                    "Visual focus: %s\n"
                    "Answer:"
                ) % (image_description, question, focus_text)
            else:
                prompt = (
                    "Answer the visual question using %s.\n"
                    "Use the original image for global context and the focused region for detail.\n"
                    "Treat the focused region as additional visual evidence, not as a separate scene.\n"
                    "Return only one word or a short phrase.\n"
                    "Question: %s\n"
                    "Visual focus: %s\n"
                    "Answer:"
                ) % (image_description, question, focus_text)
            response, confidence_details = self._call_llm_with_token_confidence(
                prompt,
                image_path=temp_path,
                max_new_tokens=getattr(self.args, "dyfo_answer_max_tokens", 32),
            )
            answer = self._clean_short_answer(self._extract_answer_from_response(response))
            return answer, response, prompt, confidence_details
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _dyfo_box_iou(box_a, box_b):
        if not box_a or not box_b:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
        bx1, by1, bx2, by2 = [float(value) for value in box_b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _dyfo_build_region_audit(self, key, question, image_path, original, nodes, best_node):
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key)).strip("_") or "sample"
        audit_root = getattr(self.args, "dyfo_region_audit_dir", "") or os.path.join(
            self.args.output_path, "region_audit_assets"
        )
        sample_dir = os.path.join(audit_root, safe_key)
        save_crops = bool(getattr(self.args, "dyfo_region_audit_save_crops", False))
        if save_crops:
            os.makedirs(sample_dir, exist_ok=True)

        node_index = {id(node): index for index, node in enumerate(nodes)}
        node_records = []
        focused_boxes = []
        for index, node in enumerate(nodes):
            crop_path = ""
            if node.depth > 0:
                focused_boxes.append(tuple(node.box))
                if save_crops:
                    action = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node.action))
                    crop_path = os.path.join(
                        sample_dir, "node_%02d_depth%d_%s.jpg" % (index, node.depth, action)
                    )
                    self._dyfo_crop_for_node(original, node.box).save(crop_path, quality=95)
            confidence = getattr(node, "local_answer_confidence", {}) or {}
            node_records.append({
                "index": index,
                "parent_index": node_index.get(id(node.parent)) if node.parent is not None else None,
                "depth": node.depth,
                "action": node.action,
                "focus": node.focus,
                "box": node.box,
                "area_ratio": node.area_ratio,
                "reward": node.reward,
                "visits": node.visits,
                "all_targets_present": node.all_targets_present,
                "missing_targets": node.missing_targets,
                "visual_hit": node.visual_hit,
                "local_answer": node.local_answer,
                "local_answer_confidence": confidence.get("confidence", 0.0),
                "crop_path": crop_path,
                "is_best_node": node is best_node,
            })

        pairwise_ious = []
        for left in range(len(focused_boxes)):
            for right in range(left + 1, len(focused_boxes)):
                pairwise_ious.append(self._dyfo_box_iou(focused_boxes[left], focused_boxes[right]))
        unique_threshold = float(getattr(self.args, "dyfo_region_audit_unique_iou", 0.90))
        representatives = []
        for box in focused_boxes:
            if not any(self._dyfo_box_iou(box, representative) >= unique_threshold for representative in representatives):
                representatives.append(box)

        masked_path = ""
        masked_answer = ""
        masked_response = ""
        masked_confidence = {}
        if best_node.depth > 0:
            masked = original.copy()
            x1, y1, x2, y2 = [int(value) for value in best_node.box]
            masked.paste((127, 127, 127), (x1, y1, x2, y2))
            if save_crops:
                masked_path = os.path.join(sample_dir, "best_region_masked_original.jpg")
            else:
                os.makedirs(self.args.cache_path, exist_ok=True)
                masked_path = os.path.join(
                    self.args.cache_path,
                    "dyfo_audit_masked_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9)),
                )
            masked.save(masked_path, quality=95)
            masked_answer, masked_response, _, masked_confidence = self._dyfo_pure_baseline_answer(
                question, masked_path
            )
            if not save_crops and os.path.exists(masked_path):
                os.remove(masked_path)
                masked_path = ""

        return {
            "sample_key": key,
            "question": question,
            "original_image_path": image_path,
            "asset_dir": sample_dir if save_crops else "",
            "node_answer_image_mode": getattr(self.args, "dyfo_node_answer_image_mode", "concat_horizontal"),
            "node_count": len(nodes),
            "focused_node_count": len(focused_boxes),
            "unique_region_count": len(representatives),
            "unique_region_iou_threshold": unique_threshold,
            "pairwise_iou_mean": float(np.mean(pairwise_ious)) if pairwise_ious else 0.0,
            "pairwise_iou_min": float(np.min(pairwise_ious)) if pairwise_ious else 0.0,
            "pairwise_iou_max": float(np.max(pairwise_ious)) if pairwise_ious else 0.0,
            "best_node_index": node_index.get(id(best_node), 0),
            "best_box": best_node.box,
            "best_crop_area_ratio": best_node.area_ratio,
            "best_masked_image_path": masked_path,
            "best_masked_answer": masked_answer,
            "best_masked_response": masked_response,
            "best_masked_confidence": masked_confidence,
            "nodes": node_records,
        }

    def _dyfo_node_reward(self, visual_hit, lmm_consistent, area_ratio, all_targets_present=True):
        if not all_targets_present:
            return 0.0
        if not visual_hit:
            return 0.0
        consistency = 1.0 if lmm_consistent > 0 else 0.0
        area_score = 1.0 - area_ratio
        return max(0.0, min(1.0, consistency * area_score))

    def _dyfo_weighted_vote(self, nodes, use_node_confidence=False):
        vote_scores = defaultdict(float)
        norm_to_answer = {}
        vote_items = []
        for node in nodes:
            if getattr(node, "required_targets", []) and not getattr(node, "all_targets_present", False):
                continue
            answer = self._clean_short_answer(getattr(node, "local_answer", ""))
            if not answer:
                continue
            norm = normalize_vqa_answer(answer)
            if not norm:
                continue
            mean_value = node.value / max(1, node.visits)
            evidence_weight = max(float(getattr(node, "reward", 0.0)), float(mean_value), 1e-6)
            confidence_details = getattr(node, "local_answer_confidence", {}) or {}
            node_confidence = float(confidence_details.get("confidence", 0.0))
            weight = evidence_weight * node_confidence if use_node_confidence else evidence_weight
            vote_scores[norm] += weight
            norm_to_answer.setdefault(norm, answer)
            vote_items.append({
                "focus": node.focus,
                "action": node.action,
                "depth": node.depth,
                "answer": answer,
                "normalized_answer": norm,
                "weight": weight,
                "evidence_weight": evidence_weight,
                "node_token_confidence": node_confidence,
                "node_token_confidence_details": confidence_details,
                "reward": node.reward,
                "visits": node.visits,
            })
        if not vote_scores:
            return "", {"vote_scores": {}, "vote_items": vote_items}
        best_norm = max(vote_scores, key=vote_scores.get)
        supporting_items = [item for item in vote_items if item["normalized_answer"] == best_norm]
        support_evidence_weight = sum(item["evidence_weight"] for item in supporting_items)
        total_evidence_weight = sum(item["evidence_weight"] for item in vote_items)
        weighted_confidence_sum = sum(
            item["node_token_confidence"] * item["evidence_weight"]
            for item in supporting_items
        )
        candidate_confidence = (
            weighted_confidence_sum / support_evidence_weight
            if support_evidence_weight > 0 else 0.0
        )
        support_ratio = (
            support_evidence_weight / total_evidence_weight
            if total_evidence_weight > 0 else 0.0
        )
        return norm_to_answer[best_norm], {
            "best_normalized_answer": best_norm,
            "vote_scores": dict(vote_scores),
            "vote_items": vote_items,
            "node_confidence_weighted_vote": bool(use_node_confidence),
            "dyfo_candidate_confidence": candidate_confidence,
            "dyfo_candidate_support_ratio": support_ratio,
            "dyfo_candidate_support_nodes": len(supporting_items),
        }

    def _dyfo_node_confidence_override(self, baseline_answer, baseline_confidence,
                                       dyfo_answer, decision_trace):
        baseline_norm = normalize_vqa_answer(baseline_answer)
        dyfo_norm = normalize_vqa_answer(dyfo_answer)
        pure_confidence = float((baseline_confidence or {}).get("confidence", 0.0))
        dyfo_confidence = float(decision_trace.get("dyfo_candidate_confidence", 0.0))
        support_ratio = float(decision_trace.get("dyfo_candidate_support_ratio", 0.0))
        support_nodes = int(decision_trace.get("dyfo_candidate_support_nodes", 0))
        threshold = float(getattr(self.args, "dyfo_node_confidence_threshold", 0.80))
        margin = float(getattr(self.args, "dyfo_node_confidence_margin", 0.10))
        min_support_ratio = float(
            getattr(self.args, "dyfo_node_confidence_support_ratio", 0.60)
        )
        min_support_nodes = int(getattr(self.args, "dyfo_node_confidence_min_support", 2))
        answers_disagree = bool(
            baseline_norm and dyfo_norm and baseline_norm != dyfo_norm
        )
        gates = {
            "answers_disagree": answers_disagree,
            "absolute_confidence": dyfo_confidence >= threshold,
            "relative_confidence": dyfo_confidence >= pure_confidence + margin,
            "support_ratio": support_ratio >= min_support_ratio,
            "support_nodes": support_nodes >= min_support_nodes,
            "valid_focus_crop": not decision_trace.get("final_crop_fallback_to_original", False),
            "all_targets_present": bool(
                decision_trace.get("best_all_targets_present", False)
            ),
        }
        override = all(gates.values())
        trace = {
            "baseline_answer": baseline_answer,
            "baseline_token_confidence": baseline_confidence,
            "dyfo_candidate_answer": dyfo_answer,
            "dyfo_candidate_confidence": dyfo_confidence,
            "confidence_threshold": threshold,
            "confidence_margin": margin,
            "confidence_delta": dyfo_confidence - pure_confidence,
            "support_ratio": support_ratio,
            "minimum_support_ratio": min_support_ratio,
            "support_nodes": support_nodes,
            "minimum_support_nodes": min_support_nodes,
            "gates": gates,
            "override_applied": override,
            "fallback_reason": "" if override else ",".join(
                name for name, passed in gates.items() if not passed
            ),
        }
        return (dyfo_answer if override else baseline_answer), trace

    def _dyfo_pure_baseline_answer(self, question, image_path):
        prompt = (
            "Answer the visual question from the original image only. "
            "Use a single word or short phrase and do not explain.\n"
            "Question: %s\n"
            "Final Answer:"
        ) % question
        response, confidence_details = self._call_llm_with_token_confidence(
            prompt,
            image_path=image_path,
            max_new_tokens=getattr(self.args, "dyfo_answer_max_tokens", 32),
        )
        answer = self._clean_short_answer(self._extract_answer_from_response(response))
        return answer, response, prompt, confidence_details

    def _dyfo_token_confidence_override(self, question, answer_image_path, baseline_answer,
                                        baseline_confidence, dyfo_answer, decision_trace):
        baseline_norm = normalize_vqa_answer(baseline_answer)
        dyfo_norm = normalize_vqa_answer(dyfo_answer)
        base_trace = {
            "baseline_answer": baseline_answer,
            "baseline_token_confidence": baseline_confidence,
            "dyfo_candidate_answer": dyfo_answer,
            "answers_disagree": bool(baseline_norm and dyfo_norm and baseline_norm != dyfo_norm),
            "verifier_called": False,
            "override_applied": False,
            "fallback_reason": "",
        }
        if not baseline_norm:
            base_trace["fallback_reason"] = "empty_baseline"
            return baseline_answer, base_trace
        if not dyfo_norm:
            base_trace["fallback_reason"] = "empty_dyfo_candidate"
            return baseline_answer, base_trace
        if baseline_norm == dyfo_norm:
            base_trace["fallback_reason"] = "answers_agree"
            return baseline_answer, base_trace
        if decision_trace.get("final_crop_fallback_to_original"):
            base_trace["fallback_reason"] = "dyfo_crop_fallback"
            return baseline_answer, base_trace
        if not decision_trace.get("best_all_targets_present", False):
            base_trace["fallback_reason"] = "required_targets_missing"
            return baseline_answer, base_trace
        if not decision_trace.get("vote_items"):
            base_trace["fallback_reason"] = "empty_vote"
            return baseline_answer, base_trace

        prompt = (
            "Answer the visual question using the original view and the focused view. "
            "Use a single word or short phrase and do not explain.\n"
            "Question: %s\n"
            "Final Answer:"
        ) % question
        response, confidence_details = self._call_llm_with_token_confidence(
            prompt,
            image_path=answer_image_path,
            max_new_tokens=getattr(self.args, "dyfo_answer_max_tokens", 32),
        )
        verified_answer = self._clean_short_answer(self._extract_answer_from_response(response))
        verified_norm = normalize_vqa_answer(verified_answer)
        confidence = float(confidence_details.get("confidence", 0.0))
        baseline_score = float((baseline_confidence or {}).get("confidence", 0.0))
        threshold = float(getattr(self.args, "dyfo_token_confidence_threshold", 0.95))
        margin = float(getattr(self.args, "dyfo_token_confidence_margin", 0.0))
        override = (
            verified_norm == dyfo_norm
            and confidence >= threshold
            and confidence - baseline_score >= margin
        )
        base_trace.update({
            "verifier_called": True,
            "verifier_prompt": prompt,
            "verifier_response": response,
            "verified_answer": verified_answer,
            "verified_answer_matches_dyfo": verified_norm == dyfo_norm,
            "dyfo_token_confidence": confidence_details,
            "confidence_threshold": threshold,
            "confidence_margin": margin,
            "confidence_delta": confidence - baseline_score,
            "override_applied": override,
            "fallback_reason": "" if override else "token_confidence_gate_failed",
        })
        return (dyfo_answer if override else baseline_answer), base_trace

    def _dyfo_parse_parallel_statements(self, response):
        text = str(response).strip()
        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            pure_statement = str(parsed.get("pure_statement", "")).strip()
            dyfo_statement = str(parsed.get("dyfo_statement", "")).strip()
            if pure_statement and dyfo_statement and pure_statement != dyfo_statement:
                return pure_statement, dyfo_statement
        return "", ""

    def _dyfo_build_parallel_statements(self, question, baseline_answer, dyfo_answer):
        prompt = (
            "Rewrite the visual question and each candidate answer as a standalone declarative image caption.\n"
            "The two captions must use the same sentence structure and differ only where the candidate answers differ.\n"
            "Preserve the exact entities, attributes, counts, relations, and polarity. Do not decide which answer is correct.\n"
            "Return exactly one JSON object with keys pure_statement and dyfo_statement.\n"
            "Question: %s\n"
            "Pure candidate: %s\n"
            "DyFo candidate: %s"
        ) % (question, baseline_answer, dyfo_answer)
        response = self._call_llm(
            prompt,
            image_path=None,
            max_new_tokens=96,
            use_images=False,
        )
        pure_statement, dyfo_statement = self._dyfo_parse_parallel_statements(response)
        return pure_statement, dyfo_statement, response, prompt

    def _dyfo_clip_statement_scores(self, original, crop, statements):
        if self.clip_full_model is None or self.clip_full_processor is None:
            raise RuntimeError("CLIP full model is required for clip_statement_override")
        text_inputs = self.clip_full_processor(
            text=list(statements),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        image_inputs = self.clip_full_processor(
            images=[original, crop],
            return_tensors="pt",
        )
        text_inputs = {key: value.to(self.clip_full_model.device) for key, value in text_inputs.items()}
        image_inputs = {key: value.to(self.clip_full_model.device) for key, value in image_inputs.items()}
        with torch.no_grad():
            text_features = self.clip_full_model.get_text_features(**text_inputs)
            image_features = self.clip_full_model.get_image_features(**image_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarities = image_features @ text_features.T
        scores = similarities.detach().float().cpu().tolist()
        return {
            "original_pure": scores[0][0],
            "original_dyfo": scores[0][1],
            "focus_pure": scores[1][0],
            "focus_dyfo": scores[1][1],
        }

    def _dyfo_clip_statement_override(self, question, original, crop, baseline_answer,
                                      dyfo_answer, decision_trace):
        baseline_norm = normalize_vqa_answer(baseline_answer)
        dyfo_norm = normalize_vqa_answer(dyfo_answer)
        trace = {
            "baseline_answer": baseline_answer,
            "dyfo_candidate_answer": dyfo_answer,
            "answers_disagree": bool(baseline_norm and dyfo_norm and baseline_norm != dyfo_norm),
            "clip_called": False,
            "override_applied": False,
            "fallback_reason": "",
        }
        if not baseline_norm:
            trace["fallback_reason"] = "empty_baseline"
            return baseline_answer, trace
        if not dyfo_norm:
            trace["fallback_reason"] = "empty_dyfo_candidate"
            return baseline_answer, trace
        if baseline_norm == dyfo_norm:
            trace["fallback_reason"] = "answers_agree"
            return baseline_answer, trace
        if decision_trace.get("final_crop_fallback_to_original"):
            trace["fallback_reason"] = "dyfo_crop_fallback"
            return baseline_answer, trace
        if not decision_trace.get("best_all_targets_present", False):
            trace["fallback_reason"] = "required_targets_missing"
            return baseline_answer, trace
        if not decision_trace.get("vote_items"):
            trace["fallback_reason"] = "empty_vote"
            return baseline_answer, trace

        pure_statement, dyfo_statement, rewrite_response, rewrite_prompt = (
            self._dyfo_build_parallel_statements(question, baseline_answer, dyfo_answer)
        )
        trace.update({
            "statement_rewrite_prompt": rewrite_prompt,
            "statement_rewrite_response": rewrite_response,
            "pure_statement": pure_statement,
            "dyfo_statement": dyfo_statement,
        })
        if not pure_statement or not dyfo_statement:
            trace["fallback_reason"] = "statement_rewrite_failed"
            return baseline_answer, trace

        scores = self._dyfo_clip_statement_scores(
            original,
            crop,
            [pure_statement, dyfo_statement],
        )
        original_margin = scores["original_dyfo"] - scores["original_pure"]
        focus_margin = scores["focus_dyfo"] - scores["focus_pure"]
        combined_margin = 0.5 * (original_margin + focus_margin)
        focus_gain = focus_margin - original_margin
        margin_threshold = float(getattr(self.args, "dyfo_clip_statement_margin", 0.0))
        gain_threshold = float(getattr(self.args, "dyfo_clip_statement_focus_gain", 0.0))
        override = (
            focus_margin >= margin_threshold
            and combined_margin >= margin_threshold
            and focus_gain >= gain_threshold
        )
        trace.update({
            "clip_called": True,
            "clip_scores": scores,
            "original_margin": original_margin,
            "focus_margin": focus_margin,
            "combined_margin": combined_margin,
            "focus_gain": focus_gain,
            "margin_threshold": margin_threshold,
            "focus_gain_threshold": gain_threshold,
            "override_applied": override,
            "fallback_reason": "" if override else "clip_statement_gate_failed",
        })
        return (dyfo_answer if override else baseline_answer), trace

    def _dyfo_parse_conservative_override(self, response, baseline_answer, dyfo_answer):
        text = str(response).strip()
        decision_match = re.search(
            r"decision\s*:\s*(KEEP_BASELINE|OVERRIDE_WITH_DYFO)", text, flags=re.IGNORECASE
        )
        confidence_match = re.search(r"confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        strength_match = re.search(
            r"evidence\s+strength\s*:\s*(WEAK|MODERATE|STRONG|EXTREME)",
            text,
            flags=re.IGNORECASE,
        )
        answer_match = re.search(r"final\s+answer\s*:\s*(.+)", text, flags=re.IGNORECASE)

        decision = decision_match.group(1).upper() if decision_match else "KEEP_BASELINE"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        strength = strength_match.group(1).lower() if strength_match else "unknown"
        parsed_answer = self._clean_short_answer(answer_match.group(1)) if answer_match else ""
        required_strength = getattr(self.args, "dyfo_override_required_strength", "extreme")
        strength_rank = {"unknown": 0, "weak": 1, "moderate": 2, "strong": 3, "extreme": 4}
        threshold = float(getattr(self.args, "dyfo_override_confidence_threshold", 95.0))
        dyfo_norm = normalize_vqa_answer(dyfo_answer)
        parsed_norm = normalize_vqa_answer(parsed_answer)
        override = (
            decision == "OVERRIDE_WITH_DYFO"
            and confidence >= threshold
            and strength_rank.get(strength, 0) >= strength_rank[required_strength]
            and bool(dyfo_norm)
            and parsed_norm == dyfo_norm
        )
        final_answer = dyfo_answer if override else baseline_answer
        return final_answer, {
            "parsed_decision": decision,
            "parsed_confidence": confidence,
            "parsed_evidence_strength": strength,
            "parsed_final_answer": parsed_answer,
            "confidence_threshold": threshold,
            "required_evidence_strength": required_strength,
            "override_applied": override,
        }

    def _dyfo_conservative_override(self, question, answer_image_path, baseline_answer, dyfo_answer,
                                    evidence, decision_trace):
        baseline_norm = normalize_vqa_answer(baseline_answer)
        dyfo_norm = normalize_vqa_answer(dyfo_answer)
        base_trace = {
            "baseline_answer": baseline_answer,
            "dyfo_candidate_answer": dyfo_answer,
            "answers_disagree": bool(baseline_norm and dyfo_norm and baseline_norm != dyfo_norm),
            "arbiter_called": False,
            "override_applied": False,
            "fallback_reason": "",
        }
        if not baseline_norm:
            base_trace["fallback_reason"] = "empty_baseline"
            return baseline_answer, base_trace
        if not dyfo_norm:
            base_trace["fallback_reason"] = "empty_dyfo_candidate"
            return baseline_answer, base_trace
        if baseline_norm == dyfo_norm:
            base_trace["fallback_reason"] = "answers_agree"
            return baseline_answer, base_trace
        if decision_trace.get("final_crop_fallback_to_original"):
            base_trace["fallback_reason"] = "dyfo_crop_fallback"
            return baseline_answer, base_trace
        if not decision_trace.get("best_all_targets_present", False):
            base_trace["fallback_reason"] = "required_targets_missing"
            return baseline_answer, base_trace
        if not decision_trace.get("vote_items"):
            base_trace["fallback_reason"] = "empty_vote"
            return baseline_answer, base_trace

        vote_scores = decision_trace.get("vote_scores", {})
        prompt = (
            "You are a highly conservative visual answer arbiter. The baseline answer was produced "
            "from the original image. The DyFo answer was produced after searching and magnifying "
            "visual evidence. Keep the baseline unless the provided image and DyFo evidence make the "
            "baseline unmistakably wrong and the DyFo answer unmistakably correct. Ordinary confidence, "
            "a plausible crop, or a detector hit is not enough. Override only with extreme, direct, "
            "unambiguous visual evidence. If uncertain for any reason, keep the baseline.\n"
            "The image contains the original view and the DyFo focused view.\n"
            "Question: %s\n"
            "Pure baseline answer: %s\n"
            "DyFo candidate answer: %s\n"
            "DyFo weighted vote scores: %s\n"
            "DyFo evidence: %s\n"
            "Output exactly these five lines:\n"
            "Decision: KEEP_BASELINE or OVERRIDE_WITH_DYFO\n"
            "Confidence: <integer 0-100>\n"
            "Evidence Strength: WEAK, MODERATE, STRONG, or EXTREME\n"
            "Reason: <one short sentence grounded in visible evidence>\n"
            "Final Answer: <exactly the baseline answer or DyFo candidate answer>"
        ) % (
            question,
            baseline_answer,
            dyfo_answer,
            json.dumps(vote_scores, ensure_ascii=False),
            self._truncate_text(evidence, 700),
        )
        response = self._call_llm(
            prompt,
            image_path=answer_image_path,
            max_new_tokens=getattr(self.args, "dyfo_override_max_tokens", 160),
        )
        final_answer, parse_trace = self._dyfo_parse_conservative_override(
            response, baseline_answer, dyfo_answer
        )
        base_trace.update(parse_trace)
        base_trace.update({
            "arbiter_called": True,
            "arbiter_prompt": prompt,
            "arbiter_response": response,
            "fallback_reason": "" if parse_trace["override_applied"] else "arbiter_kept_baseline",
        })
        return final_answer, base_trace

    def _run_dyfo_visual_evidence_search(self, data_row, obj_list, attr_list):
        key = data_row["key"]
        image_path = data_row["image_path"]
        question = data_row["question"]
        question_type = self._classify_vqa_question_type(question)
        if not self._dyfo_should_trigger(question, question_type):
            return {"evidence": "", "focus_image_path": None, "trace": "skipped_by_trigger"}

        baseline_answer = ""
        baseline_response = ""
        baseline_prompt = ""
        baseline_confidence = {}
        if self.args.dyfo_decision_mode in ("conservative_override", "token_confidence_override", "node_confidence_override", "clip_statement_override"):
            baseline_answer, baseline_response, baseline_prompt, baseline_confidence = self._dyfo_pure_baseline_answer(
                question, image_path
            )
            print("[dyfo override] pure baseline answer:", baseline_answer)
            print("[dyfo override] pure token confidence:", baseline_confidence.get("confidence", 0.0))

        free_key_objects, key_object_response = self.extract_key_objects_from_question(question)
        selected_objects = []
        candidate_fallback_triggered = False

        def ensure_candidate_selected(reason):
            nonlocal selected_objects, candidate_fallback_triggered
            if not selected_objects:
                candidate_fallback_triggered = True
                _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
                print("[dyfo] candidate-selected objects (%s): %s" % (reason, selected_objects))
            return selected_objects

        if not free_key_objects:
            ensure_candidate_selected("free_key_object_extraction_failed")
        else:
            print("[dyfo] candidate-selected objects (not triggered yet): []")

        required_targets = self._dyfo_unique_targets(free_key_objects or selected_objects)
        obj_list = list(dict.fromkeys((free_key_objects or []) + (obj_list or []) + selected_objects))
        if not obj_list:
            obj_list = selected_objects

        def locate_with_lazy_candidates(image_pil, focus_text):
            box, query, source, fallback_triggered = self._dyfo_locate_with_fallbacks(
                image_pil, focus_text, key_objects=free_key_objects, selected_objects=selected_objects
            )
            if box is None and not selected_objects:
                ensure_candidate_selected("langsam_focus_and_key_object_failed")
                box, query, source, fallback_triggered = self._dyfo_locate_with_fallbacks(
                    image_pil, "", key_objects=[], selected_objects=selected_objects
                )
            return box, query, source, fallback_triggered

        original = Image.open(image_path).convert("RGB")
        image_area = max(1, original.width * original.height)
        if required_targets and getattr(self.args, "dyfo_dual_visual_experts", False):
            initial_required_locate = self._dyfo_dual_expert_locate_required_targets(
                original, required_targets
            )
        elif required_targets:
            initial_required_locate = self._dyfo_locate_required_targets(
                original, required_targets, allow_joined_query=True
            )
        else:
            initial_required_locate = {
            "required_targets": [],
            "target_boxes": {},
            "missing_targets": [],
            "all_targets_located": True,
            "support_boxes": [],
            "joined_box": None,
            "union_box": None,
            "query_log": [],
        }
        active_look_highlight = None
        if getattr(self.args, "dyfo_dual_visual_experts", False):
            active_look_highlight = self._dyfo_build_active_look_highlight(
                original, initial_required_locate
            )
        initial_focus, initial_focus_response = self._dyfo_initial_focus(
            question, obj_list=selected_objects or obj_list, key_objects=free_key_objects
        )
        print("[dyfo] final initial focus cue:", initial_focus)
        print("[dyfo] free key-object fallback triggered:", not bool(free_key_objects))
        print("[dyfo] required_targets:", required_targets)
        print("[dyfo] initial required-target missing:", initial_required_locate.get("missing_targets", []))

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
                self.local_answer_confidence = {}
                self.langsam_query = ""
                self.langsam_query_source = "root" if action == "root" else ""
                self.langsam_fallback_triggered = False
                self.required_targets = []
                self.target_boxes = {}
                self.missing_targets = []
                self.all_targets_present = True
                self.target_retention_trace = {}
                self.target_recovery_trace = {}
                self.area_ratio = 1.0
                self.answer_image_fallback = False

        root = _Node(initial_focus, (0, 0, original.width, original.height), 0)
        root.required_targets = required_targets
        root.target_boxes = initial_required_locate.get("target_boxes", {})
        root.missing_targets = initial_required_locate.get("missing_targets", [])
        root.all_targets_present = True
        root.target_retention_trace = {
            "all_targets_present": True,
            "present_targets": required_targets,
            "missing_targets": [],
            "locate_trace": initial_required_locate.get("query_log", []),
            "note": "root uses the original image as fallback when no valid focused crop exists",
        }
        if initial_required_locate.get("suspicious_regions"):
            root.untried.insert(0, "expert_suspicious")
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
                if action == "expert_suspicious":
                    suspicious_region = max(
                        initial_required_locate.get("suspicious_regions", []),
                        key=lambda item: float(item.get("score", 0.0)),
                    )
                    focus = "%s suspicious detector region" % suspicious_region["target"]
                    focus_response = "dual-expert disagreement selected for stare"
                    box = self._dyfo_expand_box(
                        suspicious_region["box"], original.size, 1.5
                    )
                    locate_query = suspicious_region["target"]
                    locate_source = "dual_expert_suspicious"
                    locate_fallback = False
                else:
                    parent_crop_path = None
                    try:
                        parent_crop_path = os.path.join(
                            self.args.cache_path,
                            "dyfo_parent_%s_%s.jpg" % (os.getpid(), random.randint(0, 10**9))
                        )
                        os.makedirs(self.args.cache_path, exist_ok=True)
                        parent_crop.save(parent_crop_path)
                        focus, focus_response = self._dyfo_refine_focus(
                            question, leaf.focus, action, parent_crop_path
                        )
                    finally:
                        if parent_crop_path and os.path.exists(parent_crop_path):
                            os.remove(parent_crop_path)

                if action == "semantic_focus":
                    base_crop = parent_crop
                    local_box, locate_query, locate_source, locate_fallback = locate_with_lazy_candidates(base_crop, focus)
                    if local_box:
                        box = self._dyfo_abs_box_from_local(local_box, leaf.box)
                    else:
                        box = leaf.box
                elif action == "semantic_scatter":
                    located, locate_query, locate_source, locate_fallback = locate_with_lazy_candidates(parent_crop, focus)
                    if located:
                        local_abs = self._dyfo_abs_box_from_local(located, leaf.box)
                        box = self._dyfo_expand_box(local_abs, original.size, self.args.dyfo_scatter_scale)
                    else:
                        box = self._dyfo_expand_box(leaf.box, original.size, self.args.dyfo_scatter_scale)

                target_locate_trace = {}
                if required_targets:
                    parent_target_locate = self._dyfo_locate_required_targets(
                        parent_crop, required_targets, allow_joined_query=True
                    )
                    abs_support_boxes = [
                        self._dyfo_abs_box_from_local(local_target_box, leaf.box)
                        for local_target_box in parent_target_locate.get("support_boxes", [])
                    ]
                    abs_support_boxes = [local_target_box for local_target_box in abs_support_boxes if local_target_box]
                    union_inputs = list(abs_support_boxes)
                    if box:
                        union_inputs.append(box)
                    target_union = self._dyfo_union_boxes(
                        union_inputs, original.size, self.args.dyfo_focus_padding
                    )
                    if target_union is not None:
                        box = target_union
                    target_locate_trace = parent_target_locate
                box = self._dyfo_expand_box(box, original.size, self.args.dyfo_focus_padding)
                target = _Node(focus, box, leaf.depth + 1, parent=leaf, action=action)
                target.focus_response = focus_response
                target.langsam_query = locate_query
                target.langsam_query_source = locate_source
                target.langsam_fallback_triggered = locate_fallback
                target.required_targets = required_targets
                target.target_retention_trace = {"candidate_locate": target_locate_trace}
                leaf.children.append(target)
                nodes.append(target)

            crop = self._dyfo_crop_for_node(original, target.box)
            hit_box, hit_query, hit_source, hit_fallback = locate_with_lazy_candidates(crop, target.focus)
            visual_hit = hit_box is not None
            retention = self._dyfo_check_all_targets_present(crop, required_targets, question)
            if required_targets and not retention["all_targets_present"]:
                recovered_box, recovery_trace = self._dyfo_recover_missing_targets(
                    original, target.box, required_targets, retention["missing_targets"], question
                )
                recovered_crop = self._dyfo_crop_for_node(original, recovered_box)
                recovered_retention = recovery_trace.get("retention", retention)
                if recovered_retention.get("all_targets_present"):
                    target.box = recovered_box
                    target.image_region = recovered_box
                    crop = recovered_crop
                    retention = recovered_retention
                target.target_recovery_trace = recovery_trace
                hit_box, hit_query, hit_source, hit_fallback = locate_with_lazy_candidates(crop, target.focus)
                visual_hit = hit_box is not None
            lmm_consistent, lmm_reply = self._dyfo_consistency_check(crop, target.focus, question)
            x1, y1, x2, y2 = target.box
            area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / image_area))
            reward = self._dyfo_node_reward(
                visual_hit, lmm_consistent, area_ratio,
                all_targets_present=retention["all_targets_present"]
            )
            target.reward = reward
            target.visual_hit = visual_hit
            target.lmm_reply = lmm_reply
            target.area_ratio = area_ratio
            target.all_targets_present = retention["all_targets_present"]
            target.missing_targets = retention["missing_targets"]
            target.target_boxes = retention["target_boxes"]
            previous_trace = target.target_retention_trace or {}
            previous_trace.update(retention)
            target.target_retention_trace = previous_trace
            if hit_query:
                target.langsam_query = hit_query
                target.langsam_query_source = hit_source
                target.langsam_fallback_triggered = target.langsam_fallback_triggered or hit_fallback
            if (
                self.args.dyfo_decision_mode in ("best_focus_answer", "weighted_vote", "conservative_override", "token_confidence_override", "node_confidence_override", "clip_statement_override")
                and not target.local_answer
                and (not required_targets or target.all_targets_present)
            ):
                local_answer, local_response, local_prompt, local_confidence = self._dyfo_answer_from_crop(
                    crop, target.focus, question, original=original,
                    highlight=active_look_highlight,
                )
                target.local_answer = local_answer
                target.local_answer_response = local_response
                target.local_answer_prompt = local_prompt
                target.local_answer_confidence = local_confidence
            node = target
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

        valid_nodes = [
            node for node in nodes
            if (not getattr(node, "required_targets", []) or getattr(node, "all_targets_present", False))
        ]
        focused_valid_nodes = [node for node in valid_nodes if node.depth > 0]
        if focused_valid_nodes:
            best_node = max(focused_valid_nodes, key=lambda n: (n.reward, n.value / max(1, n.visits), -n.depth))
            final_crop_fallback = False
        else:
            best_node = root
            best_node.answer_image_fallback = True
            final_crop_fallback = True
            print("[dyfo] no focused crop passed all-target retention; fallback to original image")
        best_crop = self._dyfo_crop_for_node(original, best_node.box)
        if self.args.dyfo_decision_mode == "best_focus_answer" and not best_node.local_answer:
            local_answer, local_response, local_prompt, local_confidence = self._dyfo_answer_from_crop(
                best_crop, best_node.focus, question, original=original,
                highlight=active_look_highlight,
            )
            best_node.local_answer = local_answer
            best_node.local_answer_response = local_response
            best_node.local_answer_prompt = local_prompt
            best_node.local_answer_confidence = local_confidence
        image_filename = os.path.basename(image_path)
        focus_image_path = os.path.join(self.args.cache_path, "dyfo_focus_%s" % image_filename)
        os.makedirs(self.args.cache_path, exist_ok=True)
        best_crop.save(focus_image_path)
        active_look_highlight_path = None
        if active_look_highlight is not None:
            active_look_highlight_path = os.path.join(
                self.args.cache_path, "dyfo_active_look_%s" % image_filename
            )
            active_look_highlight.save(active_look_highlight_path)
        answer_image_path = self._dyfo_build_answer_image(original, best_crop, image_filename)

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
            "Focus: %s. Search action: %s. Reward: %.3f. Required targets: %s. Evidence: %s"
            % (
                best_node.focus,
                best_node.action,
                best_node.reward,
                ", ".join(required_targets) if required_targets else "none",
                self._truncate_text(evidence_response, 700),
            )
        )
        final_answer = ""
        decision_trace = {
            "mode": self.args.dyfo_decision_mode,
            "node_answer_image_mode": getattr(
                self.args, "dyfo_node_answer_image_mode", "concat_horizontal"
            ),
            "dual_visual_experts": bool(
                getattr(self.args, "dyfo_dual_visual_experts", False)
            ),
            "dual_expert_conflict_rate": initial_required_locate.get("conflict_rate"),
            "dual_expert_agreement_score": initial_required_locate.get("agreement_score"),
            "dual_expert_look_mode": initial_required_locate.get("look_mode"),
            "dual_expert_confirmed_count": len(
                initial_required_locate.get("confirmed_regions", [])
            ),
            "dual_expert_suspicious_count": len(
                initial_required_locate.get("suspicious_regions", [])
            ),
            "active_look_highlight_path": active_look_highlight_path,
            "best_focus_answer": best_node.local_answer,
            "free_key_objects": free_key_objects,
            "required_targets": required_targets,
            "candidate_selected_objects": selected_objects,
            "candidate_fallback_triggered": candidate_fallback_triggered,
            "best_focus": best_node.focus,
            "best_langsam_query": best_node.langsam_query,
            "best_langsam_query_source": best_node.langsam_query_source,
            "best_langsam_fallback_triggered": best_node.langsam_fallback_triggered,
            "best_all_targets_present": best_node.all_targets_present,
            "best_missing_targets": best_node.missing_targets,
            "best_target_retention_trace": best_node.target_retention_trace,
            "final_crop_fallback_to_original": final_crop_fallback,
        }
        if self.args.dyfo_decision_mode == "best_focus_answer":
            final_answer = best_node.local_answer
        elif self.args.dyfo_decision_mode in ("weighted_vote", "conservative_override", "token_confidence_override", "node_confidence_override", "clip_statement_override"):
            final_answer, vote_trace = self._dyfo_weighted_vote(
                nodes,
                use_node_confidence=self.args.dyfo_decision_mode == "node_confidence_override",
            )
            decision_trace.update(vote_trace)
            if self.args.dyfo_decision_mode == "conservative_override":
                dyfo_candidate_answer = final_answer
                final_answer, override_trace = self._dyfo_conservative_override(
                    question,
                    answer_image_path,
                    baseline_answer,
                    dyfo_candidate_answer,
                    evidence,
                    decision_trace,
                )
                decision_trace.update({
                    "pure_baseline_answer": baseline_answer,
                    "pure_baseline_prompt": baseline_prompt,
                    "pure_baseline_response": baseline_response,
                    "dyfo_candidate_answer": dyfo_candidate_answer,
                    "conservative_override": override_trace,
                    "conservative_final_answer": final_answer,
                })
                print("[dyfo conservative] DyFo candidate answer:", dyfo_candidate_answer)
                print("[dyfo conservative] override trace:", override_trace)
                print("[dyfo conservative] final answer:", final_answer)
            elif self.args.dyfo_decision_mode == "token_confidence_override":
                dyfo_candidate_answer = final_answer
                final_answer, override_trace = self._dyfo_token_confidence_override(
                    question,
                    answer_image_path,
                    baseline_answer,
                    baseline_confidence,
                    dyfo_candidate_answer,
                    decision_trace,
                )
                decision_trace.update({
                    "pure_baseline_answer": baseline_answer,
                    "pure_baseline_prompt": baseline_prompt,
                    "pure_baseline_response": baseline_response,
                    "pure_baseline_token_confidence": baseline_confidence,
                    "dyfo_candidate_answer": dyfo_candidate_answer,
                    "token_confidence_override": override_trace,
                    "token_confidence_final_answer": final_answer,
                })
                print("[dyfo token confidence] DyFo candidate answer:", dyfo_candidate_answer)
                print("[dyfo token confidence] override trace:", override_trace)
                print("[dyfo token confidence] final answer:", final_answer)
            elif self.args.dyfo_decision_mode == "node_confidence_override":
                dyfo_candidate_answer = final_answer
                final_answer, override_trace = self._dyfo_node_confidence_override(
                    baseline_answer,
                    baseline_confidence,
                    dyfo_candidate_answer,
                    decision_trace,
                )
                decision_trace.update({
                    "pure_baseline_answer": baseline_answer,
                    "pure_baseline_prompt": baseline_prompt,
                    "pure_baseline_response": baseline_response,
                    "pure_baseline_token_confidence": baseline_confidence,
                    "dyfo_candidate_answer": dyfo_candidate_answer,
                    "node_confidence_override": override_trace,
                    "node_confidence_final_answer": final_answer,
                })
                print("[dyfo node confidence] DyFo candidate answer:", dyfo_candidate_answer)
                print("[dyfo node confidence] override trace:", override_trace)
                print("[dyfo node confidence] final answer:", final_answer)
            elif self.args.dyfo_decision_mode == "clip_statement_override":
                dyfo_candidate_answer = final_answer
                final_answer, override_trace = self._dyfo_clip_statement_override(
                    question,
                    original,
                    best_crop,
                    baseline_answer,
                    dyfo_candidate_answer,
                    decision_trace,
                )
                decision_trace.update({
                    "pure_baseline_answer": baseline_answer,
                    "pure_baseline_prompt": baseline_prompt,
                    "pure_baseline_response": baseline_response,
                    "pure_baseline_token_confidence": baseline_confidence,
                    "dyfo_candidate_answer": dyfo_candidate_answer,
                    "clip_statement_override": override_trace,
                    "clip_statement_final_answer": final_answer,
                })
                print("[dyfo CLIP statement] DyFo candidate answer:", dyfo_candidate_answer)
                print("[dyfo CLIP statement] override trace:", override_trace)
                print("[dyfo CLIP statement] final answer:", final_answer)
        if getattr(self.args, "dyfo_region_audit", False):
            try:
                decision_trace["region_audit"] = self._dyfo_build_region_audit(
                    key, question, image_path, original, nodes, best_node
                )
                print("[dyfo region audit]", decision_trace["region_audit"])
            except Exception as exc:
                decision_trace["region_audit"] = {"error": str(exc), "sample_key": key}
                print("[dyfo region audit] failed:", exc)
        trace = {
            "free_key_objects": free_key_objects,
            "free_key_object_response": key_object_response,
            "required_targets": required_targets,
            "initial_required_target_locate": initial_required_locate,
            "candidate_selected_objects": selected_objects,
            "candidate_fallback_triggered": candidate_fallback_triggered,
            "initial_focus": initial_focus,
            "initial_focus_response": initial_focus_response,
            "best_focus": best_node.focus,
            "best_action": best_node.action,
            "best_box": best_node.box,
            "best_reward": best_node.reward,
            "best_langsam_query": best_node.langsam_query,
            "best_langsam_query_source": best_node.langsam_query_source,
            "best_langsam_fallback_triggered": best_node.langsam_fallback_triggered,
            "best_all_targets_present": best_node.all_targets_present,
            "best_missing_targets": best_node.missing_targets,
            "best_target_boxes": best_node.target_boxes,
            "best_target_retention_trace": best_node.target_retention_trace,
            "final_crop_fallback_to_original": final_crop_fallback,
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
                    "area_ratio": node.area_ratio,
                    "visits": node.visits,
                    "visual_hit": node.visual_hit,
                    "lmm_reply": node.lmm_reply,
                    "langsam_query": node.langsam_query,
                    "langsam_query_source": node.langsam_query_source,
                    "langsam_fallback_triggered": node.langsam_fallback_triggered,
                    "required_targets": node.required_targets,
                    "all_targets_present": node.all_targets_present,
                    "missing_targets": node.missing_targets,
                    "target_boxes": node.target_boxes,
                    "target_retention_trace": node.target_retention_trace,
                    "target_recovery_trace": node.target_recovery_trace,
                    "local_answer": node.local_answer,
                    "local_answer_confidence": node.local_answer_confidence,
                }
                for node in nodes
            ],
        }
        print("[dyfo] visual evidence:", evidence)
        print("[dyfo] best focus cue:", best_node.focus)
        print("[dyfo] best LangSAM query source:", best_node.langsam_query_source)
        print("[dyfo] best LangSAM fallback triggered:", best_node.langsam_fallback_triggered)
        print("[dyfo] best all-target present:", best_node.all_targets_present)
        print("[dyfo] best missing targets:", best_node.missing_targets)
        print("[dyfo] final crop fallback to original:", final_crop_fallback)
        if final_answer:
            print("[dyfo] final answer:", final_answer)
        return {
            "evidence": evidence,
            "focus_image_path": focus_image_path,
            "active_look_highlight_path": active_look_highlight_path,
            "answer_image_path": answer_image_path,
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
    
    # ------------------------------------------------------------------
    # Knowledge enhancement and retrieval
    # ------------------------------------------------------------------
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
        cached_record = self._sample_knowledge_cache_record(key, data_row)

        if cached_record and mode in ("notes", "hybrid"):
            cached_note = str(
                cached_record.get("knowledge_note")
                or cached_record.get("note")
                or cached_record.get("notes")
                or ""
            ).strip()
            if cached_note:
                if mode == "hybrid":
                    cached_items = self._knowledge_items_from_cache_record(cached_record)
                    retrieved_text = self._format_retrieved_knowledge(cached_items, self.args.knowledge_raw_max_chars)
                    if retrieved_text:
                        return "Knowledge Notes: %s\nRetrieved Knowledge: %s" % (
                            self._truncate_text(cached_note, self.args.knowledge_notes_max_chars),
                            self._truncate_text(retrieved_text, self.args.knowledge_raw_max_chars),
                        )
                return self._truncate_text(cached_note, self.args.knowledge_notes_max_chars)

        if mode == "legacy":
            return self._legacy_generate_knowledge(data_row, obj_list, image_path)

        retrieved_items = []
        if mode in ("raw_retrieved", "notes", "hybrid"):
            retrieved_items = self.retrieve_knowledge_notes_candidates(question, obj_list, data_row=data_row)

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
            title = str(record.get("title") or record.get("name") or record.get("key") or record.get("id") or "").strip()
            text = str(record.get("text") or record.get("contents") or record.get("description") or record.get("passage") or "").strip()
            if title and text:
                return "%s: %s" % (title, text)
            return title or text
        if isinstance(record, (list, tuple)):
            return " ".join(str(x) for x in record)
        return str(record)

    def _load_sample_knowledge_cache(self):
        if self.sample_knowledge_cache is not None:
            return self.sample_knowledge_cache
        cache = {}
        path = getattr(self.args, "knowledge_cache_file", "")
        if not path:
            self.sample_knowledge_cache = cache
            return cache
        if not os.path.isfile(path):
            print(f"[knowledge_cache] missing cache file, skip: {path}")
            self.sample_knowledge_cache = cache
            return cache

        def add_record(record):
            if not isinstance(record, dict):
                return
            keys = []
            key = str(record.get("key") or "").strip()
            if key:
                keys.append(key)
            image_id = record.get("image_id")
            question_id = record.get("question_id") or record.get("qid")
            if image_id is not None and question_id is not None:
                keys.append(f"{image_id}<->{question_id}")
            if question_id is not None:
                keys.append(str(question_id))
            for cache_key in keys:
                cache[cache_key] = record

        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            add_record(json.loads(line))
            else:
                data = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            value = dict(value)
                            value.setdefault("key", key)
                            add_record(value)
                elif isinstance(data, list):
                    for record in data:
                        add_record(record)
        except Exception as e:
            print(f"[knowledge_cache] failed to load {path}: {e}")

        print(f"[knowledge_cache] loaded {len(cache)} lookup keys from {path}")
        self.sample_knowledge_cache = cache
        return cache

    def _sample_knowledge_cache_record(self, key, data_row=None):
        cache = self._load_sample_knowledge_cache()
        if not cache:
            return None
        candidates = [str(key)]
        if "<->" in str(key):
            candidates.append(str(key).split("<->", 1)[1])
        if data_row:
            image_id = data_row.get("image_key") or data_row.get("image_id")
            question_id = data_row.get("question_id")
            if question_id is None and "<->" in str(key):
                question_id = str(key).split("<->", 1)[1]
            if image_id is not None and question_id is not None:
                candidates.append(f"{image_id}<->{question_id}")
            if question_id is not None:
                candidates.append(str(question_id))
        for candidate in candidates:
            if candidate in cache:
                return cache[candidate]
        return None

    def _knowledge_items_from_cache_record(self, record):
        if not record:
            return []
        items = record.get("selected_knowledge") or record.get("ctxs") or record.get("knowledge") or []
        if isinstance(items, str):
            items = [{"source": "knowledge_cache", "title": "", "text": items, "score": 1.0}]
        normalized = []
        for rank, item in enumerate(items, start=1):
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("contents") or item.get("passage") or "").strip()
            if not text:
                continue
            normalized.append({
                "source": item.get("source", "knowledge_cache"),
                "title": item.get("title") or item.get("id") or item.get("name") or "",
                "text": text,
                "score": float(item.get("score", 0.0) or 0.0),
                "rank": item.get("rank", rank),
            })
        return normalized

    def _selected_knowledge_sources(self):
        raw_sources = str(getattr(self.args, "knowledge_sources", "") or "").strip()
        if not raw_sources:
            return ["custom"] if getattr(self.args, "knowledge_corpus_file", "") else []
        aliases = {
            "okvqa": ["gs112k", "wikidata_kat"],
            "all": ["gs112k", "wikidata_kat", "wiki21m", "conceptnet"],
            "all_okvqa": ["gs112k", "wikidata_kat", "wiki21m", "conceptnet"],
        }
        sources = []
        for item in re.split(r"[,; ]+", raw_sources):
            item = item.strip().lower()
            if not item:
                continue
            expanded = aliases.get(item, [item])
            for source in expanded:
                if source not in sources:
                    sources.append(source)
        if getattr(self.args, "knowledge_corpus_file", "") and "custom" not in sources:
            sources.append("custom")
        return sources

    def _resolve_knowledge_source_path(self, source):
        root = getattr(self.args, "knowledge_dataset_root", "/data2/lizhengxue/datasets")
        if source == "custom":
            return getattr(self.args, "knowledge_corpus_file", "")
        if source == "gs112k":
            return getattr(self.args, "knowledge_gs112k_file", "") or os.path.join(root, "gs112k", "okvqa_full_clean_corpus.csv")
        if source == "wiki21m":
            return getattr(self.args, "knowledge_wiki21m_file", "") or os.path.join(root, "wiki21m", "psgs_w100.tsv")
        if source == "conceptnet":
            return getattr(self.args, "knowledge_conceptnet_file", "") or os.path.join(root, "conceptnet", "conceptnet-assertions-5.7.0.csv")
        if source == "wikidata_kat":
            return getattr(self.args, "knowledge_wikidata_kat_dir", "") or os.path.join(root, "wikidata_kat")
        return source

    def _open_text_auto(self, path):
        if str(path).endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        return open(path, "r", encoding="utf-8", errors="ignore")

    def _add_knowledge_record(self, corpus, source, title, text, metadata=None):
        text = str(text or "").strip()
        if not text:
            return
        record = {
            "source": source,
            "title": str(title or ""),
            "text": text,
        }
        if metadata:
            record.update(metadata)
        corpus.append(record)

    def _conceptnet_node_text(self, node):
        node = str(node)
        match = re.match(r"/c/en/([^/]+)", node)
        if not match:
            return ""
        return match.group(1).replace("_", " ")

    def _iter_knowledge_source_records(self, source, path, max_records, scan_limit):
        if source == "wikidata_kat":
            return
        if not path or not os.path.isfile(path):
            print(f"[knowledge_notes] missing {source} corpus file, skip: {path}")
            return

        loaded = 0
        scanned = 0
        try:
            if source == "gs112k":
                with self._open_text_auto(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        scanned += 1
                        text = row.get("text") or row.get("contents") or ""
                        title = row.get("kid") or row.get("id") or loaded
                        if text:
                            yield {"source": source, "title": title, "text": text}
                            loaded += 1
                        if loaded >= max_records or scanned >= scan_limit:
                            break
            elif source == "wiki21m":
                with self._open_text_auto(path) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        scanned += 1
                        text = row.get("text") or ""
                        title = row.get("title") or row.get("id") or loaded
                        if text:
                            yield {"source": source, "title": title, "text": text}
                            loaded += 1
                        if loaded >= max_records or scanned >= scan_limit:
                            break
            elif source == "conceptnet":
                with self._open_text_auto(path) as f:
                    reader = csv.reader(f, delimiter="\t")
                    for row in reader:
                        scanned += 1
                        if len(row) < 4:
                            continue
                        rel = row[1].replace("/r/", "").replace("_", " ")
                        head = self._conceptnet_node_text(row[2])
                        tail = self._conceptnet_node_text(row[3])
                        if not head or not tail:
                            if scanned >= scan_limit:
                                break
                            continue
                        text = "%s %s %s" % (head, rel, tail)
                        yield {"source": source, "title": rel, "text": text}
                        loaded += 1
                        if loaded >= max_records or scanned >= scan_limit:
                            break
            else:
                with self._open_text_auto(path) as f:
                    for line_id, line in enumerate(f):
                        scanned += 1
                        text = line.strip()
                        if text:
                            yield {"source": source, "title": str(line_id), "text": text}
                            loaded += 1
                        if loaded >= max_records or scanned >= scan_limit:
                            break
        except Exception as e:
            print(f"[knowledge_notes] failed to load {source} corpus {path}: {e}")

    def _load_custom_knowledge_corpus(self, path):
        corpus = []
        if not path:
            return corpus
        if not os.path.isfile(path):
            print(f"[knowledge_notes] missing custom corpus file, skip: {path}")
            return corpus

        def add_record(key, value):
            if isinstance(value, list):
                text = " ".join(str(x) for x in value)
            else:
                text = self._knowledge_record_text(value)
            self._add_knowledge_record(corpus, "custom", key, text)

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
                            if isinstance(item, dict):
                                item = dict(item)
                                item.setdefault("source", "custom")
                                corpus.append(item)
                            else:
                                self._add_knowledge_record(corpus, "custom", "", text)
            elif path.endswith(".json"):
                data = json.load(open(path, "r"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        add_record(key, value)
                elif isinstance(data, list):
                    for item in data:
                        text = self._knowledge_record_text(item).strip()
                        if text:
                            if isinstance(item, dict):
                                item = dict(item)
                                item.setdefault("source", "custom")
                                corpus.append(item)
                            else:
                                self._add_knowledge_record(corpus, "custom", "", text)
            else:
                with self._open_text_auto(path) as f:
                    for line_id, line in enumerate(f):
                        self._add_knowledge_record(corpus, "custom", line_id, line.strip())
        except Exception as e:
            print(f"[knowledge_notes] failed to load custom corpus {path}: {e}")
        return corpus

    def _load_external_knowledge_corpus(self):
        if self.external_knowledge_corpus is not None:
            return self.external_knowledge_corpus

        corpus = []
        source_counts = {}
        max_records = max(1, int(getattr(self.args, "knowledge_source_max_records", 50000)))
        scan_limit = max(max_records, int(getattr(self.args, "knowledge_source_scan_limit", 500000)))

        for source in self._selected_knowledge_sources():
            if source == "custom":
                records = self._load_custom_knowledge_corpus(self._resolve_knowledge_source_path(source))
            elif source == "wikidata_kat":
                records = []
                source_counts[source] = "per_sample"
                continue
            else:
                path = self._resolve_knowledge_source_path(source)
                records = list(self._iter_knowledge_source_records(source, path, max_records, scan_limit))
            corpus.extend(records)
            source_counts[source] = len(records)

        self.external_knowledge_corpus = corpus
        self.external_knowledge_index = [
            self._knowledge_tokenize(self._knowledge_record_text(record)) for record in corpus
        ]
        self.external_knowledge_source_counts = source_counts
        print(f"[knowledge_notes] loaded external corpora: sources={source_counts}, total_records={len(corpus)}")
        return corpus

    def _load_wikidata_kat_cache(self):
        if self.wikidata_kat_cache is not None:
            return self.wikidata_kat_cache
        base_dir = self._resolve_knowledge_source_path("wikidata_kat")
        cache = {
            "topentities": {},
            "ontology": {},
        }
        if not base_dir or not os.path.isdir(base_dir):
            print(f"[knowledge_notes] missing wikidata_kat directory, skip: {base_dir}")
            self.wikidata_kat_cache = cache
            return cache
        try:
            okvqa_dir = os.path.join(base_dir, "okvqa_kat", "okvqa")
            for split in ("train2014", "val2014"):
                path = os.path.join(okvqa_dir, split, f"wikidata_okvqa_{split}_topentities.pkl")
                if os.path.isfile(path):
                    cache["topentities"].update(pickle.load(open(path, "rb")))
            ontology_path = os.path.join(base_dir, "wikidata_ontology.pkl")
            if os.path.isfile(ontology_path):
                cache["ontology"] = pickle.load(open(ontology_path, "rb"))
            print(
                "[knowledge_notes] loaded wikidata_kat: topentity_images=%s ontology=%s"
                % (len(cache["topentities"]), len(cache["ontology"]))
            )
        except Exception as e:
            print(f"[knowledge_notes] failed to load wikidata_kat: {e}")
        self.wikidata_kat_cache = cache
        return cache

    def _wikidata_kat_candidates_for_sample(self, data_row, question, obj_list):
        if "wikidata_kat" not in self._selected_knowledge_sources():
            return []
        cache = self._load_wikidata_kat_cache()
        image_path = str((data_row or {}).get("image_path", ""))
        image_name = os.path.basename(image_path)
        entities_payload = cache.get("topentities", {}).get(image_name)
        if not entities_payload:
            return []
        if isinstance(entities_payload, tuple):
            entities = entities_payload[0]
        else:
            entities = entities_payload
        query_terms = self._knowledge_tokenize(question)
        for obj in obj_list:
            query_terms.update(self._knowledge_tokenize(obj))

        scored = []
        for item in entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, desc = item[0], item[1]
            else:
                title, desc = str(item), ""
            text = "%s: %s" % (title, desc)
            terms = self._knowledge_tokenize(text)
            score = len(query_terms & terms)
            scored.append((score, str(title), text))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for score, title, text in scored[: max(1, self.args.knowledge_per_source_top_k)]:
            candidates.append({"source": "wikidata_kat", "title": title, "text": text, "score": score + 100})
        return candidates

    def retrieve_knowledge_notes_candidates(self, question, obj_list, data_row=None):
        candidates = []
        seen = set()
        key = (data_row or {}).get("key", "")
        cached_record = self._sample_knowledge_cache_record(key, data_row=data_row)
        for item in self._knowledge_items_from_cache_record(cached_record):
            text = item.get("text", "")
            if text and text not in seen:
                seen.add(text)
                candidates.append(item)
        if candidates and getattr(self.args, "knowledge_cache_only", False):
            return candidates[: self.args.knowledge_top_k]

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

        for item in self._wikidata_kat_candidates_for_sample(data_row, question, obj_list):
            text = item.get("text", "")
            if text and text not in seen:
                seen.add(text)
                candidates.append(item)

        corpus = self._load_external_knowledge_corpus()
        scored_by_source = defaultdict(list)
        if corpus and self.args.knowledge_retrieval_mode in ("lexical", "hybrid"):
            for idx, record in enumerate(corpus):
                record_text = self._knowledge_record_text(record)
                terms = self.external_knowledge_index[idx] if self.external_knowledge_index else self._knowledge_tokenize(record_text)
                overlap = len(query_terms & terms)
                object_overlap = sum(1 for term in object_terms if term in terms)
                score = overlap + object_overlap * 2
                if score <= 0:
                    continue
                source = corpus[idx].get("source", "corpus") if isinstance(corpus[idx], dict) else "corpus"
                heapq.heappush(scored_by_source[source], (score, idx, record_text))
                if len(scored_by_source[source]) > max(1, self.args.knowledge_per_source_top_k):
                    heapq.heappop(scored_by_source[source])
            scored = []
            for source_items in scored_by_source.values():
                scored.extend(source_items)
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, idx, text in scored:
                if text and text not in seen:
                    seen.add(text)
                    record = corpus[idx]
                    title = record.get("title", str(idx)) if isinstance(record, dict) else str(idx)
                    source = record.get("source", "corpus") if isinstance(record, dict) else "corpus"
                    candidates.append({"source": source, "title": title, "text": text, "score": score})

        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
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
    # ------------------------------------------------------------------
    # Context selection and model backends
    # ------------------------------------------------------------------
    def pick_example(self, key):
        image_key = image_key_from_sample_key(key, self.args, getattr(self, "image_dict", None))
        scene_graph_path = os.path.join(self.dataset.sg_attr_dir, str(image_key).zfill(12) + ".json")
        scene_graph_attr = json.load(open(scene_graph_path))
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if attr['class'] in ['girl', 'boy', 'man', 'woman'] and len(attr['attr']) > 0:
                description = attr['attr'][0]
                self.temp_question = f"What is the {description} {attr['class']} doing?"
                return True
        return False
    
    def load_caption_qwen(self):
        if is_dataset(self.args, "mme"):
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

    def _call_llm_with_token_confidence(self, prompt, image_path=None, max_new_tokens=512,
                                        use_images=True, history=None):
        if self.use_vllm:
            return chat_with_qwen_vllm(
                self.vllm_client, self.vllm_model_name, prompt,
                image_path=image_path, max_new_tokens=max_new_tokens,
                use_images=use_images, history=history,
                return_token_confidence=True,
            )
        return chat_with_qwen_vl(
            self.model, self.processor, prompt,
            image_path=image_path, max_new_tokens=max_new_tokens,
            use_images=use_images, history=history,
            return_token_confidence=True,
        )

    def initialize_lang_sam(self):
        with torch.no_grad():
            self.sam = LangSAM(
                gdino_model_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base", 
                gdino_processor_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base")

    def ensure_lang_sam(self):
        if self.sam is None:
            self.initialize_lang_sam()


# Backward compatibility for scripts that import the historical lowercase name.
onion = Onion


def save_final_results(args, answers, full_answers, accuracy):
    """Write detailed and submission-style predictions for a completed run."""
    timestamp = datetime.datetime.now()
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    print(f"当前时间: {timestamp:%Y-%m-%d %H:%M:%S}")

    prompt_dir = os.path.join(args.output_path, f"prompt_answer_{date_str}")
    format_dir = os.path.join(args.output_path, f"format_answer_{date_str}")
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(format_dir, exist_ok=True)

    output_name = (
        f"VisualCOT_{args.caption_type}_n{args.n_shot}_repeat{args.n_ensemble}_"
        f"{args.similarity_metric}_{accuracy:.6f}.json"
    )
    format_predictions = []
    for answer in answers:
        prediction = {
            "answer": answer[1],
            "question_id": answer[0].split("<->")[1],
        }
        if args.chain_of_thoughts:
            prediction["thoughts"] = answer[5]
        format_predictions.append(prediction)

    with open(os.path.join(prompt_dir, output_name), "w") as f:
        json.dump(full_answers, f)
    with open(os.path.join(format_dir, output_name), "w") as f:
        json.dump(format_predictions, f)
    write_official_prediction_file(args, answers, format_dir, output_name)


# ---------------------------------------------------------------------------
# Executable entry point
# ---------------------------------------------------------------------------
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

    dataset = build_dataset(args)
    model = Onion(args, dataset=dataset)

    # 生成推理结果
    # answers是所有问题的答案列表,full_answers是包含更多信息的完整答案列表
    answers, full_answers = model.inference(save_every_step=True)

    report = direct_answer_eval_report(args, answers)
    acc = report["primary_pct"]
    for line in report["lines"]:
        print(line)

    save_final_results(args, answers, full_answers, acc)


if __name__ == "__main__":
    main()
