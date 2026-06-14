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
import base64
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer
from lang_sam import LangSAM

from sam_utils import process_langsam_results_to_visualization, combine_masks_max_simple, clean_string_basic

from aokvqa_utils import aokvqa_dataset
from qwen_utils import chat_with_qwen_vl, chat_with_qwen_vllm, string_to_list_if_possible
from mcts import MCTSQuestionSample


def process_answer(answer):
    answer = str(answer).replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)


class onion:    
    def __init__(self, args, dataset):

        self.dataset = dataset
        self.args = args
        self.messages = None
        self.attention_object = []
        self.qwen_global_caption_cache = {}
        self.qwen_local_caption_cache = {}
        
        # 引擎初始化
        self.initialize_qwen(self.args.engine)

        # 图像处理部分初始化
        self.initialize_lang_sam()

        # 加载caption部分
        self.caption_qwen = self.load_caption_qwen()

        # 加载wit外部知识
        self.wit_knowkedge = self.load_wit_knowkedge()

        if args.with_clip_verify or args.choice_only or args.use_clip_thought_verify:
            model = CLIPTextModel.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            model = model.cuda()
            processor = CLIPProcessor.from_pretrained("/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-model/clip-vit-base-patch16")
            self.clip_model, self.clip_processor = model, processor

        # MCTS图像增强所需：加载完整CLIPModel（视觉+文本）用于reward计算
        self.clip_full_model = None
        self.clip_full_processor = None
        if args.use_image_enhance:
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
                                 qwen_global_caption="", qwen_local_caption=""):
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
        if self.args.cot_style in ("direct_verify", "reviewer_evidence"):
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

            caption_evidence = state.get("enhanced_caption")
            if caption_evidence:
                lines.append("Caption evidence: %s" % self._truncate_text(caption_evidence))

            knowledge_evidence = state.get("enhanced_knowledge")
            if knowledge_evidence:
                lines.append("Knowledge evidence: %s" % self._truncate_text(knowledge_evidence))

        return "\n".join(lines)

    def _make_round_state(self, round_idx, onion_instruction, enhance_image_path,
                          enhance_caption, enhance_knowledge, pred_answer,
                          final_score, pred_candidates):
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
        
        for idx, key in enumerate(tqdm(self.dataset.val_keys)):

            # 数据分片：只处理属于当前shard的样本
            if idx % self.args.num_shards != self.args.shard_id:
                continue

            print('----------inference----------processing sample %s/%s----------for loop----------' % (str(idx), str(len(self.dataset.val_keys))))

            # 如果已保存该样本结果则跳过
            if save_every_step:
                # 这里没有修改关于时间戳的内容
                out_file_name = "%s/prompt_samples/sample_%s_*.json" % (self.args.output_path, str(idx))
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
                json.dump(answers[-1], open("%s/prompt_samples/sample_%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(float(answers[-1][3]))), 'w'))
                json.dump(full_answers[-1], open("%s/format_samples/sample_%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(float(answers[-1][3]))), 'w'))
            
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
        scene_graph_attr = json.load(open(scene_graph_path))
        
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
                if attr['class'] in self.val_ocr_text[image_key]:
                    tmp_attr.append(self.val_ocr_text[image_key][attr['class']])
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
                noticed_caption_list.append(self.caption_qwen[str(image_key)])

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
        caption = self.dataset.inputtext_dict[image_key][0]
        # caption += ' '
        # print(type(caption))
        # print(type(self.caption_qwen[str(image_key)]))
        qwen_caption = self.caption_qwen[str(image_key)]
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
        selective_mode = self.args.cot_style == "reviewer_evidence" and self.args.reviewer_evidence_scope == "selective"
        
        # ========== 三个核心增强模块（由args控制开关） ==========
        if effective_use_image_enhance and onion_instruction[0] == 'image':
            enhance_image_path = self.enhance_image_object(data_row, onion_instruction[1], attr_list)
            print('-----enhance_image-----MCTS增强图像已生成-----')

        if effective_use_caption_enhance and (onion_instruction[0] == 'caption' or selective_mode):
            enhance_caption = self.enhance_caption_object(data_row, onion_instruction[1], attr_list)
            print('-----enhance_caption-----强化的针对目标描述-----+++++-----beg')
            print('enhance_caption:', enhance_caption)
            print('-----enhance_caption-----强化的针对目标描述-----+++++-----end')
            print()

        if effective_use_knowledge_enhance and (onion_instruction[0] == 'knowledge' or selective_mode):
            enhance_knowledge = self.enhance_knowledge_object(data_row, onion_instruction[1], attr_list)
            print('-----enhance_knowledge-----强化的针对目标知识-----+++++-----beg')
            print('enhance_knowledge:', enhance_knowledge)
            print('-----enhance_knowledge-----强化的针对目标知识-----+++++-----end')
            print()

        print('-----onion_instruction-----类别输出指示-----+++++-----beg')
        print('onion_instruction:', onion_instruction)
        if effective_use_caption_enhance and (onion_instruction[0] == 'caption' or selective_mode):
            print('enhance_caption:', enhance_caption)
        if effective_use_knowledge_enhance and (onion_instruction[0] == 'knowledge' or selective_mode):
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
                prompt += self._format_cot_answer_prompt(prompt_before_answer)
            else:
                prompt += '=== Please fill in the answer with a short phrase or a single word:\n'
                prompt += '%s' % (prompt_before_answer)

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
                    if self.args.cot_style == "direct_verify":
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
            # ====================== 评分方式1：自定义分档（当前使用） ======================
            counter = 0
            processed_pred_answer = process_answer(pred_answer)
            for ii in range(len(answer)):
                if processed_pred_answer == process_answer(answer[ii]): counter += 1
            final_score = min(1., float(counter) * 0.3)
            # # ====================== 评分方式2：AOK-VQA标准评分（精确匹配任一标注者即得1分） ======================
            # final_score = 1.0 if any(pred_answer == ans for ans in answer) else 0.0
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
            pred_answer=pred_answer,
            final_score=final_score,
            pred_candidates=pred_candidates
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

        # 如果是补充list，则补充并且去重
        _, selected_objects = self.init_attention_object(key, attr_list, image_path, ban_option=[])
        obj_list = list(dict.fromkeys(obj_list + selected_objects))
        # 在多轮活动中交叠补充
        self.attention_object = list(dict.fromkeys(self.attention_object + obj_list))
        obj_list = list(dict.fromkeys(self.attention_object + obj_list))

        # # wit查询相关知识
        # knowledge = ''
        # for obj in obj_list:
        #     knowledge += obj + ': '  # 添加标题分隔符
        #     if obj in self.wit_knowkedge:
        #         # 将列表中的多个描述合并成一个字符串
        #         descriptions = ' '.join(self.wit_knowkedge[obj])  # 用空格连接多个描述
        #         knowledge += descriptions
        #     knowledge += '\n'

        # 模型补充知识
        prompt = 'I am giving you a question, an image, and some supplementary information, but you do not need to answer it.\n'
        prompt += 'Please supplement additional knowledge about the specified target based on the question and image I provide, rather than information already present in the image.\n'
        prompt += 'Object: %s\n' % str(obj_list)

        response = self._call_llm(prompt, image_path=image_path)
        knowledge = response

        return knowledge

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
        file_path = '/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/caption_onion/aokvqa_val_caption_8b_256.json'
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
    # 汇总模式：不推理，只读取prompt_samples目录计算全量准确率
    parser.add_argument('--merge_only', action='store_true', help="merge shard results and compute accuracy")
    parser.add_argument('--summary_log', type=str, default='', help="path to write accuracy summary line")
    # 实验类型-模型结构
    parser.add_argument('--choice_only', action='store_true')
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
    parser.add_argument('--mcts_n_simulations', type=int, default=20, help="number of MCTS simulations for image enhancement")
    parser.add_argument('--mcts_trigger_mode', type=str, default='all',
                        choices=['all', 'visual_detail_only', 'count_color_object_only'],
                        help='controls which questions can trigger MCTS image enhancement')
    parser.add_argument('--mcts_action_mode', type=str, default='all',
                        choices=['all', 'outline_only', 'marker_only', 'no_crop'],
                        help='controls the MCTS image operation set')
    parser.add_argument('--mcts_filter_objects', action='store_true',
                        help='filter generic MCTS key objects and align them to selected scene-graph objects')
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
                                 'reflective_answer_first', 'adaptive_reflective_answer_first'],
                        help='prompt style used when --chain_of_thoughts is enabled')
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
    # ----caption策略
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    # 数据集选择-验证测试
    parser.add_argument('--dataset_name', type=str, default='aokvqa', help='aokvqa, okvqa')
    parser.add_argument('--split_name', type=str, default='val', help='train, val, test')
    # 描述文本选择
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg, vinvl_ocr')
    # 路径相关
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--cache_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/cache')
    # 不确定要不要修改的路径
    parser.add_argument('--raw_image_dir', type=str, default="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco17")
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--concept_caption_path', type=str, default='scene_graph_coco17_caption')
    parser.add_argument('--sg_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text')
    parser.add_argument('--similarity_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_clip_new')
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--train_sim_metric', type=str, default='rationale')
    parser.add_argument('--train_sim_file', type=str, default='')
    parser.add_argument('--val_sim_file', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_annotations')
    parser.add_argument('--valcaption_file', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/vinvl_caption/VinVL_base_val2014.tsv')

    args = parser.parse_args()

    return args


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

    # 计算全量准确率
    acc = sum(float(a[3]) for a in answers)
    total = len(answers)
    acc_pct = acc * 100.0 / total

    print(f"\n{'='*50}")
    print(f"全量准确率: {acc_pct:.2f}% ({int(acc)}/{total})")
    print(f"{'='*50}\n")

    # 如果指定了summary_log，将准确率写入汇总日志
    if args.summary_log:
        with open(args.summary_log, 'a') as f:
            f.write(f"全量准确率: {acc_pct:.2f}% ({int(acc)}/{total})\n")

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
    aokvqa_data = aokvqa_dataset(args)

    aokvqa_onion = onion(args, dataset=aokvqa_data)

    # 生成推理结果
    # answers是所有问题的答案列表,full_answers是包含更多信息的完整答案列表
    answers, full_answers = aokvqa_onion.inference(save_every_step = True)

    prediction = {}
    acc = 0.
    for answer in answers:
        prediction[answer[0]] = [answer[1], answer[2]]
        acc += float(answer[3])

    format_prediction = []
    for answer in answers:
        if args.chain_of_thoughts:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1],
                                      "thoughts": answer[5]})
        else:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1]})

    print("acc:", acc * 100. / len(answers), len(answers))
    acc = acc * 100. / len(answers)

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

if __name__ == '__main__':
    main()
