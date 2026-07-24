"""Command-line configuration and dataset path resolution for onion experiments."""

import argparse

from dataset_utils import resolve_dataset_paths


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
    parser.add_argument('--knowledge_cache_file', type=str, default='',
                        help='optional per-question JSON/JSONL cache from offline retrieval or Knowledge Notes generation')
    parser.add_argument('--knowledge_cache_only', action='store_true',
                        help='use only --knowledge_cache_file for retrieval candidates, without online corpus scan')
    parser.add_argument('--knowledge_sources', type=str, default='',
                        help='comma-separated local knowledge sources: gs112k,wikidata_kat,wiki21m,conceptnet,custom,okvqa,all')
    parser.add_argument('--knowledge_dataset_root', type=str, default='/data2/lizhengxue/datasets',
                        help='root directory for local knowledge sources')
    parser.add_argument('--knowledge_gs112k_file', type=str, default='',
                        help='override GS112K OK-VQA corpus csv path')
    parser.add_argument('--knowledge_wiki21m_file', type=str, default='',
                        help='override Wiki21M passages tsv path')
    parser.add_argument('--knowledge_conceptnet_file', type=str, default='',
                        help='override ConceptNet assertions csv path')
    parser.add_argument('--knowledge_wikidata_kat_dir', type=str, default='',
                        help='override Wikidata-KAT root directory')
    parser.add_argument('--knowledge_source_max_records', type=int, default=50000,
                        help='max records loaded per large text knowledge source')
    parser.add_argument('--knowledge_source_scan_limit', type=int, default=500000,
                        help='max lines scanned per large text knowledge source when filtering')
    parser.add_argument('--knowledge_per_source_top_k', type=int, default=3,
                        help='max lexical retrieval candidates kept from each knowledge source before global reranking')
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
    parser.add_argument('--dyfo_dual_visual_experts', action='store_true',
                        help='use GroundingDINO/LangSAM and OWLv2 to classify confirmed and suspicious target regions')
    parser.add_argument('--dyfo_owlv2_model_path', type=str,
                        default='/data2/lizhengxue/WorkSpace/PreTrainModel/owlv2/owlv2-large-patch14-ensemble',
                        help='local OWLv2 model used by the optional second visual expert')
    parser.add_argument('--dyfo_owlv2_threshold', type=float, default=0.10,
                        help='minimum OWLv2 box confidence')
    parser.add_argument('--dyfo_dual_iou_threshold', type=float, default=0.60,
                        help='base IoU threshold for matching GroundingDINO and OWLv2 boxes')
    parser.add_argument('--dyfo_dual_iou_delta', type=float, default=0.10,
                        help='dynamic IoU adjustment used by glance-versus-stare conflict routing')
    parser.add_argument('--dyfo_dual_conflict_low', type=float, default=0.50,
                        help='conflict rate below which dual-expert matching uses a looser IoU threshold')
    parser.add_argument('--dyfo_dual_conflict_high', type=float, default=0.70,
                        help='conflict rate above which dual-expert matching uses a stricter IoU threshold')
    parser.add_argument('--dyfo_dual_max_boxes_per_target', type=int, default=3,
                        help='maximum boxes retained from each visual expert for one target')
    parser.add_argument('--dyfo_area_reward', type=str, default='compact',
                        choices=['compact', 'paper'],
                        help='kept for compatibility; DyFo reward now hard-gates target retention and uses 1 - area_ratio')
    parser.add_argument('--dyfo_text_focus_use_image', action='store_true',
                        help='let Qwen see the current crop while updating the textual focus')
    parser.add_argument('--dyfo_use_focus_image_as_answer', action='store_true',
                        help='answer on a DyFo-derived image instead of only injecting evidence')
    parser.add_argument('--dyfo_answer_image_mode', type=str, default='crop',
                        choices=['crop', 'resized_crop', 'concat_horizontal', 'concat_vertical'],
                        help='image passed to final MLLM when --dyfo_use_focus_image_as_answer is set: best crop, resized crop, or original plus resized focus crop')
    parser.add_argument('--dyfo_node_answer_image_mode', type=str, default='concat_horizontal',
                        choices=['crop', 'resized_crop', 'concat_horizontal', 'concat_vertical', 'active_look_horizontal'],
                        help='image used by each DyFo focus node when producing its local answer')
    parser.add_argument('--dyfo_decision_mode', type=str, default='evidence_inject',
                        choices=['evidence_inject', 'best_focus_answer', 'weighted_vote', 'conservative_override', 'token_confidence_override', 'node_confidence_override', 'clip_statement_override'],
                        help='DyFo final decision: inject evidence, use best focus, weighted vote, token confidence, or CLIP statement support')
    parser.add_argument('--dyfo_force_run_all_samples', action='store_true',
                        help='run DyFo for every sample even when the ONION instruction did not select image enhancement')
    parser.add_argument('--dyfo_override_confidence_threshold', type=float, default=95.0,
                        help='minimum 0-100 arbiter confidence required to replace the pure MLLM baseline')
    parser.add_argument('--dyfo_override_required_strength', type=str, default='extreme',
                        choices=['strong', 'extreme'],
                        help='minimum evidence-strength label required for conservative DyFo override')
    parser.add_argument('--dyfo_override_max_tokens', type=int, default=160,
                        help='max tokens for the conservative baseline-vs-DyFo arbiter')
    parser.add_argument('--dyfo_token_confidence_threshold', type=float, default=0.95,
                        help='minimum exp(mean(answer-token logprob)) required for a DyFo override')
    parser.add_argument('--dyfo_token_confidence_margin', type=float, default=0.0,
                        help='minimum DyFo minus pure answer-token confidence required for an override')
    parser.add_argument('--dyfo_node_confidence_threshold', type=float, default=0.80,
                        help='minimum weighted node-token confidence required to replace the pure answer')
    parser.add_argument('--dyfo_node_confidence_margin', type=float, default=0.10,
                        help='minimum weighted node confidence minus pure confidence required for replacement')
    parser.add_argument('--dyfo_node_confidence_support_ratio', type=float, default=0.60,
                        help='minimum fraction of valid node vote weight supporting the DyFo candidate')
    parser.add_argument('--dyfo_node_confidence_min_support', type=int, default=2,
                        help='minimum number of valid focus nodes supporting the DyFo candidate')
    parser.add_argument('--dyfo_clip_statement_margin', type=float, default=0.0,
                        help='minimum CLIP cosine margin favoring the DyFo statement')
    parser.add_argument('--dyfo_clip_statement_focus_gain', type=float, default=0.0,
                        help='minimum crop-vs-original gain in CLIP preference for the DyFo statement')
    parser.add_argument('--dyfo_focus_max_tokens', type=int, default=32,
                        help='max tokens for textual focus generation')
    parser.add_argument('--dyfo_answer_max_tokens', type=int, default=32,
                        help='max tokens for each DyFo focus-node local answer')
    parser.add_argument('--dyfo_evidence_max_tokens', type=int, default=96,
                        help='max tokens for final DyFo visual evidence generation')
    parser.add_argument('--dyfo_evidence_context_max_chars', type=int, default=700,
                        help='max characters of DyFo visual evidence injected into answer context')
    parser.add_argument('--dyfo_region_audit', action='store_true',
                        help='persist node-level region diagnostics and run a best-region occlusion counterfactual')
    parser.add_argument('--dyfo_region_audit_save_crops', action='store_true',
                        help='save every non-root DyFo node crop for manual region inspection')
    parser.add_argument('--dyfo_region_audit_unique_iou', type=float, default=0.90,
                        help='IoU threshold used to count near-duplicate node regions in region-audit mode')
    parser.add_argument('--dyfo_region_audit_dir', type=str, default='',
                        help='optional persistent asset directory for DyFo region-audit crops')
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
                                 'direct_rephrase_consistency', 'notemr_conservative_candidate'],
                        help='prompt style used when --chain_of_thoughts is enabled')
    parser.add_argument('--notemr_candidate_trigger', type=str, default='knowledge_qtype_or_weak',
                        choices=['always', 'never', 'knowledge_qtype', 'weak_direct', 'knowledge_qtype_or_weak'],
                        help='when --cot_style notemr_conservative_candidate should generate knowledge candidate')
    parser.add_argument('--notemr_relevance_max_tokens', type=int, default=64,
                        help='max tokens for NoteMR knowledge relevance check')
    parser.add_argument('--notemr_candidate_max_tokens', type=int, default=24,
                        help='max tokens for NoteMR knowledge candidate answer')
    parser.add_argument('--notemr_judge_max_tokens', type=int, default=96,
                        help='max tokens for NoteMR conservative final judge')
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
                        choices=['protected_reflective', 'answer_first_locked', 'complex_decompose', 'dyfo_evidence'],
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
    parser.add_argument('--multi_strategy_router_source', type=str, default='profile',
                        choices=['profile', 'mllm'],
                        help='profile uses retrieved strategy scores; mllm asks the model to choose a strategy from the question')
    # ----caption策略
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    # 数据集选择-验证测试
    parser.add_argument('--dataset_root', type=str, default='/data2/lizhengxue/datasets',
                        help='root directory containing local VQA benchmark folders')
    parser.add_argument('--dataset_name', type=str, default='aokvqa',
                        help='aokvqa, okvqa, vqav2, gqa, textvqa, infoseek, pope, mme, mme_realworld, hallusionbench, mmstar')
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
    parser.add_argument('--gqa_question_file', type=str, default='',
                        help='Optional GQA question JSON file or zip member override.')
    parser.add_argument('--mme_manifest_file', type=str, default='',
                        help='Prepared MME jsonl manifest with materialized image paths.')
    parser.add_argument('--valcaption_file', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/vinvl_caption/VinVL_base_val2014.tsv')

    args = parser.parse_args()
    args = resolve_dataset_paths(args)

    return args
