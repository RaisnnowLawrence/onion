# Onion Experiment Project Handoff

Last updated: 2026-06-14

This file is written for a new Codex session. Read this first when taking over the project.

## Project Goal

This repository is the unified working repo for VisualCoT / A-OKVQA experiments with local Qwen3-VL-4B.

The main research question is:

> Can VisualCoT-style iterative reasoning, image enhancement, caption/context enhancement, knowledge enhancement, MCTS/SAM, or reviewer-style CoT improve A-OKVQA accuracy over a strong direct-answer baseline?

The latest roadmap also adds a NoteMR-inspired direction: replace the current weak model-generated `knowledge_enhance` with a two-stage `retrieval -> Knowledge Notes -> final answer` pipeline, first as a hard-case / knowledge-question candidate generator.

The current empirical answer is nuanced:

- Direct no-CoT is very strong.
- Traditional step-by-step CoT hurts badly.
- Answer-first / reviewer-style CoT can recover most of the loss and sometimes slightly beat direct.
- MCTS/SAM image enhancement currently runs, but has not reliably beaten direct.
- The best current reported result is around 59.60% on A-OKVQA val.

## Main Working Directory

Use this directory as the main project root:

```text
/data2/lizhengxue/WorkSpace/onion
```

GitHub remote:

```text
git@github.com:RaisnnowLawrence/visualcot_experiment_repo.git
```

The original folders still exist, but future work should normally happen in this repo:

```text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-shell
/data2/lizhengxue/WorkSpace/onion_output
```

Do not move or delete the original `huchunning/VisualCoT-pure/forward_code` unless the user explicitly asks.

## Important Directories

```text
forward_code/
```

Current runnable code. The main entry point is:

```text
forward_code/onion.py
```

```text
ablation_summary/
```

Experiment scripts, merge/watch scripts, result CSVs, and older summary reports.

```text
docs/
```

Clean Markdown experiment documentation. Start with:

```text
docs/00_master_summary.md
docs/06_code_evolution_and_comparability.md
docs/07_reflective_oracle_analysis.md
```

```text
local_artifacts/
```

Ignored by Git. Contains large local-only copies of outputs and smoke-test artifacts.

## Environment

Use the existing conda environment, but do not modify it unless the user explicitly approves:

```text
/data2/lizhengxue/anaconda3/envs/sam/bin/python
```

Common model:

```text
qwen3-VL-4B
```

Common dataset:

```text
A-OKVQA val, 1145 samples
```

Common data root used by scripts:

```text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data
```

Typical paths:

```text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags
```

OCR note: OCR flags exist, but the expected OCR JSON files were not found locally during the latest smoke test. OCR is skipped gracefully when files are absent.

## Current Code State

Latest known pushed commit after the MCTS/SAM smoke fix:

```text
b7a96a9 Fix MCTS cache directory creation
```

The latest small code fix was in `forward_code/onion.py`: MCTS image enhancement now creates the cache directory before saving the enhanced image.

Reason: a smoke test with MCTS/SAM reached image generation, then failed because the cache directory did not exist.

## Recent Smoke Test

A one-sample smoke test was run with most modules enabled:

- VinVL caption
- scene graph / knowledge enhancement
- regional captions
- Qwen global and local caption
- reviewer evidence CoT
- direct verify policy
- MCTS
- SAM / LangSAM image enhancement

It completed successfully on GPU 3.

Output:

```text
local_artifacts/smoke_outputs/all_modules_sam_idx40/
```

Enhanced MCTS image:

```text
local_artifacts/smoke_cache/all_modules_sam_idx40/mcts_COCO_val2014_000000463522.jpg
```

Result for that sample:

```text
answer: dog
score: 1.0
```

Important conclusion: MCTS + SAM can run in the current repo. The issue found was cache directory creation, not SAM model loading itself.

## Useful Smoke Command Pattern

Use a tiny shard to process one sample:

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_visualcot_smoke \
/data2/lizhengxue/anaconda3/envs/sam/bin/python forward_code/onion.py \
  --cache_path /data2/lizhengxue/WorkSpace/onion/local_artifacts/smoke_cache/all_modules_sam_idx40 \
  --output_path /data2/lizhengxue/WorkSpace/onion/local_artifacts/smoke_outputs/all_modules_sam_idx40 \
  --caption_type vinvl \
  --n_shot 1 \
  --n_ensemble 1 \
  --rounds 1 \
  --iterative_strategy caption \
  --engine qwen3-VL-4B \
  --sg_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text \
  --train_sim_metric answer \
  --train_sim_file /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/scene_graph_text/train_object_select_answer.pk \
  --tag_path /data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/input_text/coco_caption_pred_tags \
  --context_mode no_round_state \
  --chain_of_thoughts \
  --cot_style reviewer_evidence \
  --reviewer_evidence_scope all \
  --direct_verify_policy conflict_only \
  --use_image_enhance \
  --mcts_n_simulations 1 \
  --mcts_trigger_mode all \
  --mcts_action_mode marker_only \
  --mcts_filter_objects \
  --use_caption_enhance \
  --use_knowledge_enhance \
  --use_all_regional_captions \
  --max_regional_captions 5 \
  --use_ocr_context \
  --use_qwen_blip2_caption \
  --qwen_caption_mode both \
  --qwen_caption_max_tokens 64 \
  --qwen_caption_final_max_chars 300 \
  --shard_id 40 \
  --num_shards 1145
```

Use `nvidia-smi` before launching long jobs. The user often asks to pack shards onto idle GPUs.

## Core Experimental Findings

### Best Results

Current strongest known results:

| Method | Accuracy |
|---|---:|
| `reflective_answer_first_caption_r3` | 59.60% (682/1145) |
| `answer_first_locked_no_caption` | 59.51% (681/1145) |
| `reflective_review_empty_context_caption_r3` | 59.46% (680/1145) |
| `no_cot_rounds1` | 59.24% (678/1145) |

### Main Interpretation

The best story so far:

1. Direct short answer is a very strong baseline.
2. Free-form CoT usually hurts because it invents or overweights wrong visual facts.
3. The most promising CoT structure is answer-first:
   - first answer directly,
   - then collect minimal evidence,
   - then revise only if evidence clearly contradicts the initial answer.
4. The three enhancement modules are best framed as evidence providers, not answer generators.
5. MCTS/SAM should probably provide minimal visual hints or evidence, not aggressively alter the image.

## Major Code/Method Families

### Direct / No-CoT

Usually the strongest simple baseline.

Representative method:

```text
no_cot_rounds1
```

### Traditional CoT

Older step-by-step CoT is much weaker, often around the low-to-mid 50s.

Do not assume more rounds help. More rounds often add noise.

### Answer-First CoT

Promising direction.

Representative styles:

```text
answer_first_locked
reflective_answer_first
```

The key idea is to lock the initial answer before asking for reasoning.

### Reviewer Evidence CoT

Promising and conceptually aligned with the three enhancement modules.

Representative style:

```text
reviewer_evidence
```

Interpretation:

```text
enhancement modules find evidence
reviewer CoT decides whether evidence is enough to overturn the initial answer
```

### Direct Verify

Two-stage answer verification.

Representative style:

```text
direct_verify
```

Best policy tends to be conservative:

```text
conflict_only
```

### MCTS / SAM Image Enhancement

Now runnable after the cache directory fix.

Important args:

```text
--use_image_enhance
--mcts_n_simulations
--mcts_trigger_mode
--mcts_action_mode
--mcts_filter_objects
```

MCTS has not reliably beaten direct. Narrower actions such as marker-only are better than aggressive image edits.

## Result Documentation

Primary docs:

```text
docs/00_master_summary.md
```

Code-version comparability:

```text
docs/06_code_evolution_and_comparability.md
```

Oracle analysis:

```text
docs/07_reflective_oracle_analysis.md
```

Older summary tables:

```text
ablation_summary/ablation_summary_all_final.md
ablation_summary/ablation_results_all_final.csv
```

## Important Caveat About Comparability

The experiments were not all run with exactly the same code. The code evolved while experiments were being designed:

- answer extraction changed,
- context modes were added,
- MCTS args were added,
- direct verify was added,
- reviewer evidence was added,
- answer-first and reflective CoT were added.

Therefore:

- Compare experiments inside the same method family more confidently.
- Compare across method families as development evidence, not a perfectly controlled leaderboard.
- For a paper-ready table, rerun a small set of key methods on the latest code.

## Version Control Workflow

Before significant code changes:

```bash
git -C /data2/lizhengxue/WorkSpace/onion status --short
git -C /data2/lizhengxue/WorkSpace/onion log --oneline -5
```

After a meaningful change:

```bash
git -C /data2/lizhengxue/WorkSpace/onion add <files>
git -C /data2/lizhengxue/WorkSpace/onion commit -m "<message>"
git -C /data2/lizhengxue/WorkSpace/onion push
```

Do not commit large local outputs. `local_artifacts/` is intentionally ignored.

## User Preferences

- The user usually wants concrete action, not just suggestions.
- Use Chinese when reporting progress.
- When running experiments, the user often wants shards packed onto available GPUs.
- The user cares about which code version produced which result.
- The user prefers Markdown summaries saved under the repo or `/data2/lizhengxue/WorkSpace/onion_output`.
- Be careful not to modify unrelated original project folders.

## Good Next Steps

If a new session continues this project, useful next actions are:

1. Run a slightly larger MCTS/SAM smoke, for example 5-20 samples, to check stability beyond one sample.
2. Rerun key baselines on the latest code for stricter comparison:
   - `no_cot_rounds1`
   - `answer_first_locked_no_caption`
   - `reflective_answer_first_caption_r3`
   - best reviewer evidence setting
   - best MCTS marker/narrow setting
3. Use oracle analysis to design routing:
   - when to trust direct,
   - when to trigger reviewer,
   - when to ignore caption,
   - when to use evidence modules.
4. Keep experiment summaries updated in `docs/` or `ablation_summary/`.
