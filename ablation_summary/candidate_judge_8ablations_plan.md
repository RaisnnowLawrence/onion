# Candidate Judge 8-Ablation Plan

Generated: 2026-06-14

This plan explores the next direction:

```text
single-model single-answer inference
-> multi-strategy candidate generation + question-type routing + conservative evidence arbitration
```

Launch script:

```bash
ablation_summary/run_candidate_judge_8ablations_3shards.sh
```

The script uses the current unified repo code:

```text
/data2/lizhengxue/WorkSpace/onion/forward_code/onion.py
```

## Experiments

| ID | Experiment | Main Idea |
|---|---|---|
| 1 | `candidate_judge_core` | Base candidate judge: direct-context, image-only, answer-first, plus routed visual/knowledge candidates by question type. |
| 2 | `candidate_judge_always_judge` | Always run the judge even when multiple candidates agree. Tests whether judge improves or over-edits consensus. |
| 3 | `candidate_judge_caption_candidate` | Adds a separate caption-only candidate. Tests whether caption helps as a candidate source rather than final context injection. |
| 4 | `candidate_judge_strict_consensus3` | Requires 3 matching candidates to skip judge. Tests whether more conflicts should be arbitrated. |
| 5 | `candidate_judge_routed_caption_knowledge` | Enables routed caption/knowledge enhancement. Tests evidence providers as candidate support, not answer generators. |
| 6 | `candidate_judge_regions_ocr` | Adds limited regional captions and OCR context. Targets evidence coverage failures in local/text questions. |
| 7 | `candidate_judge_marker_mcts` | Adds marker-only MCTS for visual-detail/category questions and lets judge inspect enhanced image. Tests minimal visual hints. |
| 8 | `candidate_judge_allow_new_answer` | Allows judge to output a new answer outside candidates. Tests upper flexibility versus over-generation risk. |

## Recommended Reading Of Results

Do not only compare total accuracy. Also inspect per-sample overlap with:

- `no_cot_rounds1`
- `answer_first_locked_no_caption`
- `reflective_answer_first_caption_r3`
- `reflective_review_empty_context_caption_r3`
- best MCTS run

Key questions:

1. Does candidate judge rescue direct/no-CoT errors?
2. Does it damage direct/no-CoT correct samples?
3. Which candidate source contributes most often to correct final answers?
4. Which question types benefit: text/OCR, visual detail, knowledge, category, or general?
5. Does allowing new answers help or create hallucinated answers?

## Expected Best Bets

The most promising configurations are likely:

- `candidate_judge_core`
- `candidate_judge_caption_candidate`
- `candidate_judge_regions_ocr`
- `candidate_judge_marker_mcts`

The riskiest configuration is:

- `candidate_judge_allow_new_answer`

because it may undo the conservative candidate-selection constraint.

