# DyFo-Style Visual Evidence

This note documents the DyFo-inspired path added to onion.

## Motivation

The original onion MCTS path mainly edits the image with crop, outline, marker, or mask operations. DyFo suggests a cleaner framing:

- maintain a focus state with image region plus textual focus;
- use two conservative actions, semantic focus and semantic scatter;
- reward agreement between the LMM textual focus and the visual expert localization;
- support either visual-evidence injection or DyFo-native node answers with reward-weighted voting.

## New Runtime Path

Enable it with:

```bash
--use_image_enhance \
--mcts_action_mode dyfo_evidence \
--use_dyfo_visual_evidence
```

The new path:

1. routes only visual-detail-like questions by default;
2. asks Qwen for a short textual focus cue;
3. runs a small focus tree with `semantic_focus` and `semantic_scatter`;
4. localizes each focus with LangSAM;
5. checks crop/focus consistency with Qwen;
6. optionally asks every focus node to answer from its own crop;
7. chooses the highest-reward focus region;
8. either injects short visual evidence into the normal answer path, answers from the best focus node, or performs reward-weighted voting across focus nodes.

By default the final answer still uses the original image plus DyFo evidence:

```bash
--dyfo_decision_mode evidence_inject
```

Two more DyFo-native decision modes are available:

- `--dyfo_decision_mode best_focus_answer`: answer from the highest-reward focus node.
- `--dyfo_decision_mode weighted_vote`: answer from all focus nodes and vote after official VQA answer normalization, weighted by node reward/value.

The older image-routing switch is still available:

```bash
--dyfo_use_focus_image_as_answer
```

It sends the normal final-answer prompt to the best DyFo crop, but it does not do node-level voting.

## Important Args

- `--dyfo_trigger_mode visual_detail`: trigger on visual detail, OCR, category, count, and color questions.
- `--dyfo_n_simulations 6`: number of focus-tree simulations.
- `--dyfo_max_depth 3`: maximum focus-tree depth.
- `--dyfo_area_reward compact`: reward consistent and compact regions.
- `--dyfo_text_focus_use_image`: let Qwen see the current crop while refining textual focus.
- `--dyfo_decision_mode evidence_inject`: DyFo final decision mode. Options: `evidence_inject`, `best_focus_answer`, `weighted_vote`.
- `--dyfo_answer_max_tokens 32`: max tokens for each focus-node answer.
- `--dyfo_evidence_context_max_chars 700`: max DyFo evidence injected into answer context.

## Suggested Ablations

1. Direct baseline.
2. DyFo evidence only:

```bash
--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence
```

3. DyFo evidence plus reviewer:

```bash
--chain_of_thoughts --cot_style reviewer_evidence \
--use_image_enhance --mcts_action_mode dyfo_evidence --use_dyfo_visual_evidence
```

4. DyFo best focus-node answer:

```bash
--use_image_enhance --mcts_action_mode dyfo_evidence \
--use_dyfo_visual_evidence --dyfo_decision_mode best_focus_answer
```

5. DyFo multi-grained weighted voting:

```bash
--use_image_enhance --mcts_action_mode dyfo_evidence \
--use_dyfo_visual_evidence --dyfo_decision_mode weighted_vote
```

6. More DyFo search:

```bash
--use_image_enhance --mcts_action_mode dyfo_evidence \
--use_dyfo_visual_evidence --dyfo_n_simulations 10 --dyfo_max_depth 4
```
