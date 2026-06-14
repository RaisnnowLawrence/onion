# VisualCoT Experiment Repo Manifest

Last updated: 2026-06-14

This repository is now the unified local project folder for the VisualCoT experiment work.

## Tracked Project Files

These files are intended to be version-controlled and pushed to GitHub:

| Path | Content |
|---|---|
| `forward_code/` | Current runnable forward code, including `onion.py`, MCTS, Qwen helpers, and A-OKVQA utilities. |
| `ablation_summary/` | Experiment launch scripts, analysis scripts, CSV summaries, and lightweight reports. |
| `docs/` | Clean Markdown documentation copied from `onion_output/md_docs`. Start with `docs/00_master_summary.md`. |
| `README.md` | Repository-level notes. |
| `ARTIFACTS_MANIFEST.md` | This file. |

## Local-Only Artifact Archive

The directory below is intentionally ignored by Git:

```text
local_artifacts/
```

It keeps a local copy of useful large artifacts without pushing multi-GB outputs to GitHub.

| Path | Source | Content |
|---|---|---|
| `local_artifacts/VisualCoT-pure/` | `/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/` | Snapshot of the working code tree, logs, and knowledge file from the original workspace. |
| `local_artifacts/VisualCoT-shell/` | `/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-shell/` | Original shell scripts. |
| `local_artifacts/onion_output_light/` | `/data2/lizhengxue/WorkSpace/onion_output/` | Lightweight experiment output archive. |

## Omitted Intermediate Outputs

The lightweight `onion_output` archive intentionally excludes per-question intermediate output folders:

```text
prompt_samples/
format_samples/
prompt_answer_*/
format_answer_*/
```

Reason: these folders contain hundreds of thousands of small JSON files. They are useful for deep debugging, but slow to copy and are not needed for the main reports or final accuracy summaries.

The original source directory was not deleted:

```text
/data2/lizhengxue/WorkSpace/onion_output
```

If deep per-question debugging is needed later, use the original source directory or selectively copy only the relevant experiment.

## Current Best Result

As of this snapshot:

| Method | Accuracy |
|---|---:|
| `reflective_answer_first_caption_r3` | `59.60% (682/1145)` |
| `answer_first_locked_no_caption` | `59.51% (681/1145)` |
| `reflective_review_empty_context_caption_r3` | `59.46% (680/1145)` |

See `docs/00_master_summary.md` and `docs/07_reflective_oracle_analysis.md` for the full story.

## Re-running Experiments

Use the current code in:

```text
forward_code/onion.py
```

Useful launch scripts are in:

```text
ablation_summary/
```

The local environment used throughout these experiments was:

```text
/data2/lizhengxue/anaconda3/envs/sam/bin/python
```
