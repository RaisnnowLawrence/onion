# VisualCoT Experiment Repository

This repository combines the local VisualCoT forward-code snapshot and the A-OKVQA ablation summaries.

It was created as a single GitHub-friendly repository without moving the original working folders.

## Layout

```text
forward_code/
ablation_summary/
```

`forward_code/` contains the local VisualCoT forward-code variant used for Qwen3-VL-4B A-OKVQA experiments.

`ablation_summary/` contains lightweight experiment summaries, CSV result tables, and launch/watcher scripts.

## Original Local Sources

The contents were copied from the following local repositories:

```text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/forward_code
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
```

The original `forward_code` folder was not moved.

## Current Headline Results

- Best configuration: `no_cot_rounds1`
- Best accuracy: `59.24%` (`678/1145`)
- MCTS n=5 image-enhancement result: `53.92%` (`617/1145`)

See:

```text
ablation_summary/ablation_summary_all_final.md
ablation_summary/ablation_results_all_final.csv
ablation_summary/mcts_round1_n5_summary.md
```

## Versioning Notes

Large runtime artifacts are intentionally ignored:

- per-sample outputs
- logs
- generated images
- model checkpoints
- cache folders
- NumPy/PyTorch binary files

This repository is meant to track code, scripts, and lightweight experiment summaries.
