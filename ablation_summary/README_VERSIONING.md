# VisualCoT Ablation Summary Versioning

This repository tracks lightweight experiment metadata and final summaries for the local VisualCoT A-OKVQA/Qwen3-VL-4B runs.

Tracked files include:

- Final result tables such as `ablation_results_all_final.csv`
- Human-readable summaries such as `ablation_summary_all_final.md`
- Experiment launch and watcher scripts
- Small metadata files such as `followup2_experiments.tsv`

Ignored files include:

- Runtime logs and `.out` files
- Per-shard log directories
- Python caches
- Generated images, samples, checkpoints, and other large artifacts

The paired code repository is:

```text
/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/forward_code
```

Current headline result:

- Best configuration: `no_cot_rounds1`
- Accuracy: `59.24%` (`678/1145`)
- MCTS n=5 image-enhancement result: `53.92%` (`617/1145`)
