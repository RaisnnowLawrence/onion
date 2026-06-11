# VisualCoT Forward Code Versioning

This repository tracks the local forward-code variant used for A-OKVQA/Qwen3-VL-4B ablation experiments.

Tracked files include:

- Python source files such as `onion.py`, `mcts.py`, `aokvqa_utils.py`, `qwen_utils.py`, and `sam_utils.py`
- Small experiment launch scripts in this directory
- `.gitignore` and versioning notes

Ignored files include:

- Runtime logs and controller output
- Python cache files
- Generated prompt/format samples
- Image/model/cache artifacts
- Large binary arrays or checkpoints

Related experiment summaries are tracked separately in:

```text
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary
```

Current notable code state:

- Added answer normalization support via `process_answer`
- Added follow-up ablation args for context modes, Qwen-caption modes, answer extraction strategies, OCR/context options, and thought verification
- Added `--mcts_n_simulations` to control MCTS image-enhancement search count
