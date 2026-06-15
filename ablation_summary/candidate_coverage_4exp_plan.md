# Candidate Coverage 4-Experiment Plan

Date: 2026-06-15

Goal: reduce the `gold_not_in_any_prediction` failure mode by expanding candidate coverage from four directions.

## Experiments

| GPU | Experiment | Output Directory | Main Idea |
|---:|---|---|---|
| 0 | `coverage_scan` | `qwen3-VL-4B_forward2_candidate_coverage_scan_3shards_gpu0` | Add a full-image/regional/OCR/enhanced-evidence candidate to recover missed small or background answers. |
| 1 | `count_specialist` | `qwen3-VL-4B_forward2_candidate_count_specialist_3shards_gpu1` | Add a count-specific candidate that returns number words and uses marker MCTS/enhanced image evidence. |
| 2 | `ocr_specialist` | `qwen3-VL-4B_forward2_candidate_ocr_specialist_3shards_gpu2` | Add OCR/text-specific candidate with `vinvl_ocr`, OCR context, and regional evidence. |
| 3 | `diverse_pool` | `qwen3-VL-4B_forward2_candidate_diverse_pool_3shards_gpu3` | Add a larger candidate pool: caption, count, OCR, coverage scan, and contrastive alternative; always run judge and allow new answer. |

## Code Changes

Implemented in `forward_code/onion.py`:

- `--candidate_judge_include_count_candidate`
- `--candidate_judge_include_ocr_candidate`
- `--candidate_judge_include_coverage_candidate`
- `--candidate_judge_include_contrast_candidate`

New candidate prompt styles:

- `count_specialist`
- `ocr_specialist`
- `coverage_scan`
- `contrastive`

## Launcher

Script:

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_coverage_4exp_3shards.sh
```

The current environment kills detached `nohup` GPU jobs before they write logs, so the experiments were launched in `tmux` sessions:

```text
cand_cov_gpu0
cand_count_gpu1
cand_ocr_gpu2
cand_diverse_gpu3
```

Each experiment uses `NUM_SHARDS=3`. Shards are run sequentially within each GPU session via `SHARD_PARALLEL=0`, while the four experiments run in parallel across GPUs 0-3.

## Logs

Master logs:

```text
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/candidate_coverage_scan_gpu0.master.out
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/candidate_count_specialist_gpu1.master.out
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/candidate_ocr_specialist_gpu2.master.out
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/candidate_diverse_pool_gpu3.master.out
```

Shard logs:

```text
/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/logs_candidate_coverage_4exp_3shards/
```
