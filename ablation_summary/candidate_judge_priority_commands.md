# Candidate Judge Experiment Priority Queue

Generated: 2026-06-14

Goal:

```text
single-model single-answer inference
-> multi-strategy candidate generation + question-type routing + conservative evidence arbitration
```

Launcher:

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh
```

Each experiment below runs 3 shards on one selected GPU, then merges results into `accuracy.log`.

## Priority Order

### 1. Core Candidate Judge

Highest-priority sanity and baseline for the new method.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh core <GPU>
```

### 2. Caption Candidate

Tests whether caption helps as a separate candidate source instead of being injected as dominant final context.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh caption_candidate <GPU>
```

### 3. Regions + OCR

Targets evidence coverage failures, especially local visual/text questions.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh regions_ocr <GPU>
```

### 4. Strict Consensus 3

Requires 3 matching candidates before skipping judge. Tests whether more conflicts should be arbitrated.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh strict_consensus3 <GPU>
```

### 5. Routed Caption + Knowledge

Enables question-type routed caption and knowledge enhancement.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh routed_caption_knowledge <GPU>
```

### 6. Always Judge

Always runs the judge, even when candidates agree. This tests judge value versus over-editing risk.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh always_judge <GPU>
```

### 7. Marker MCTS

Adds marker-only MCTS and lets judge inspect the enhanced image. More expensive and riskier, but useful for visual-detail questions.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh marker_mcts <GPU>
```

### 8. Allow New Answer

Allows the judge to output an answer outside the candidate set. This is the boldest and riskiest setting.

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/run_candidate_judge_8ablations_3shards.sh allow_new_answer <GPU>
```

## Auto-Scheduler

To start experiments only when a GPU has at least 40 GB free for 10 consecutive minutes:

```bash
/data2/lizhengxue/WorkSpace/onion/ablation_summary/schedule_candidate_judge_when_gpu_free.sh
```

Useful overrides:

```bash
MIN_FREE_MIB=42000 CHECK_INTERVAL=60 REQUIRED_STABLE_CHECKS=10 \
/data2/lizhengxue/WorkSpace/onion/ablation_summary/schedule_candidate_judge_when_gpu_free.sh
```

Limit to specific GPUs:

```bash
GPU_LIST=3,4,5 \
/data2/lizhengxue/WorkSpace/onion/ablation_summary/schedule_candidate_judge_when_gpu_free.sh
```

