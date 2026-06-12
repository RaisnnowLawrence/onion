# Direct-Verify / CoT Rescue Experiment Summary

Date: 2026-06-12

## Motivation

The previous no-CoT baseline is still the strongest single result:

| Method | Accuracy |
|---|---:|
| direct no-CoT, rounds=1 | 59.24% (678/1145) |

Standard CoT was much worse, so the recent code changes changed CoT from an answer-generation chain into an answer-verification chain. The intended flow is:

```text
Initial direct answer -> evidence check -> keep or revise -> final short answer
```

This makes the three enhancement modules easier to interpret:

| Module | Role in direct-verify |
|---|---|
| image / MCTS enhancement | visual evidence source |
| caption enhancement | semantic evidence source |
| knowledge enhancement | commonsense evidence source |
| direct-verify chain | evidence aggregator and conservative answer reviser |

## Completed CoT Rescue Results

| Experiment | Accuracy | Interpretation |
|---|---:|---|
| old CoT, rounds=1 | 55.33% (633/1145) | Explicit step-by-step CoT hurts direct answering |
| old CoT, rounds=5 | 52.72% (603/1145) | More rounds add noise |
| compact CoT, rounds=1 | 57.70% (660/1145) | Compact evidence cues recover part of the CoT loss |
| compact CoT + marker MCTS n=10 | 58.40% (668/1145) | Best compact-CoT result so far |
| compact CoT + outline MCTS n=10 | 58.36% (668/1145) | Similar to marker MCTS |
| compact CoT, rounds=3 | 58.10% (665/1145) | More rounds do not beat MCTS |
| compact CoT, rounds=5 | 57.38% (657/1145) | Longer interaction still degrades |
| answer-first CoT, rounds=1 | 1.23% (14/1145) | Prompt failed; model output visual cue lists as answers |

## Main Takeaways

- Compact CoT is clearly better than the old step-by-step CoT, but it still does not beat direct no-CoT.
- MCTS helps compact CoT slightly, but the best compact result is still below 59.24%.
- Multi-round CoT remains noisy. Increasing rounds from 1 to 3 or 5 does not solve the problem.
- The failed answer-first run showed that answer formatting and object-list pollution are serious risks.

## Direct-Verify Code Change

The new code path adds:

```bash
--cot_style direct_verify
```

It performs two Qwen calls per candidate:

1. Direct short answer:

```text
Answer: The answer is ...
```

2. Verification prompt:

```text
Initial Answer: ...
Evidence Check: supported / contradicted / uncertain
Evidence: ...
Final Answer:
```

The final answer extractor uses `Final Answer`. If the final answer looks like a visual cue list, the default behavior falls back to the initial direct answer.

Additional policy knobs were added for follow-up ablations:

| Argument | Meaning |
|---|---|
| `--direct_verify_policy balanced` | default conservative verifier |
| `--direct_verify_policy keep_stronger` | revise only with clear, specific contradiction |
| `--direct_verify_policy conflict_only` | keep initial unless `Evidence Check` is contradicted |
| `--direct_verify_policy revise_freely` | allow more aggressive revision |
| `--direct_verify_policy no_fallback` | disable object-list fallback via policy |
| `--disable_direct_verify_fallback` | explicit fallback disable switch |

## Running Direct-Verify Experiments

Primary direct-verify experiments currently running:

| Experiment | GPU | Shards | Purpose |
|---|---:|---:|---|
| `cot_rescue_direct_verify_rounds1_4shards` | 0 | 4 | baseline direct-verify |
| `cot_rescue_direct_verify_marker_mcts_rounds1_n10_4shards` | 1 | 4 | direct-verify + marker-only MCTS |
| `cot_rescue_direct_verify_outline_mcts_rounds1_n10_4shards` | 2 | 4 | direct-verify + outline-only MCTS |

Follow-up direct-verify experiments currently running:

| Experiment | GPU | Shards | Purpose |
|---|---:|---:|---|
| `cot_rescue_direct_verify_keep_stronger_rounds1_4shards` | 3 | 4 | test stricter keep-initial policy |
| `cot_rescue_direct_verify_conflict_only_rounds1_4shards` | 4 | 4 | revise only when contradicted |
| `cot_rescue_direct_verify_no_fallback_rounds1_4shards` | 5 | 4 | measure value of object-list fallback |
| `cot_rescue_direct_verify_qwen_caption_rounds1_4shards` | 6 | 4 | test Qwen-generated caption as verifier evidence |

## What To Watch

The most important comparisons are:

| Comparison | Question |
|---|---|
| direct no-CoT vs direct-verify | Does verification beat the strongest simple baseline? |
| direct-verify balanced vs keep-stronger | Is the verifier too willing to revise? |
| direct-verify balanced vs conflict-only | Do uncertain revisions cause harm? |
| direct-verify balanced vs no-fallback | How much does object-list fallback protect accuracy? |
| direct-verify balanced vs Qwen-caption evidence | Does better evidence help the verifier? |

The target to beat remains:

```text
direct no-CoT rounds1: 59.24% (678/1145)
```
