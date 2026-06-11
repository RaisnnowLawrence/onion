# VisualCoT A-OKVQA Ablation Summary

Generated: 2026-06-10

Baseline: `52.72%` (`603/1145`)

## Ranked Results

| Rank | ID | Experiment | Accuracy | Delta vs Baseline | Samples | Note |
|---:|---|---|---:|---:|---:|---|
| 1 | A2 | no_cot | 58.04 | +5.32 | 1145 | Best result; disabling CoT helps strongly. |
| 2 | A3.1 | rounds1 | 54.85 | +2.13 | 1147 | Has 2 duplicate samples; still clearly above baseline. |
| 3 | B4 | qwen_thought | 53.09 | +0.37 | 1145 | Small gain from Qwen thought verifier. |
| 4 | A3.2 | rounds3 | 53.27 | +0.55 | 1145 | Fewer rounds than baseline helps. |
| 5 | A4.1 | ensemble1 | 52.92 | +0.20 | 1145 | Slight gain; reducing ensemble is not harmful. |
| 6 | B1 | ocr | 52.90 | +0.18 | 1145 | Tiny gain. |
| 7 | A4.2 | ensemble3 | 52.86 | +0.14 | 1145 | Tiny gain. |
| 8 | B2 | clip_thought | 52.86 | +0.14 | 1145 | Tiny gain. |
| 9 | A1 | remove_caption | 52.85 | +0.13 | 1145 | Tiny gain. |
| 10 | B5 | all_regions | 52.82 | +0.10 | 1145 | Tiny gain. |
| 11 | 0 | baseline | 52.72 | +0.00 | 1145 | CoT, 5 rounds, 5 ensembles, 1-shot, VinVL caption. |
| 12 | A7.2 | caption_vinvl_sg | 52.69 | -0.03 | 1145 | Essentially tied with baseline. |
| 13 | A5.1 | nshot0 | 52.52 | -0.20 | 1145 | Slight drop. |
| 14 | A5.2 | nshot4 | 52.48 | -0.24 | 1145 | Slight drop. |
| 15 | A6 | sim_question | 52.45 | -0.27 | 1145 | Slight drop vs image+question retrieval. |
| 16 | B6 | ensemble_norm | 52.43 | -0.29 | 1145 | Slight drop. |
| 17 | A7.1 | caption_vinvl_tag | 51.71 | -1.01 | 1145 | Predicted tags hurt. |
| 18 | B3 | qwen_caption | 44.46 | -8.26 | 1145 | Qwen caption helper hurts badly. |
| 19 | B7 | all_added | 43.90 | -8.82 | 1145 | Combining added modules hurts badly, likely dominated by harmful caption helper/interactions. |

## Main Takeaways

1. The strongest result is `no_cot` at `58.04%`, which is `+5.32` over baseline. For this model/setup, explicit CoT appears to hurt A-OKVQA accuracy.
2. Reducing interaction rounds helps: `rounds1` and `rounds3` both outperform baseline. This suggests the 5-round baseline may accumulate noisy evidence or overthink.
3. Most small module changes are within a very narrow band around baseline: OCR, CLIP thought verification, remove-caption, all-regions, and ensemble changes are around `+0.10` to `+0.20`.
4. `qwen_thought` is the best added module among B-series, but the gain is still modest: `53.09%`, `+0.37`.
5. Caption changes are risky. `caption_vinvl_sg` is basically tied with baseline, `caption_vinvl_tag` drops by `-1.01`, and Qwen-generated captioning drops heavily.
6. `qwen_caption` and `all_added` are clear negative results. The all-added combination likely suffers from harmful interactions, especially from the Qwen caption helper.

## Recommended Configurations

For best accuracy from this run:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5
```

This is the `no_cot` setting.

For a CoT-style setting with better accuracy than baseline:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 3 --chain_of_thoughts
```

or, with the caveat that `rounds1` has 2 duplicate samples:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 1 --chain_of_thoughts
```

## Caveats

- `rounds1` has `1147` merged samples instead of `1145`, due to two duplicate outputs from earlier duplicate shard launches. Treat its `54.85%` as highly suggestive but not perfectly clean.
- Differences smaller than roughly `0.2-0.3` percentage points are only a few samples and should be interpreted cautiously.
- All results are for the current local Qwen3-VL-4B A-OKVQA setup and the current forward code/evaluation pipeline.
