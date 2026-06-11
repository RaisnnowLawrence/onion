# VisualCoT A-OKVQA Ablation Summary

Generated: 2026-06-11

Baseline: `52.72%` (`603/1145`)

This summary combines the first ablation batch and the follow-up 2 batch. The follow-up 2 results are taken from the final `accuracy.log` files, because several manually resumed experiments were merged after `followup2_results.csv` was first written.

## Overall Ranking

| Rank | Group | ID | Experiment | Accuracy | Delta | Score | Main Change |
|---:|---|---|---|---:|---:|---:|---|
| 1 | followup2 | NC1 | no_cot_rounds1 | 59.24 | +6.52 | 678/1145 | no CoT, 1 round |
| 2 | followup2 | NC3 | no_cot_rounds3 | 58.18 | +5.46 | 666/1145 | no CoT, 3 rounds |
| 3 | followup2 | NE1 | no_cot_ensemble1 | 58.18 | +5.46 | 666/1145 | no CoT, 1 ensemble |
| 4 | main | A2 | no_cot | 58.04 | +5.32 | 664/1145 | remove CoT |
| 5 | followup2 | NE3 | no_cot_ensemble3 | 57.36 | +4.64 | 656/1145 | no CoT, 3 ensembles |
| 6 | followup2 | CTX0 | context_empty | 55.90 | +3.18 | 640/1145 | empty context, CoT |
| 7 | followup2 | CTXN | context_no_round_state | 55.48 | +2.76 | 635/1145 | remove round state, CoT |
| 8 | followup2 | C1 | clean_rounds1 | 55.33 | +2.61 | 633/1145 | clean 1-round CoT |
| 9 | followup2 | CTXC | context_caption_only | 55.17 | +2.45 | 631/1145 | caption-only context, CoT |
| 10 | followup2 | CTXO | context_objects_only | 55.10 | +2.38 | 630/1145 | object-only context, CoT |
| 11 | main | A3.1 | rounds1 | 54.85 | +2.13 | 629/1147 | old 1-round CoT run |
| 12 | main | A3.2 | rounds3 | 53.27 | +0.55 | 609/1145 | 3-round CoT |
| 13 | main | B4 | qwen_thought | 53.09 | +0.37 | 608/1145 | Qwen thought verifier |
| 14 | main | A4.1 | ensemble1 | 52.92 | +0.20 | 606/1145 | 1 ensemble, CoT |
| 15 | main | B1 | ocr | 52.90 | +0.18 | 606/1145 | OCR context |
| 16 | main | A4.2 | ensemble3 | 52.86 | +0.14 | 605/1145 | 3 ensembles, CoT |
| 17 | main | B2 | clip_thought | 52.86 | +0.14 | 605/1145 | CLIP thought verifier |
| 18 | main | A1 | remove_caption | 52.85 | +0.13 | 605/1145 | remove caption |
| 19 | main | B5 | all_regions | 52.82 | +0.10 | 605/1145 | all regional captions |
| 20 | main | 0 | baseline | 52.72 | +0.00 | 603/1145 | 5-round CoT baseline |
| 21 | main | A7.2 | caption_vinvl_sg | 52.69 | -0.03 | 603/1145 | scene-graph caption |
| 22 | followup2 | QNF | qwen_caption_no_final | 52.66 | -0.06 | 602/1145 | query Qwen caption but do not inject |
| 23 | main | A5.1 | nshot0 | 52.52 | -0.20 | 601/1145 | 0-shot |
| 24 | main | A5.2 | nshot4 | 52.48 | -0.24 | 601/1145 | 4-shot |
| 25 | main | A6 | sim_question | 52.45 | -0.27 | 601/1145 | question-only retrieval |
| 26 | main | B6 | ensemble_norm | 52.43 | -0.29 | 600/1145 | normalized voting |
| 27 | followup2 | AS | answer_strict_final | 52.43 | -0.29 | 600/1145 | strict final-answer extraction |
| 28 | followup2 | AR | answer_raw | 52.37 | -0.35 | 599/1145 | raw answer extraction |
| 29 | followup2 | AL | answer_last_line | 52.25 | -0.47 | 598/1145 | last-line answer extraction |
| 30 | main | A7.1 | caption_vinvl_tag | 51.71 | -1.01 | 592/1145 | predicted tags caption |
| 31 | followup2 | QL | qwen_caption_local | 45.23 | -7.49 | 517/1145 | local Qwen caption |
| 32 | main | B3 | qwen_caption | 44.46 | -8.26 | 509/1145 | Qwen caption helper |
| 33 | followup2 | QG | qwen_caption_global | 44.06 | -8.66 | 504/1145 | global Qwen caption |
| 34 | main | B7 | all_added | 43.90 | -8.82 | 503/1145 | all added modules |
| 35 | followup2 | QS | qwen_caption_short | 43.87 | -8.85 | 502/1145 | short Qwen caption |

## Main Conclusions

### 1. Removing CoT is the clearest win

The strongest family is no-CoT:

| Experiment | Accuracy | Delta |
|---|---:|---:|
| no_cot_rounds1 | 59.24 | +6.52 |
| no_cot_rounds3 | 58.18 | +5.46 |
| no_cot_ensemble1 | 58.18 | +5.46 |
| no_cot | 58.04 | +5.32 |
| no_cot_ensemble3 | 57.36 | +4.64 |

This is the most important result. In this Qwen3-VL-4B A-OKVQA setup, explicit chain-of-thought prompting consistently hurts compared with short-answer direct inference.

### 2. If CoT is kept, fewer/noisier context paths should be removed

CoT can be improved by simplifying the context or interaction loop:

| Experiment | Accuracy | Delta |
|---|---:|---:|
| context_empty | 55.90 | +3.18 |
| context_no_round_state | 55.48 | +2.76 |
| clean_rounds1 | 55.33 | +2.61 |
| context_caption_only | 55.17 | +2.45 |
| context_objects_only | 55.10 | +2.38 |

The pattern suggests that the baseline 5-round CoT pipeline accumulates noisy intermediate state. The previous-round state and/or long context appears more harmful than helpful.

### 3. Qwen-generated captions are strongly harmful when injected into the final prompt

| Experiment | Accuracy | Delta |
|---|---:|---:|
| qwen_caption_no_final | 52.66 | -0.06 |
| qwen_caption_local | 45.23 | -7.49 |
| qwen_caption | 44.46 | -8.26 |
| qwen_caption_global | 44.06 | -8.66 |
| qwen_caption_short | 43.87 | -8.85 |

This is very clear: merely querying Qwen captions is not the problem, because `qwen_caption_no_final` is basically tied with baseline. The accuracy collapses when those generated captions are injected into the answer prompt. The likely failure mode is that generated captions introduce misleading or over-specified visual claims.

### 4. Answer extraction is not the bottleneck

| Experiment | Accuracy | Delta |
|---|---:|---:|
| answer_strict_final | 52.43 | -0.29 |
| answer_raw | 52.37 | -0.35 |
| answer_last_line | 52.25 | -0.47 |

Changing answer extraction strategy does not recover the CoT loss. The weakness is upstream in the prompting/reasoning/context, not mainly in post-processing.

### 5. Most small add-ons are near noise level

OCR, CLIP thought verification, all-regions, ensemble changes, and caption removal are all within about `+0.1` to `+0.2` over baseline. These may not be meaningful without repeated runs.

`qwen_thought` is the best small add-on at `53.09%`, but the gain is only `+0.37`.

### 6. MCTS image enhancement is not helpful for the best no-CoT pipeline

The MCTS image-enhancement experiment was run after the main table:

| Experiment | Accuracy | Delta vs Baseline | Delta vs no_cot_rounds1 |
|---|---:|---:|---:|
| mcts_image_no_cot_rounds1_n5_4shards | 53.92 | +1.20 | -5.32 |

This setting uses `--use_image_enhance --mcts_n_simulations 5` on top of `no_cot_rounds1`. It is above the original CoT baseline, but far below the direct no-CoT result. For the current pipeline, MCTS/SAM image enhancement should not be enabled by default.

## Recommended Configurations

Best overall:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 1
```

This corresponds to `no_cot_rounds1`, with `59.24%`.

Cheaper no-CoT configuration:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 1 --rounds 5
```

This corresponds to `no_cot_ensemble1`, with `58.18%`, only `-1.06` below the best result and much cheaper than 5 ensembles.

Best CoT-style configuration:

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --context_mode empty --chain_of_thoughts
```

This corresponds to `context_empty`, with `55.90%`.

Avoid:

```bash
--use_qwen_blip2_caption
```

unless `--qwen_caption_no_final_context` is also enabled or the caption is used only for analysis outside the final answer prompt.

## Paper/Report Talking Points

1. Direct short-answer prompting is substantially stronger than CoT for Qwen3-VL-4B on this A-OKVQA setup.
2. Multi-round CoT appears to accumulate noise. Simplifying the context improves CoT but still does not match direct no-CoT inference.
3. Generated visual captions are dangerous when fed back into the final answer prompt. This effect is robust across global, local, and shortened Qwen-caption variants.
4. Post-processing answer extraction has limited effect, indicating the main failure is not answer parsing.
5. Efficiency and accuracy align: no-CoT with fewer ensembles remains high-performing.
6. MCTS image enhancement with `n_simulations=5` improves over the original CoT baseline but substantially underperforms the best no-CoT configuration.

## Files

- Full combined CSV: `ablation_results_all_final.csv`
- MCTS n=5 summary: `mcts_round1_n5_summary.md`
- First-batch summary: `ablation_summary_final.md`
- Follow-up controller/report files: `followup2_report.md`, `followup2_results.csv`

## Caveats

- `rounds1` from the first batch has `1147` samples due to duplicate shard output. Use `clean_rounds1` as the clean CoT 1-round result.
- Differences below about `0.2-0.3` percentage points correspond to only a few samples and should be interpreted cautiously.
- These conclusions are specific to the current local Qwen3-VL-4B, A-OKVQA split, and forward-code evaluation pipeline.

---

# 中文对照版

生成时间：2026-06-11

基线结果：`52.72%`，即 `603/1145`。

这份总结合并了第一批消融实验和 follow-up 2 消融实验。follow-up 2 中有几组实验是后续手动补跑并由 watcher 合并的，因此这里的最终结果以各实验目录下最终生成的 `accuracy.log` 为准，而不是只看最早写出的 `followup2_results.csv`。

## 总体结论

### 1. 去掉 CoT 是最明确、最稳定的提升

表现最好的实验几乎都属于 no-CoT 系列：

| 实验 | 准确率 | 相对基线提升 |
|---|---:|---:|
| no_cot_rounds1 | 59.24 | +6.52 |
| no_cot_rounds3 | 58.18 | +5.46 |
| no_cot_ensemble1 | 58.18 | +5.46 |
| no_cot | 58.04 | +5.32 |
| no_cot_ensemble3 | 57.36 | +4.64 |

这是最重要的发现：在当前 Qwen3-VL-4B + A-OKVQA 的设置下，显式的 chain-of-thought 推理提示并没有帮助模型，反而明显降低了最终准确率。直接让模型输出短答案更强。

### 2. 如果保留 CoT，需要减少多轮上下文带来的噪声

在 CoT 系列里，简化上下文或减少多轮状态可以提升效果：

| 实验 | 准确率 | 相对基线提升 |
|---|---:|---:|
| context_empty | 55.90 | +3.18 |
| context_no_round_state | 55.48 | +2.76 |
| clean_rounds1 | 55.33 | +2.61 |
| context_caption_only | 55.17 | +2.45 |
| context_objects_only | 55.10 | +2.38 |

这说明 baseline 的 5 轮 CoT 流程可能在逐轮积累噪声。前一轮的状态、推理痕迹或过长的上下文并没有稳定提供帮助，反而可能干扰模型回答。

### 3. Qwen 生成 caption 一旦注入最终回答 prompt，会显著伤害结果

| 实验 | 准确率 | 相对基线变化 |
|---|---:|---:|
| qwen_caption_no_final | 52.66 | -0.06 |
| qwen_caption_local | 45.23 | -7.49 |
| qwen_caption | 44.46 | -8.26 |
| qwen_caption_global | 44.06 | -8.66 |
| qwen_caption_short | 43.87 | -8.85 |

这个结论非常清楚：只调用 Qwen 生成 caption 本身不是主要问题，因为 `qwen_caption_no_final` 基本和 baseline 持平。真正的问题是把 Qwen 生成的 caption 注入最终回答 prompt 后，准确率会大幅下降。可能原因是生成的 caption 会引入错误细节、过度描述，或者让模型过度依赖文本而忽略图像。

### 4. answer extraction 不是主要瓶颈

| 实验 | 准确率 | 相对基线变化 |
|---|---:|---:|
| answer_strict_final | 52.43 | -0.29 |
| answer_raw | 52.37 | -0.35 |
| answer_last_line | 52.25 | -0.47 |

严格抽取 `Final Answer`、取最后一行、或者直接使用原始回答，都没有明显改善结果。因此当前 CoT 的问题主要不在答案后处理，而是在前面的 prompt、上下文构造和推理过程。

### 5. 大多数小模块影响很小，可能接近统计波动

OCR、CLIP thought verification、all-regions、ensemble 数量变化、remove caption 等实验大多只比 baseline 高 `+0.1` 到 `+0.2` 左右。这种幅度只对应几个样本，最好谨慎解读。

其中 `qwen_thought` 是第一批新增模块里最好的，准确率为 `53.09%`，比 baseline 高 `+0.37`，但提升仍然比较有限。

### 6. MCTS 图像增强不适合当前最优 no-CoT 主线

后续补跑了一个 MCTS 图像增强实验：

| 实验 | 准确率 | 相对 baseline | 相对 no_cot_rounds1 |
|---|---:|---:|---:|
| mcts_image_no_cot_rounds1_n5_4shards | 53.92 | +1.20 | -5.32 |

这个实验是在 `no_cot_rounds1` 的基础上加入：

```bash
--use_image_enhance --mcts_n_simulations 5
```

结果比原始 CoT baseline 高，但比最好的 `no_cot_rounds1` 低很多。因此在当前 pipeline 里，MCTS/SAM 图像增强不适合作为默认开启模块。它可能会裁剪或突出错误区域，导致模型丢失 A-OKVQA 所需的全局上下文、文字信息或常识线索。

## 推荐配置

### 最佳整体配置

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 1
```

对应实验：`no_cot_rounds1`。  
结果：`59.24%`，是目前所有实验中最好的。

### 更省算力的 no-CoT 配置

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 1 --rounds 5
```

对应实验：`no_cot_ensemble1`。  
结果：`58.18%`，只比最佳结果低 `1.06` 个百分点，但 ensemble 数量更少，推理成本更低。

### 如果必须保留 CoT，推荐的 CoT 配置

```bash
--caption_type vinvl --n_shot 1 --n_ensemble 5 --rounds 5 --context_mode empty --chain_of_thoughts
```

对应实验：`context_empty`。  
结果：`55.90%`，是 CoT 类实验中表现最好的配置。

### 不推荐开启

```bash
--use_qwen_blip2_caption
```

除非同时使用：

```bash
--qwen_caption_no_final_context
```

或者只把 Qwen caption 用于离线分析，而不要注入最终回答 prompt。

## 适合写进报告/论文的表述

1. 在当前 Qwen3-VL-4B + A-OKVQA 设置下，直接短答案推理显著优于显式 CoT 推理。
2. 多轮 CoT 会积累噪声；简化上下文后 CoT 能提升，但仍不如 no-CoT。
3. 将模型生成的视觉 caption 反馈进最终回答 prompt 会显著损害性能，这一现象在 global、local、short caption 设置下都成立。
4. answer extraction 的影响有限，说明主要问题不是答案解析，而是上游的 prompt 和上下文。
5. 准确率和效率在 no-CoT 设置下是一致的：减少 CoT 和 ensemble 不仅更快，而且效果更好或接近最好。
6. MCTS 图像增强在 `n_simulations=5` 时比原始 CoT baseline 略高，但明显弱于最佳 no-CoT 配置，不建议默认使用。

## 结果文件

- 完整合并 CSV：`ablation_results_all_final.csv`
- MCTS n=5 总结：`mcts_round1_n5_summary.md`
- 第一批英文总结：`ablation_summary_final.md`
- follow-up 2 原始报告：`followup2_report.md`
- follow-up 2 原始 CSV：`followup2_results.csv`

## 注意事项

- 第一批的 `rounds1` 有 `1147` 个样本，因为早期重复启动 shard 产生了 2 个重复输出。更推荐引用干净重跑的 `clean_rounds1`。
- 小于 `0.2-0.3` 个百分点的差异只对应少数几个样本，不能过度解读。
- 所有结论都限定在当前本地 Qwen3-VL-4B、A-OKVQA split 和当前 forward-code 评价流程下。
