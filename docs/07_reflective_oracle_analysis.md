# Reflective R3 逐题 Oracle 分析

生成时间：2026-06-14

目标：找出 `reflective_answer_first_caption_r3` 错、但其他强方法能做对或得分更高的题。

## 参与对比的方法

| 方法名 | 实验目录 |
|---|---|
| `r3_caption` | `qwen3-VL-4B_forward2_reflective_answer_first_caption_r3_rounds1_3shards_gpu4` |
| `answer_first_locked_no_caption` | `qwen3-VL-4B_forward2_answer_first_locked_no_caption_rounds1_3shards_gpu6` |
| `review_empty_context` | `qwen3-VL-4B_forward2_reflective_review_empty_context_caption_r3_3shards_gpu6` |
| `no_cot_rounds1` | `qwen3-VL-4B_forward2_no_cot_rounds1` |
| `r3_no_caption` | `qwen3-VL-4B_forward2_reflective_answer_first_no_caption_r3_rounds1_3shards_gpu5` |
| `keep_revise` | `qwen3-VL-4B_forward2_reflective_keep_revise_caption_r3_3shards_gpu4` |
| `visible_only` | `qwen3-VL-4B_forward2_reflective_visible_only_caption_r3_3shards_gpu5` |
| `reviewer_caption_only` | `qwen3-VL-4B_forward2_reviewer_evidence_caption_only_rounds1_3shards_gpu4` |

## 总体 Oracle 上界

- 对齐样本数：`1145`
- r3 caption soft accuracy：`59.60%`
- oracle 每题选最高分方法：`70.81%`
- 理论可提升：`+11.21` 个百分点，约 `128.4` 个 soft-correct 样本
- soft 可改进题数：`201`
- 严格 r3=0 且其他方法>0 的题数：`127`

## 哪些方法补到了 r3 的错题

| 方法 | 成为 best 的题数 |
|---|---:|
| `answer_first_locked_no_caption` | 123 |
| `r3_no_caption` | 21 |
| `review_empty_context` | 20 |
| `no_cot_rounds1` | 20 |
| `keep_revise` | 8 |
| `reviewer_caption_only` | 8 |
| `visible_only` | 1 |

## soft 可改进题型分布

| 题型 | 题数 |
|---|---:|
| `object_what` | 68 |
| `category_type` | 32 |
| `spatial_location` | 27 |
| `other` | 25 |
| `purpose_reason` | 21 |
| `action_state` | 17 |
| `text_ocr` | 4 |
| `count` | 4 |
| `color` | 3 |

## 严格可挽救错题题型分布

| 题型 | 题数 |
|---|---:|
| `object_what` | 39 |
| `category_type` | 20 |
| `other` | 18 |
| `purpose_reason` | 15 |
| `spatial_location` | 13 |
| `action_state` | 12 |
| `text_ocr` | 4 |
| `count` | 4 |
| `color` | 2 |

## Top 可改进样例

| idx | type | question | r3 | r3 score | best method | best answer | best score |
|---:|---|---|---|---:|---|---|---:|
| 34 | `category_type` | The visible bottles most likely contain what kind of items? | shampoo and conditioner | 0.0 | `review_empty_context` | shampoo | 1.0 |
| 43 | `other` | Which food item is the knife for? | food | 0.0 | `r3_no_caption` | meat | 1.0 |
| 44 | `purpose_reason` | What is the orange container on the left near the man in the red shirt used for? | cooling | 0.0 | `answer_first_locked_no_caption` | cooler | 1.0 |
| 61 | `object_what` | What festive season are these fruits usually ingested? | holiday | 0.0 | `answer_first_locked_no_caption` | christmas | 1.0 |
| 68 | `purpose_reason` | Why are the men wearing yellow vests? | work | 0.0 | `answer_first_locked_no_caption` | safety | 1.0 |
| 70 | `object_what` | What is the object in the middle called? | counter | 0.0 | `answer_first_locked_no_caption` | island | 1.0 |
| 90 | `object_what` | What seems to be contained in the nook underneath the TV? | tv | 0.0 | `answer_first_locked_no_caption` | fireplace | 1.0 |
| 104 | `other` | In which way are the adults shown here likely related to the child? | parent | 0.0 | `no_cot_rounds1` | parents | 1.0 |
| 128 | `other` | The temperature outside is likely what range? | cool | 0.0 | `answer_first_locked_no_caption` | cold | 1.0 |
| 174 | `object_what` | What other animal is this animal traditionally an enemy of? | rat | 0.0 | `answer_first_locked_no_caption` | cat | 1.0 |
| 198 | `purpose_reason` | What are the horses being used for? | plow | 0.0 | `answer_first_locked_no_caption` | plowing | 1.0 |
| 276 | `spatial_location` | What type of buildings are located here? | house | 0.0 | `answer_first_locked_no_caption` | houses | 1.0 |
| 288 | `other` | The glasses contain only this beverage type? | uncertain | 0.0 | `answer_first_locked_no_caption` | beer | 1.0 |
| 291 | `other` | Whos is sitting in the chair? | man and child | 0.0 | `r3_no_caption` | man | 1.0 |
| 294 | `object_what` | What brand of soda is the can in the car? | uncertain | 0.0 | `answer_first_locked_no_caption` | pepsi | 1.0 |
| 301 | `purpose_reason` | The bag which the cat is standing is used for what? | carry | 0.0 | `answer_first_locked_no_caption` | travel | 1.0 |
| 313 | `category_type` | What type of dog is it? | labrador | 0.0 | `answer_first_locked_no_caption` | german shepherd | 1.0 |
| 319 | `object_what` | What family member in a household would this room most likely belong to? | adult | 0.0 | `keep_revise` | child | 1.0 |
| 323 | `purpose_reason` | The piece of clothing the woman in the picture is wearing on her hands normally serves what purpose? | protect | 0.0 | `answer_first_locked_no_caption` | protection | 1.0 |
| 378 | `purpose_reason` | Why is the person's vest yellow? | safety | 0.0 | `r3_no_caption` | visibility | 1.0 |

## 初步结论

1. 如果能学会在少量题上从 r3 caption 切换到其他强方法，理论上足够突破 60%。
2. 最有价值的不是新增长 CoT，而是做一个轻量 routing：什么题保持 r3，什么题切换到 no-caption/empty-review/locked。
3. 下一步建议人工抽查 `reflective_oracle_r3_error_cases.csv` 中 delta 最大的题，归纳可写成规则的失败模式。
4. 严格 r3=0 的可挽救错题更适合做规则，因为它们不是 soft-score 微小差异。

## 输出文件

- soft 可改进题：`/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/reflective_oracle_r3_error_cases.csv`
- 严格 r3=0 可挽救题：`/data2/lizhengxue/WorkSpace/onion_output/ablation_summary/reflective_oracle_r3_strict_zero_cases.csv`
