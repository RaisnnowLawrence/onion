# Onion VQA Research Roadmap

Last updated: 2026-06-16

## Project Goal

本项目当前目标是把 A-OKVQA / VQA 开放式问答准确率从约 59% 稳定提升到 70% 以上，并尽量形成可以写进论文的方法路线，而不是只靠零散工程选项堆叠。

当前最重要的认识是：

```text
direct 决定下限，候选覆盖决定上限，router / arbitration 决定能否接近 oracle。
```

因此后续工作不应该只问“哪个单策略最高”，而要同时维护三个指标：

- 单策略 accuracy：实际可用性能。
- oracle accuracy：候选池理论上限。
- all-wrong 数量：所有策略都无法覆盖的困难题规模。

## Evaluation Protocol

2026-06-16 已将本工程的 A-OKVQA direct-answer 评测逻辑对齐到官方仓库 `allenai/aokvqa`：

- 官方 DA 只评测 `difficult_direct_answer == False` 的样本，不把 difficult direct-answer 样本计入 DA 分母。
- 单题 DA 分数为 exact match：`min(1.0, num_match / 3.0)`。
- 官方代码不做我们旧版里的小写、去标点、去冠词等额外归一化。
- 全局 DA score 是上述单题分数在官方 DA 子集上的平均值乘以 100。

本工程当前默认同时汇报官方指标和旧内部指标，便于论文报告和历史实验对比。每次 merge / 正常跑完会输出：

- `官方DA准确率`：官方子集 + exact match + `matches / 3`，论文优先使用。
- `旧指标@官方DA子集`：官方子集 + 旧归一化，便于判断预处理差异。
- `全量官方exact诊断`：全量样本 + 官方 exact-match 单题公式，作为内部诊断。
- `旧指标@全量诊断`：全量样本 + 旧归一化，对齐历史 `1145` 口径。

仍然保留两个显式参数用于控制主指标和逐样本保存分数：

- `--eval_all_direct_answers`：把 difficult direct-answer 样本也计入全量内部诊断。
- `--legacy_answer_normalization`：使用旧版 `process_answer` 归一化和 `0.3 * match` 评分。

注意：历史实验中的 `59.xx% (xxx/1145)` 多数是旧的全量内部口径，不应直接当作官方 A-OKVQA DA leaderboard 分数。后续论文报告优先使用官方 DA 口径；全量 1145 指标只作为内部分析辅助。

## Current Baselines And Observations

### Strong Direct Baseline

当前最强的普通单策略基线仍然是：

```text
no_cot_rounds1: 59.24% (678/1145)
```

如果严格定义 direct 为 no-CoT 直接短答，那么 `no_cot_rounds1` 是当前 direct best。
如果把 direct-preserving / answer-first 类方法也算作 direct-like 基本盘，那么当前最高单策略是：

```text
reflective_answer_first_caption_r3: 59.60% (682/1145)
```

这两者差距只有约 0.36 个百分点，也就是约 4 个 soft-correct。考虑到不同代码阶段、prompt 细节和 soft score 波动，这个差距不应被过度解读。工程推进时，二者任意一个都可以作为基本盘：

- `no_cot_rounds1` 更适合作为论文里的干净 direct baseline，因为它最简单、变量最少。
- `reflective_answer_first_caption_r3` 更适合作为工程冲分的 direct-like base，因为它是当前最高单策略，且仍然保留了先答后复核的 direct-preserving 结构。
- 做 oracle / router / arbitration 时，建议两个都保留，因为它们存在互补。

这说明 Qwen3-VL-4B 的 image-question direct 能力本身很强。很多 CoT / evidence-heavy 方法没有提升，反而会把 direct 本来答对的题改错。

直接结论：

- direct 不是一个可以随便替换的弱模块。
- 后续任何 router 方法都必须保护强 direct 底座。
- 如果新方法内部的 direct prompt 比 `no_cot_rounds1` 弱，那么 overall 很容易下降。

### CoT And Complex Decomposition

复杂拆解实验结果：

| Strategy | Accuracy |
| --- | ---: |
| `complex_decompose_adaptive_verify` | 56.90% |
| `complex_decompose_conservative_verify` | 56.41% |
| `complex_decompose_adaptive` | 55.93% |
| `complex_decompose_always` | 54.79% |

结论：

- “所有题都拆解”会伤害简单题。
- 复杂拆解单独作为主策略不够强。
- 复杂拆解作为候选补充有少量价值，但不能直接替代 direct。

复杂拆解 4 策略 oracle 约为 61.08%，加入旧候选池后只带来有限增益：

```text
old9 oracle: 71.62%
old9 + complex4 oracle: 72.28%
```

### Candidate Expansion And Oracle

旧策略集合的 oracle 演进：

| Strategy Set | Oracle | All-wrong |
| --- | ---: | ---: |
| old6 | 69.33% | 220 |
| old9 = old6 + coverage/count/diverse | 71.62% | 193 |
| old9 + train-RAG4 | 71.98% | 187 |

这里的 `old6` 包含：

- `no_cot`
- `reflective_r3`
- `answer_first_no_caption`
- `reflective_empty_review`
- `candidate_marker_mcts`
- `rag_protected_n400`

这里的 `old9` 在 `old6` 基础上增加：

- `coverage_scan`
- `count_specialist`
- `diverse_pool`

重要结论：

- 候选扩展比 train-RAG router 更能提升 oracle。
- 当前主要瓶颈仍然是正确答案没有出现在候选集合里。
- 如果 oracle 不涨，router 再好也没有空间。

### Train-RAG Router Experiment

2026-06-15/16 跑了四个 train-guided RAG router 方向，每个方向用约 10k train profile，每个实验 3 shard：

| Experiment | Accuracy | Complex Triggered | Complex Subset Accuracy |
| --- | ---: | ---: | ---: |
| `direct_vs_complex` | 57.13% | 129 | 57.98% |
| `qtype_conditional` | 56.86% | 93 | 52.15% |
| `conservative_risk` | 56.62% | 153 | 59.02% |
| `direct_failure` | 56.23% | 94 | 47.77% |

对旧 6 策略全错的 220 题：

| Experiment | Score On 220 | Accuracy On 220 | Positive Cases |
| --- | ---: | ---: | ---: |
| `direct_failure` | 10.2 / 220 | 4.64% | 14 |
| `direct_vs_complex` | 11.7 / 220 | 5.32% | 18 |
| `qtype_conditional` | 10.1 / 220 | 4.59% | 15 |
| `conservative_risk` | 8.5 / 220 | 3.86% | 13 |
| train-RAG4 oracle | 13.6 / 220 | 6.18% | 21 |

结论：

- train-RAG router 有少量区分信号，尤其 `conservative_risk` 触发的 complex 子集比 direct 分支略好。
- 但这版整体没有提升，因为 router 内部的 direct 底座只有 56-57%，弱于 `no_cot_rounds1`。
- 更关键的是，它对 220 个旧全错困难题覆盖很弱，说明困难题不是简单的 router 问题，而是候选/证据/视觉识别问题。

## Main Research Directions

### 1. Improve Direct Accuracy

目标：提升简单题和普通题的基本盘。

为什么重要：

- direct 是目前最稳的单策略。
- direct 的每 1 个点提升都会直接反映到最终系统下限。
- router 如果默认分支弱，会拖垮整体。

具体方向：

- 严格复现并保护 `no_cot_rounds1` prompt 和上下文设置。
- 比较 direct prompt 的微小差异，避免 router/direct_verify 分支无意中换弱 prompt。
- 尝试更强的 answer extraction，但不要引入解释性 CoT。
- 对 direct 正确、CoT 错误的样本做诊断，提炼“不要改”的保护规则。

更具体的推进原则：

- direct 的目标不是“更会解释”，而是“短答更稳、更不跑偏”。
- direct prompt 变体应该小而干净，不要一次引入多个变量。
- 后续所有新方法都必须和 `no_cot_rounds1`、`reflective_answer_first_caption_r3` 并排比较。
- 如果一个方法提升困难题但明显伤害 direct-correct 样本，它不适合作为默认策略，只能作为候选生成模块。

推荐 direct 微调实验：

- `direct_image_only`：只看图和问题，减少 caption 噪声。
- `direct_image_caption`：图像 + 全局 caption 的短答 direct。
- `direct_regional_context`：图像 + top regional caption 的短答 direct。
- `direct_answer_first_strict`：先给答案，不允许解释，强制短语输出。
- `direct_type_specialist`：根据问题类型选择输出约束，例如 count 输出数字、color 输出颜色词、OCR 输出读到的文本。

预期目标：

```text
clean direct baseline: 59% 左右
短期目标: 稳定超过 60%
中期目标: 61-62%，且不依赖复杂 CoT
```

### 2. Improve Hard-Case Accuracy

目标：专门提升旧策略全错的 220 题，后续也可以聚焦 old9 剩余的 193 题或最新 187 题。

为什么重要：

- 这部分决定继续提升 oracle 的空间。
- 如果所有策略都答不出，router 没有任何办法。

当前观察：

- 220 题里，train-RAG4 oracle 只有 6.18%。
- 很多失败不是推理形式问题，而是正确答案没有进入证据或候选。
- count、OCR、细粒度属性、局部目标、空间关系、常识桥接都是高风险类型。

具体方向：

- 针对 220/193/187 题做错误类型分桶。
- 为 count/OCR/spatial/attribute/brand/commonsense 分别设计专家候选。
- 增加区域裁剪、局部放大、目标定位，而不是只改语言 prompt。
- 让模型显式列出可能答案候选，再做答案覆盖检查。

更具体的推进原则：

- 困难题阶段先不追求单策略 accuracy，优先追求 oracle。
- 专家模块的责任是“把正确答案带进候选池”，不是马上成为最终答案。
- complex decomposition 只用于困难题候选生成，不应该全量覆盖 direct。
- 如果正确答案没有出现在候选池、OCR、object、caption 或 regional evidence 中，router 没有提升空间。

推荐困难题分桶：

- `ocr_text`：答案依赖文字、logo、标志、号码。
- `count`：答案依赖精确计数。
- `spatial_relation`：答案依赖左右、前后、相邻、遮挡等空间关系。
- `fine_attribute`：答案依赖颜色、材质、状态、动作等细粒度属性。
- `small_object_region`：关键目标很小或只在局部区域出现。
- `commonsense_bridge`：需要先识别图像事实，再做常识桥接。
- `answer_format`：模型语义答对但格式、同义词、粒度与标注不一致。
- `ambiguous_or_unanswerable`：图像/标注本身存在歧义。

推荐 hard-case candidate generator：

- `hard_ocr_specialist`：专门读图中文字，并把可能文本答案加入候选。
- `hard_count_specialist`：对 count 问题进行显式视觉计数，输出数字候选。
- `hard_region_crop_specialist`：根据问题关键词选择局部区域，放大后重新短答。
- `hard_relation_specialist`：针对空间关系和交互关系生成候选。
- `hard_commonsense_enumerator`：先列出图像事实，再枚举 3-5 个可能短答案。
- `hard_answer_canonicalizer`：对候选做同义词、单复数、短语粒度归一化，减少格式错配。

预期目标：

```text
old9 + train-RAG4 oracle: 71.98%
短期目标: oracle > 73%
中期目标: oracle 74-75%，all-wrong 从 187 降到 150 以下
```

### 3. Improve Routing Accuracy

目标：在候选池足够强之后，选择 direct、complex、candidate specialist 或保持原答案。

为什么重要：

- CoT 和 complex 容易把 direct 本来答对的题改错。
- router 的价值在于减少策略互相伤害。

当前教训：

- router 不应该重新生成弱 direct。
- router 应该读取或复用强 direct 输出。
- complex 不能默认覆盖 direct，必须通过保守证据仲裁。

具体方向：

- 使用 `no_cot_rounds1` 作为固定 direct candidate。
- RAG 只负责判断是否额外运行 complex/specialist。
- 对触发样本记录 rescue/damage rate。
- 只有在 `rescue_rate - damage_rate` 明显为正时才允许覆盖。

### 4. Improve Candidate Coverage / Oracle

目标：提高正确答案出现在候选集合中的概率。

为什么重要：

- oracle 是最终系统的理论上限。
- 当前 old9 + train-RAG4 oracle 只有 71.98%，距离 80% 还有很大缺口。

具体方向：

- 对 all-wrong 样本统计 gold answer 是否出现在候选、对象、caption、OCR、区域描述中。
- 增加 recall-oriented evidence：宁可多给候选，不要过早收窄。
- 对候选生成进行多视角扰动：direct image-only、caption-only、region-only、OCR-only、count-only、object-focused。
- 对候选池做去重、归一化和语义匹配，避免“答案其实出现但格式不匹配”。

## Recommended Next Experiments

### Priority 1: Strong Direct As Frozen Base

重新做一版 router，但 direct 不在 router 内部重新生成，而是固定使用 `no_cot_rounds1` 的已有答案。

流程：

1. 读取 `no_cot_rounds1` 作为 direct candidate。
2. RAG 只判断是否触发 complex/specialist。
3. 未触发时直接保留 no_cot。
4. 触发时加入 complex/specialist candidate。
5. 用保守仲裁决定是否覆盖。

预期价值：

- 避免 direct 底座从 59.24% 掉到 56-57%。
- 更真实地评估 router 是否能带来净增益。

### Priority 2: Hard-Case Candidate Generator

只针对旧 220 / old9 剩余 193 / 最新 187 all-wrong 题做候选覆盖实验。

目标不是马上提升 full accuracy，而是先提升 oracle：

```text
目标 1: old9 + new candidates oracle > 73%
目标 2: all-wrong 从 187 降到 150 以下
```

可尝试候选：

- count specialist with explicit visual counting
- OCR/text specialist
- region crop specialist
- object relation specialist
- answer option / plausible answer enumeration

### Priority 3: Direct Specialist Baseline Sweep

目标是在不引入解释性 CoT 的前提下提升 direct 基本盘。

实验组建议：

1. `direct_image_only`
2. `direct_image_caption`
3. `direct_regional_context`
4. `direct_answer_first_strict`
5. `direct_type_specialist`

每个实验需要同时报告：

- 官方 DA accuracy
- 旧指标全量 accuracy
- direct-correct damage count
- 相对 `no_cot_rounds1` 的 rescue / damage
- 在 220 / 193 / 187 hard subset 上的覆盖变化

保留标准：

- 如果 full accuracy 没有提升，但 oracle 或 hard subset 覆盖提升，可以作为候选模块保留。
- 如果 full accuracy 提升但 hard subset 无变化，可以作为 direct base 候选保留。
- 如果伤害大量 direct-correct 样本，则不能作为默认 direct。

### Priority 4: Hard Candidate Generator Sweep

目标不是马上提升 final accuracy，而是提升 hard subset oracle 和候选覆盖。

实验组建议：

1. `hard_ocr_specialist`
2. `hard_count_specialist`
3. `hard_region_crop_specialist`
4. `hard_relation_specialist`
5. `hard_commonsense_enumerator`
6. `hard_answer_canonicalizer`

每个实验需要同时报告：

- 对 old9 all-wrong 193 / latest all-wrong 187 的新增 positive cases。
- 是否让 gold answer 出现在 candidate pool。
- 对全量 val oracle 的增益。
- 是否误伤 direct-correct 样本。

保留标准：

- 优先保留能提升 oracle 的模块，即使单策略 accuracy 不高。
- 若模块只在极少数题上有效，但错误类型清晰，也可以作为专家候选保留。
- 若模块大幅增加噪声但不增加 gold coverage，应删除或收紧触发条件。

### Priority 5: Evidence Coverage Diagnostics

建立自动报告：

- gold 是否出现在任何 candidate。
- gold 是否出现在 detected objects。
- gold 是否出现在 OCR。
- gold 是否出现在 caption/regional caption。
- 错误是否属于 count/OCR/attribute/spatial/knowledge。

这个报告用于指导候选生成，而不是只看最终 accuracy。

## Working Hypotheses

### Hypothesis A

很多困难题不是“模型不会推理”，而是 evidence pipeline 没有覆盖正确答案所需的对象、属性、文本或区域。

### Hypothesis B

CoT / complex decomposition 的价值主要在 hard subset，不适合作为全局默认策略。

### Hypothesis C

Train-RAG 可以帮助识别 hard subset，但它不能替代候选生成；如果候选池没有正确答案，router 没有上限空间。

### Hypothesis D

想达到 70% 以上，最现实路径不是单一策略突破，而是：

```text
strong direct base + high-recall candidate generation + conservative arbitration
```

## Experimental Hygiene

后续每次实验应记录：

- run name / output path
- exact strategy and key args
- full val accuracy
- oracle contribution
- all-wrong remaining count
- direct-correct damage count
- direct-wrong rescue count
- hard-subset accuracy on 220 / 193 / latest all-wrong set

如果一个实验只提高 full accuracy 但降低 oracle，需要单独分析是否只是随机波动。如果一个实验 full accuracy 不高但 oracle 增加，也可能值得保留为候选生成模块。
