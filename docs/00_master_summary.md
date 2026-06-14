# VisualCoT / A-OKVQA 消融实验总总结

生成时间：2026-06-14

本文档是当前实验的总入口。更细的分组文档在同目录下：

- `01_baseline_context_caption.md`：基础设置、caption/context、n-shot、ensemble、answer extraction、Qwen caption 等早期消融。
- `02_mcts_image_enhancement.md`：MCTS/SAM 图像增强相关实验。
- `03_cot_rescue_direct_verify_reviewer.md`：direct-verify、conflict-only、reviewer evidence 等“证据审查式 CoT”实验。
- `04_pure_cot_answer_first_reflective.md`：纯 CoT、answer-first locked、visual facts、当前 reflective answer-first 实验。
- `05_all_results_index.md`：所有已有 `accuracy.log` 的结果索引。
- `06_code_evolution_and_comparability.md`：不同实验对应的主要代码版本、代码逻辑变化和可比性风险。
- `07_reflective_oracle_analysis.md`：逐题 oracle 分析，找 r3 caption 错但其他强方法能补对的题。

## 统一实验背景

除非特别说明，实验设置如下：

| 项目 | 设置 |
|---|---|
| 数据集 | A-OKVQA val |
| 样本数 | 1145 |
| 模型 | qwen3-VL-4B |
| 主代码 | `/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-pure/forward_code/onion.py` |
| 输出根目录 | `/data2/lizhengxue/WorkSpace/onion_output/aokvqa` |
| 结果指标 | A-OKVQA soft accuracy，最终写入各实验目录的 `accuracy.log` |
| 常用 caption | `--caption_type vinvl` |
| 常用检索 | `--train_sim_metric answer` |

需要注意：早期少数实验样本数不是 1145，例如旧 `rounds1` 有 1147 的重复 shard 问题，应以 `clean_rounds1` 或后续重新 merge 的结果为主。

## 代码版本与可比性提醒

这一批实验不是在完全同一个代码版本上一次性跑完的。更准确地说，它们是在不断修正工程问题、把开关 args 化、添加新推理结构的过程中逐批完成的。因此表格里的数字可以用来判断“方法族趋势”，但不能把所有实验都当成同一份静态代码下的严格公平 leaderboard。

主要代码演化如下：

| 阶段 | 大致对应实验 | 代码主要做了什么 | 可比性说明 |
|---|---|---|---|
| 早期 forward / baseline | `forward_baseline`、`rounds1/3`、caption、n-shot、OCR、CLIP thought | 基于原 forward 流程，保留旧式多轮 CoT、caption/context 注入、基础增强开关 | 可作为早期 baseline，但和后续 answer extraction / shard merge 修正后的结果不完全同代码 |
| clean / context / no-CoT | `clean_rounds1`、`no_cot_rounds1`、`context_*`、`answer_*` | 修正/统一 merge 与 answer extraction，加入 `context_mode`、`answer_extraction_strategy` 等开关 | 这是后续大部分实验更可靠的比较基准 |
| MCTS 参数化 | `mcts_*`、`marker/outline/narrow/safe` | 将 MCTS 搜索数、触发条件、动作集合、对象过滤 args 化；测试 crop/marker/outline 等图像增强 | 主要和 no-CoT / conflict 系列比较，不能直接等同早期原始 MCTS |
| compact / conflict CoT rescue | `cot_rescue_compact_*`、`cot_rescue_conflict_*` | 把 CoT 从长 step-by-step 改成短 visual cues 或 conflict-only 复核；减少推理链自由度 | 可和旧 CoT 比较 CoT 结构变化，但它已经不是原始 CoT |
| direct_verify | `direct_verify_*` | 新增 `cot_style=direct_verify`，先生成 direct answer，再 verifier 判断 supported/contradicted/uncertain | 是“两次模型调用”的审查链，成本和结构都不同于 direct/no-CoT |
| reviewer evidence | `reviewer_evidence_*` | 新增 `cot_style=reviewer_evidence`，把 image/caption/knowledge/MCTS 等模块输出整理为 evidence 给 reviewer | 三增强模块在这里变成证据源，不再是直接生成答案的主链 |
| pure CoT answer-first | `pure_cot_answer_first_locked`、`visual_facts` | 新增 `answer_first_locked`、`visual_facts` 等纯 CoT prompt；answer-first 只抽第一行答案 | 证明“先答再解释”可接近 direct，但与旧 step-by-step CoT 代码不同 |
| reflective answer-first | 当前 running 的 `reflective_*` | 新增 `reflective_answer_first` 和 `reflect_rounds`：先答、补证据、保守复核 | 当前最新代码；结果出来后应单独和 answer-first/direct 比较 |

因此，推荐报告方式是：

1. 按方法族汇报：direct/no-CoT、旧 CoT、MCTS、direct-verify/reviewer、answer-first。
2. 每个方法族内比较更可信，因为通常共享同一批代码开关。
3. 跨方法族比较时，要明确它们背后代码结构已经变化，尤其是是否多次调用模型、是否使用增强图像、是否只抽第一行答案。
4. 最终如果要写正式表格，建议选定当前最新代码，重跑少数关键配置：`no_cot_rounds1`、`answer_first_locked`、`reviewer_evidence_caption_only`、最佳 MCTS、最新 reflective。

## 当前最高结果

| 排名 | 实验 | 准确率 | 正确数 | 核心设置 |
|---:|---|---:|---:|---|
| 1 | `reflective_answer_first_caption_r3` | 59.60% | 682/1145 | 先答、补最小证据、保守复核，带基础 caption |
| 2 | `answer_first_locked_no_caption` | 59.51% | 681/1145 | 无基础 caption，先给答案，再给短理由，只取第一行 |
| 3 | `reflective_review_empty_context_caption_r3` | 59.46% | 680/1145 | 初答带 caption，复核阶段去掉 caption |
| 4 | `no_cot_rounds1` | 59.24% | 678/1145 | 直接短答，不使用 CoT，rounds=1 |
| 5 | `reflective_keep_revise_caption_r3` | 59.24% | 678/1145 | reviewer 只能 keep/revise |
| 6 | `reflective_answer_first_no_caption_r3` | 59.22% | 678/1145 | 无基础 caption，先答、补证据、保守复核 |
| 7 | `reflective_visible_only_caption_r3` | 59.21% | 677/1145 | 证据阶段只允许可见事实 |

上一版 reflective answer-first r3 已经首次超过 `direct no-CoT rounds1`：`59.60%`，多 4 题。它说明“先保留模型直觉答案，再让 CoT 做最小证据审查”比传统 CoT 更有效。

## 总体结论

### 1. Direct no-CoT 是当前最强基线

`no_cot_rounds1` 达到 `59.24%`。它不是“只看图像+问题”的裸 direct，而是在当前 forward pipeline 中保留基础输入结构和 vinvl caption/context，去掉显式 CoT 推理链后直接短答。

对照：

| 实验 | 准确率 | 说明 |
|---|---:|---|
| `no_cot_rounds1` | 59.24% | 当前最佳 |
| `direct_image_question_only` | 58.54% | 只给图像+问题，去掉基础 caption/context |
| `clean_rounds1` | 55.33% | 旧式 CoT，1 round |
| `forward_baseline` | 52.72% | 早期 5-round CoT baseline |

这说明基础 caption/context 对 direct 仍有用，但显式 step-by-step CoT 会显著伤害模型。

### 2. CoT 本身不是完全无效，关键是不能让推理链先污染答案

最重要的新发现是 `answer_first_locked`：

| 实验 | 准确率 | 说明 |
|---|---:|---|
| `pure_cot_answer_first_locked` | 59.17% | 先输出 Answer，再给短理由；最终只抽第一行 Answer |
| `visual_facts` | 53.20% | 先列 visible facts 再答，明显失败 |
| `visual_facts_no_caption` | 53.31% | 无 caption 版本仍失败 |
| `old CoT rounds1` | 55.33% | step-by-step 风格，明显低于 direct |

结论：如果 CoT 先写 reasoning，再根据 reasoning 生成答案，容易被语言先验、错误视觉事实、格式污染带偏；如果先锁定 direct answer，再让理由只作为解释，性能接近 direct best。

### 3. “审查者 CoT”比“生成式 CoT”更适合当前工程

direct-verify / reviewer evidence 的整体思想是：

```text
先直接回答 -> 三增强模块或 caption/context 提供证据 -> reviewer 只判断证据是否足以推翻初始答案
```

这个方向比传统 CoT 稳定：

| 实验 | 准确率 | 判断 |
|---|---:|---|
| `reviewer_evidence_caption_only` | 59.13% | 强，接近 best |
| `conflict_no_round_state` | 59.10% | 强，说明保守修改有效 |
| `direct_verify_conflict_only` | 58.96% | 接近 best |
| `reviewer_evidence_all` | 58.49%-58.54% | 证据太多时反而略降 |
| `reviewer_evidence_selective` | 58.49% | selective 目前没有明显提升 |

最有价值的原则是“证据只用来推翻明显错误，不用来重新生成答案”。

### 4. MCTS 图像增强目前不应作为默认主线

MCTS 相关实验的最好结果接近 direct，但不稳定超过 direct：

| 实验 | 准确率 | 说明 |
|---|---:|---|
| `mcts_narrow_no_cot_n10` | 59.09% | 最好 MCTS 结果，仍低于 direct best |
| `mcts_marker_narrow_n20` | 58.97% | 搜索数更高，略低于 n10 |
| `mcts_marker_narrow_n10` | 58.76% | 接近 direct image-question only |
| `mcts_image_no_cot_n5` | 53.92% | 原始图像增强动作明显伤害 |

原因判断：A-OKVQA 很依赖全局上下文、常识、问题语义和 caption；MCTS/SAM 若直接改变图像，容易突出错误区域或丢失上下文。因此我们后续把 MCTS 从“改变图像”改成“给图像加最小必要提示”是合理方向，但当前还未证明能超过 direct。

### 5. Qwen 生成 caption 注入最终 prompt 基本不可用

| 实验 | 准确率 |
|---|---:|
| `qwen_caption_no_final` | 52.66% |
| `qwen_caption_local` | 45.23% |
| `qwen_caption_global` | 44.06% |
| `qwen_caption_short` | 43.87% |
| `forward_qwen_caption` | 44.46% |

只生成但不注入最终回答时没有明显问题；一旦注入最终 prompt，准确率大幅下降。说明模型生成的 caption 容易引入错误视觉细节或让最终回答过度依赖二手文本。

## 上一版 reflective answer-first 实验结果

本轮刚改的代码已推送 GitHub，commit：

```text
a9a7a18 add reflective answer first ablations
```

新结构：

```text
Round 1：先短答
Round 2：只补最小视觉证据，不允许改答案
Round 3：保守复核，只有 Evidence Check=contradicted 才允许改答案
Round 4/5：可重复一轮证据和复核
```

实验结果：

| 实验 | 准确率 | 正确数 | 结论 |
|---|---:|---:|---|
| `reflective_answer_first_caption_r3` | 59.60% | 682/1145 | 当前最高，证明 3 阶段审查链有效 |
| `reflective_answer_first_no_caption_r3` | 59.22% | 678/1145 | 去掉基础 caption 后接近 direct best |
| `answer_first_locked_no_caption` | 59.51% | 681/1145 | 无 caption 仍很强，answer-first 本身有效 |
| `reflective_answer_first_caption_r5` | 58.68% | 671/1145 | 多一轮证据/复核后掉点，说明多轮会累积噪声 |

这批实验的结论：

- 有 caption 的 r3 最强，基础 VinVL caption 对 reflective 审查链有正收益。
- 无 caption 的 answer-first locked 仍达到 59.51%，说明 answer-first 结构本身很关键。
- r5 明显低于 r3，说明继续增加 round 不是方向，下一步应做 adaptive/选择性复核，而不是简单加轮数。

## 实验操作与版本控制

已经做过的关键代码方向：

| 方向 | 代码/参数表现 | 目的 |
|---|---|---|
| context 消融 | `--context_mode empty/caption_only/objects_only/no_round_state` | 判断上下文噪声来源 |
| answer extraction | `--answer_extraction_strategy` | 判断是否是答案解析造成掉点 |
| direct verify | `--cot_style direct_verify` | 把 CoT 改成答案复核 |
| reviewer evidence | `--cot_style reviewer_evidence` | 三增强模块作为证据提供者 |
| MCTS 参数化 | `--mcts_n_simulations`、`--mcts_action_mode`、`--mcts_trigger_mode` | 控制图像增强搜索强度和动作 |
| pure CoT rescue | `--cot_style answer_first_locked/visual_facts` | 测试纯 CoT 是否能接近 direct |
| reflective answer-first | `--cot_style reflective_answer_first --reflect_rounds` | 先答、补证据、保守复核 |

GitHub 仓库：

```text
/data2/lizhengxue/WorkSpace/visualcot_experiment_repo
git@github.com:RaisnnowLawrence/visualcot_experiment_repo.git
```

## 推荐写作表述

如果写论文/报告，建议把故事线组织为：

1. 直接短答是强基线，传统 CoT 明显掉点。
2. 掉点原因不是答案解析，而是 CoT 推理链引入错误视觉事实和语言先验。
3. 三增强模块不适合作为“答案生成器”，更适合作为“证据提供器”。
4. CoT 的合理位置不是先行推理，而是作为 reviewer 判断证据是否足以推翻初始答案。
5. 最强纯 CoT 变体是 answer-first locked，说明“先保留模型直觉，再要求解释”是比 step-by-step 更稳定的结构。

## 当前建议

短期最值得继续的是：

1. 等 reflective answer-first 四组跑完，重点看是否超过 `answer_first_locked` 或 `direct no-CoT`。
2. 若 `answer_first_locked_no_caption` 仍接近 59%，说明基础 caption 不是关键，answer-first 结构本身有效。
3. 若 `reflective r5` 低于 `reflective r3`，说明多轮复核仍然会累积噪声，应限制为 3 阶段。
4. 若 `reflective no_caption` 明显低于 caption 版，说明基础 VinVL caption 是 direct/answer-first 的重要证据。

## Follow-up 5 实验结果

基于上一版 `reflective_answer_first_caption_r3 = 59.60%`，当前已启动 follow-up 5：

| 实验 | 准确率 | 正确数 | 结论 |
|---|---:|---:|---|
| `reflective_adaptive_highrisk_lowconf_caption_r3` | 58.59% | 670/1145 | 选择性触发复核明显掉点，规则漏掉了需要复核的问题或置信度不可靠 |
| `reflective_keep_revise_caption_r3` | 59.24% | 678/1145 | 强约束 reviewer 后回到 direct best，但低于原 r3 |
| `reflective_visible_only_caption_r3` | 59.21% | 677/1145 | 禁止常识/用途类证据略降，说明 A-OKVQA 仍需要一点常识语义 |
| `reflective_review_empty_context_caption_r3` | 59.46% | 680/1145 | 本轮最好，说明 caption 主要帮助初答，复核阶段去 caption 可以减少文本干扰 |
| `reflective_initial_ensemble3_keep_revise_caption_r3` | 58.76% | 672/1145 | 初答三投票没有帮助，额外采样反而引入不稳定 |

对应代码 commit：`d6d35bb add adaptive reflective answer first ablations`。

这轮结论：继续收窄 reviewer 是有用的，但不能简单跳过复核，也不能用初答 ensemble 替代更好的审查。当前最稳方向是 `caption for initial answer + no caption for review`。

## 逐题 Oracle 分析

已新增逐题 oracle 分析：`07_reflective_oracle_analysis.md`。

核心结果：

| 项目 | 数值 |
|---|---:|
| 对齐样本数 | 1145 |
| r3 caption soft accuracy | 59.60% |
| oracle 每题选最高分方法 | 70.81% |
| 理论可提升 | +11.21 points |
| soft 可改进题数 | 201 |
| 严格 r3=0 且其他方法>0 的题数 | 127 |

最能补 r3 错题的方法是 `answer_first_locked_no_caption`，成为 best 的题数为 123。这个结果说明：突破 60 的潜力主要来自逐题 routing，而不是继续增加 CoT 轮数。
