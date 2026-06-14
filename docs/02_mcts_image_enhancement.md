# MCTS / SAM 图像增强消融

生成时间：2026-06-14

本文档整理 MCTS 图像增强相关实验。这里的 MCTS 主要用于选择图像增强动作，例如 crop、outline、marker 等；后续又加入了 narrow/safe/trigger/action mode 等收窄策略。

## 对应代码阶段说明

MCTS 相关实验横跨了两个代码阶段：

| 阶段 | 包含实验 | 当时代码主要特点 |
|---|---|---|
| 原始 MCTS 图像增强 | `mcts_image_no_cot_rounds1_n5` 等 | MCTS/SAM 直接对图像做增强，动作集合较宽，容易 crop 或突出错误区域 |
| 参数化/收窄 MCTS | `mcts_narrow_*`、`mcts_marker_*`、`mcts_safe_*`、`conflict_marker_mcts_*` | 加入 `mcts_n_simulations`、`mcts_action_mode`、`mcts_trigger_mode`、`mcts_filter_objects` 等开关，逐步减少对图像的破坏 |

所以，MCTS 表格不能只按 n_simulations 横向解释；动作集合、触发条件、是否 no-CoT/CoT/reviewer 都会影响结果。

## 实验动机

最初设想：

```text
问题 -> 找关键视觉对象 -> 用 SAM/MCTS 对图像做局部增强 -> Qwen-VL 基于增强图像回答
```

后来发现直接改变图像容易造成：

- 裁剪后丢失全局上下文。
- 标记/轮廓突出错误对象。
- A-OKVQA 的答案常依赖常识和场景语义，而不是单个局部区域。

因此后续思路逐渐从“改变图像”转向“给图像加最小必要提示”。

## 主要结果

| 实验 | 准确率 | 正确数 | 主要设置 |
|---|---:|---:|---|
| `mcts_narrow_no_cot_rounds1_n10_6shards` | 59.09% | 676/1145 | no-CoT + narrow MCTS, n=10 |
| `mcts_marker_narrow_no_cot_rounds1_n20_2shards` | 58.97% | 675/1145 | marker + narrow, n=20 |
| `mcts_marker_narrow_no_cot_rounds1_n10_4shards` | 58.76% | 672/1145 | marker + narrow, n=10 |
| `mcts_narrow_no_cot_rounds1_n5_6shards` | 58.72% | 672/1145 | narrow MCTS, n=5 |
| `mcts_marker_narrow_no_cot_rounds1_n5_2shards` | 58.72% | 672/1145 | marker + narrow, n=5 |
| `mcts_narrow_no_cot_rounds1_n20_12shards` | 58.66% | 671/1145 | narrow MCTS, n=20 |
| `mcts_marker_visual_no_cot_rounds1_n10_2shards` | 58.59% | 670/1145 | marker + visual trigger |
| `mcts_safe_no_cot_rounds1_n5_6shards` | 57.48% | 658/1145 | safe MCTS |
| `mcts_image_no_cot_rounds1_n5_4shards` | 53.92% | 617/1145 | 原始图像增强，n=5 |

对比基线：

| 基线 | 准确率 |
|---|---:|
| `no_cot_rounds1` | 59.24% |
| `direct_image_question_only` | 58.54% |
| `clean_rounds1` | 55.33% |

## 搜索数 n_simulations 的影响

| 设置 | 准确率 |
|---|---:|
| marker narrow n=1 | 58.60% |
| marker narrow n=5 | 58.72% |
| marker narrow n=10 | 58.76% |
| marker narrow n=20 | 58.97% |

判断：

- n 从 1 到 20 有轻微上升，但幅度不大。
- 增加搜索数不能保证超过 direct no-CoT。
- 由于 MCTS 很慢，n=20 的性价比不高。

## Action Mode 的影响

| 设置 | 准确率 | 判断 |
|---|---:|---|
| narrow no-CoT n=10 | 59.09% | 最好 MCTS 结果 |
| marker narrow n=10 | 58.76% | 稍低 |
| marker visual n=10 | 58.59% | 接近 image-question-only |
| safe n=5 | 57.48% | 过于保守或策略不匹配 |
| 原始 image enhance n=5 | 53.92% | 明显失败 |

主要结论：

- 原始 MCTS 改图像的动作集合太粗，容易伤害结果。
- narrow/marker 这类“更少改变图像”的策略明显更好。
- 但它们仍主要是在追平 direct，而不是稳定超过 direct。

## 与 CoT 结合

| 实验 | 准确率 | 说明 |
|---|---:|---|
| `compact_marker_mcts_rounds1_n10` | 58.40% | compact CoT + marker MCTS |
| `compact_outline_mcts_rounds1_n10` | 58.36% | compact CoT + outline MCTS |
| `conflict_marker_mcts_rounds1_n5` | 58.90% | conflict CoT + marker MCTS, n=5 |
| `conflict_marker_mcts_rounds1_n10` | 58.72% | conflict CoT + marker MCTS, n=10 |
| `conflict_marker_mcts_rounds1_n20` | 58.79% | conflict CoT + marker MCTS, n=20 |
| `conflict_outline_mcts_rounds1_n10` | 58.66% | conflict CoT + outline MCTS |

判断：

- MCTS 能把较弱 CoT 拉到 58% 左右，但没有超过 direct best。
- conflict-only 比 compact 更稳定，说明 MCTS 作为证据更适合被保守 reviewer 使用，而不是让 CoT 自由生成答案。

## 实验操作说明

主要参数：

| 参数 | 作用 |
|---|---|
| `--use_image_enhance` | 开启图像增强 |
| `--mcts_n_simulations` | MCTS 搜索次数 |
| `--mcts_action_mode` | 控制动作集合，如 marker/outline/no_crop |
| `--mcts_trigger_mode` | 控制哪些问题触发 MCTS |
| `--mcts_filter_objects` | 过滤泛化目标，尽量对齐 scene-graph object |

## 总结

MCTS 当前最好的定位不是主答案生成路径，而是证据模块。它的风险在于直接改图像会把 Qwen-VL 的强项，也就是全图理解和常识推断，压缩到局部对象识别。后续如果继续做 MCTS，建议保持以下原则：

1. 不裁剪或尽量少裁剪。
2. 只添加轻量提示，不改变图像语义。
3. MCTS 输出进入 reviewer evidence，而不是直接替代原图答案。
4. 使用 conflict-only 策略，只有强矛盾证据才改答案。
