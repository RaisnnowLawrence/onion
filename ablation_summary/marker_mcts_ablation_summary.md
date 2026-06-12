# Marker-only MCTS Ablation Summary

## Results

| Experiment | Trigger | Action | n_simulations | Accuracy | Correct |
| --- | --- | --- | ---: | ---: | ---: |
| no_cot_rounds1 | none | none | none | 59.24% | 678/1145 |
| outline_narrow_n10 | count_color_object_only | outline_only | 10 | 59.09% | 676/1145 |
| marker_narrow_n20 | count_color_object_only | marker_only | 20 | 58.97% | 675/1145 |
| marker_narrow_n10 | count_color_object_only | marker_only | 10 | 58.76% | 672/1145 |
| marker_narrow_n5 | count_color_object_only | marker_only | 5 | 58.72% | 672/1145 |
| marker_narrow_n1 | count_color_object_only | marker_only | 1 | 58.60% | 670/1145 |
| marker_visual_n10 | visual_detail_only | marker_only | 10 | 58.59% | 670/1145 |

## Takeaways

Marker-only actions did not outperform outline-only actions in this run.

- Best marker result: `marker_narrow_n20`, 58.97% (675/1145).
- Best MCTS result so far: `outline_narrow_n10`, 59.09% (676/1145).
- Best overall result remains `no_cot_rounds1`, 59.24% (678/1145).

The marker action is safer than broad/old MCTS, but it does not recover enough extra correct answers to beat the outline n=10 setting or the direct no-CoT baseline.

## Marker vs Outline n=10

The automatic diagnostic compares `marker_n10` against `outline_n10`.

| Comparison | Count |
| --- | ---: |
| marker_n10 better | 26 |
| outline_n10 better | 34 |
| same score | 1085 |

Score delta: marker is -3.8 score sum, or -0.33 points, versus outline n=10.

## 中文结论

这版 marker-only 的核心假设是“少改图像，只给模型最小定位提示”。从结果看，这个方向没有明显失败，但也没有超过 outline-only。

最好的 marker 配置是 `marker_narrow_n20`，准确率 58.97%，比 `outline_narrow_n10` 的 59.09% 低 1 题，比最佳 `no_cot_rounds1` 的 59.24% 低 3 题。

因此目前不建议把 marker-only 作为默认主结果。它可以作为一个合理的负结果/诊断结果：更轻量的视觉提示并没有自动带来更高准确率，MCTS 的主要瓶颈可能仍在对象选择、触发路由和 reward，而不只是图像改动幅度。
