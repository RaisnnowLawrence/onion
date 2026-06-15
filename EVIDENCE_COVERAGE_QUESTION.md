# Evidence Coverage Question

Last updated: 2026-06-14

## Core Question

在开放式 VQA 设置下，是否可能出现这样一种失败模式：

> 我们提供给 LLM 的多种证据都没有注意到某个关键答案方向，导致最终 prompt 里的线索缺少正确答案所需的信息，从而使 LLM 几乎不可能回答出正确答案？

换句话说，虽然当前实验不是选择题模式，最终没有给模型显式答案候选，但我们的 evidence pipeline 可能仍然在隐式地限制模型注意力：

- VinVL caption 可能没有描述关键物体或属性。
- scene graph 可能没有检测到关键对象。
- attention object selection 可能选错关注物体。
- caption enhancement 可能只围绕错误对象展开。
- knowledge enhancement 可能补充了无关知识。
- MCTS/SAM 可能标记或强调了错误区域。
- reviewer evidence 可能只看到这些不完整证据，因此只能在错误证据空间内复核。

如果正确答案依赖的视觉对象、属性、文本、关系或常识没有进入这些证据，LLM 可能会被迫根据不完整上下文猜测。

## Why This Matters

当前实验观察显示：

- Direct/no-CoT 很强，说明模型原始视觉直觉有价值。
- 传统 CoT 和过多 caption/context 经常伤害准确率。
- reviewer/reflective 方法更稳，但提升有限。
- MCTS/SAM 图像增强不稳定，可能强调了错误区域。
- Qwen-generated caption 注入 final prompt 会显著掉点。

这些结果共同提示一个可能问题：增强模块不一定总是在“补充正确证据”，有时可能是在缩窄或扭曲模型的注意力空间。

## Hypothesis

部分错误样本不是因为 LLM 不会答，而是因为 evidence construction 阶段没有覆盖正确答案所需的信息。

可以把这种错误称为：

```text
evidence coverage failure
```

它和普通 reasoning failure 不同：

- reasoning failure：正确线索已经在图像/上下文里，但模型推理或抽答案错了。
- evidence coverage failure：正确线索没有进入最终证据链，模型被给到的是缺失或偏置的线索。

## Possible Diagnostics

后续可以做逐题分析：

1. 找出 direct image+question 答对、但 evidence-heavy 方法答错的样本。
2. 检查这些样本中，正确答案对应的物体/属性是否出现在：
   - VinVL caption
   - scene graph object list
   - selected attention objects
   - regional captions
   - reviewer evidence
   - MCTS marked/outlined objects
3. 统计正确答案相关对象没有被选中的比例。
4. 比较 no-caption / no-round-state / image-question-only 与 evidence-heavy 方法的互补样本。
5. 对错误样本手动标注失败类型：
   - missing object
   - missing attribute
   - missing OCR text
   - wrong selected object
   - wrong region focus
   - misleading caption
   - over-revision by reviewer

## Implication For Next Code Changes

如果这个假设成立，后续不应该只增强 reviewer 或 CoT prompt，而应该加入 evidence coverage 机制，例如：

- 保留一个 image-question-only direct candidate，避免 evidence pipeline 完全遮蔽原始视觉直觉。
- 在候选答案冲突时，让 judge 同时看到原图，而不是只看文本 evidence。
- 对 attention object selection 做 recall-oriented 设计，宁可多保留一些对象，不要过早收窄。
- 对 high-risk 问题触发 broader regional context。
- 加入 candidate answer generation，让不同上下文产生多个答案，再做保守仲裁。
- 记录每题 evidence coverage metadata，方便分析正确答案是否曾进入证据链。

## Open Research Note

这个问题直接关系到后续能否把准确率从约 60% 提升到更高：

如果许多错误来自 evidence coverage failure，那么继续堆 CoT 或 reviewer 可能收益有限；更关键的是让系统在回答前拥有更高召回率的证据空间。

