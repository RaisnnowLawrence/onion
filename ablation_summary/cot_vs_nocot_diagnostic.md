# CoT vs no-CoT Per-sample Diagnostic

## Runs Compared

| Run | Setting | Score Sum | Accuracy |
| --- | --- | ---: | ---: |
| nocot_r1 | no CoT, rounds=1 | 678.3/1145 | 59.24% |
| cot_r1 | CoT, rounds=1 clean rerun | 633.5/1145 | 55.33% |
| cot_r5 | CoT baseline, rounds=5 | 603.7/1145 | 52.72% |

## Pairwise Buckets

| Comparison | First Better | Second Better | Same |
| --- | ---: | ---: | ---: |
| nocot_r1 vs cot_r1 | 125 | 74 | 946 |
| nocot_r1 vs cot_r5 | 169 | 82 | 894 |
| cot_r1 vs cot_r5 | 107 | 69 | 969 |

## By Question Type

| Type | Samples | cot_r1 - nocot | cot_r5 - nocot | cot_r5 - cot_r1 | noCoT > cot1 | cot1 > noCoT | cot1 > cot5 | cot5 > cot1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 575 | -20.7 | -41.0 | -20.3 | 63 | 43 | 51 | 34 |
| why | 61 | -8.8 | -11.7 | -2.9 | 13 | 0 | 6 | 3 |
| what_type_kind | 175 | -5.1 | -4.8 | +0.3 | 16 | 8 | 16 | 14 |
| where | 54 | -3.1 | -1.9 | +1.2 | 5 | 1 | 7 | 6 |
| text_sign_logo | 47 | -2.4 | -1.9 | +0.5 | 5 | 1 | 2 | 4 |
| which | 65 | -1.9 | -4.1 | -2.2 | 8 | 6 | 9 | 3 |
| what_object | 21 | -1.6 | +0.7 | +2.3 | 2 | 1 | 0 | 3 |
| color | 20 | -1.3 | -3.2 | -1.9 | 2 | 1 | 3 | 0 |
| how_many | 33 | +0.0 | -1.0 | -1.0 | 0 | 0 | 1 | 0 |
| activity_event | 94 | +0.1 | -5.7 | -5.8 | 11 | 13 | 12 | 2 |

## Answer Length

| Run | Avg predicted answer words |
| --- | ---: |
| nocot_r1 | 1.21 |
| cot_r1 | 1.44 |
| cot_r5 | 1.84 |

## CoT Instruction Distribution

| Run | Instruction counts |
| --- | --- |
| cot_r1 | {'': 1145} |
| cot_r5 | {'': 1145} |

## no-CoT Correct / CoT-r1 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 49 | why | 1.0 | 0.0 | 0.0 | to eat | To eat the pastries and drinks. | To enjoy a meal or snack. | Why would you sit at this table? |
| 78 | other | 1.0 | 0.0 | 0.0 | phone | cell phone | cell phone | What is the man in the gray suit on the left looking down to check? |
| 99 | where | 1.0 | 0.0 | 1.0 | chair | in a room | chair | Where is this dog located? |
| 110 | other | 1.0 | 0.0 | 1.0 | triangle | triangular | triangle | In what shape is the quiche cut? |
| 128 | other | 1.0 | 0.0 | 0.0 | cold | cool | cool | The temperature outside is likely what range? |
| 174 | other | 1.0 | 0.0 | 0.0 | cat | rabbit | rabbit | What other animal is this animal traditionally an enemy of? |
| 211 | other | 1.0 | 0.0 | 1.0 | jackets | clothes | jackets | What is hanging on the right wall next the desk? |
| 229 | other | 1.0 | 0.0 | 0.0 | string | hands | hands | What is needed to control the large object? |
| 246 | what_type_kind | 1.0 | 0.0 | 1.0 | kitchen | residential | kitchen | What type of region is this likely? |
| 272 | other | 1.0 | 0.0 | 0.0 | weight | weight gain | weight gain | What do people gain when they eat too many of these? |
| 274 | activity_event | 1.0 | 0.0 | 1.0 | singing | speaking into microphone | singing | What is the man on TV doing? |
| 281 | other | 1.0 | 0.0 | 0.0 | uniform | blue shirt with uniform insignia | She is wearing a uniform. | How do we know this woman is at work? |

## CoT-r1 Correct / no-CoT Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 70 | other | 0.0 | 1.0 | 0.0 | counter | island | bar stool | What is the object in the middle called? |
| 198 | activity_event | 0.0 | 1.0 | 1.0 | plow | plowing | plowing | What are the horses being used for? |
| 287 | other | 0.0 | 1.0 | 0.0 | drawer | desk | drawer | What does the cat have his paw up against? |
| 313 | what_type_kind | 0.0 | 1.0 | 0.0 | labrador | german shepherd | Cannot determine | What type of dog is it? |
| 366 | text_sign_logo | 0.0 | 1.0 | 1.0 | laptop | apple | Apple | What brand of electronics is the person using on the desk? |
| 463 | other | 0.0 | 1.0 | 1.0 | skiing | snow | snow | What is required to be on the ground to do this sport? |
| 642 | which | 0.0 | 1.0 | 1.0 | duck | Donald Duck | Donald Duck | Which Disney mascot is pictured next to the Christmas Tree? |
| 741 | activity_event | 0.0 | 1.0 | 1.0 | washing | brushing teeth | brushing teeth | What is the man with no shirt doing? |
| 806 | activity_event | 0.0 | 1.0 | 0.6 | print | printing | printer | What is the device on the right filled with paper used for? |
| 831 | other | 0.0 | 1.0 | 0.0 | photographer | photographers | photographer | What profession is the majority of the sideline? |
| 942 | other | 0.0 | 1.0 | 0.0 | hoof | hooves | hoof | What are the horses feet called? |
| 964 | other | 0.0 | 1.0 | 0.0 | produce market | produce | produce market | What section are the people in? |

## CoT-r1 Correct / CoT-r5 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 18 | activity_event | 1.0 | 1.0 | 0.0 | dinner | dinner | dinner party | What event is this most likely? |
| 70 | other | 0.0 | 1.0 | 0.0 | counter | island | bar stool | What is the object in the middle called? |
| 208 | other | 1.0 | 1.0 | 0.0 | eat | eat | take them home | What do people do with the items purchased at this corner? |
| 287 | other | 0.0 | 1.0 | 0.0 | drawer | desk | drawer | What does the cat have his paw up against? |
| 309 | what_type_kind | 1.0 | 1.0 | 0.0 | shoes | shoes | shoe | What kind of footwear is the walker wearing? |
| 313 | what_type_kind | 0.0 | 1.0 | 0.0 | labrador | german shepherd | Cannot determine | What type of dog is it? |
| 391 | why | 1.0 | 1.0 | 0.0 | balance | balance | balancing | Why is the man in the green shirt holding his arms out? |
| 462 | why | 1.0 | 1.0 | 0.0 | safety | safety | For safety while riding a horse. | Why is the kid wearing a helmet? |
| 468 | text_sign_logo | 1.0 | 1.0 | 0.0 | was | WAS | ONE | What is the first visible red word? |
| 483 | other | 1.0 | 1.0 | 0.0 | soda | soda | The answer is soda. | The paper cups probably contain what beverage? |
| 510 | where | 1.0 | 1.0 | 0.0 | kitchen | kitchen | not specified | Where is this room located at? |
| 514 | how_many | 1.0 | 1.0 | 0.0 | one | one | 1 | How many people can likely eat one cake? |

## CoT-r5 Correct / CoT-r1 Wrong Examples

| ID | Type | noCoT | CoT1 | CoT5 | noCoT Pred | CoT1 Pred | CoT5 Pred | Question |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 99 | where | 1.0 | 0.0 | 1.0 | chair | in a room | chair | Where is this dog located? |
| 110 | other | 1.0 | 0.0 | 1.0 | triangle | triangular | triangle | In what shape is the quiche cut? |
| 211 | other | 1.0 | 0.0 | 1.0 | jackets | clothes | jackets | What is hanging on the right wall next the desk? |
| 246 | what_type_kind | 1.0 | 0.0 | 1.0 | kitchen | residential | kitchen | What type of region is this likely? |
| 274 | activity_event | 1.0 | 0.0 | 1.0 | singing | speaking into microphone | singing | What is the man on TV doing? |
| 332 | other | 0.0 | 0.0 | 1.0 | van | van | bicycle | What is the vehicle in front of the cars? |
| 395 | what_type_kind | 1.0 | 0.0 | 1.0 | fisheye | wide-angle | fisheye | What type of lens was used to give this particular photo the distorted look? |
| 584 | what_type_kind | 0.0 | 0.0 | 1.0 | cattle | cattle | cows | What type of animal is in the grass? |
| 585 | other | 1.0 | 0.0 | 1.0 | table | wooden table | table | What is the laptop on? |
| 625 | what_object | 0.0 | 0.0 | 1.0 | bird | bird | swan | What animal is near the water? |
| 630 | where | 0.0 | 0.0 | 1.0 | in bathroom | in bathroom | bathroom | Where is the photographer standing? |
| 970 | where | 0.0 | 0.0 | 1.0 | grass covered field | grass covered field | field | Where are these zebras located? |

## Chinese Interpretation

主要结论：

1. no-CoT rounds=1 比 CoT rounds=1 高 45 分左右，说明显式 CoT 本身会带来明显损失。
2. CoT rounds=5 又比 CoT rounds=1 低 30 分左右，说明多轮上下文/状态继续带来额外损失。
3. 因此 CoT 变差有两层原因：第一层是显式推理让模型更容易发散或被中间结论带偏；第二层是多轮 ONION 上下文会积累噪声。
4. 题型表里 `cot_r1 - nocot` 越负，说明该类题越不适合显式 CoT；`cot_r5 - cot_r1` 越负，说明该类题越容易被多轮上下文伤害。

建议：

- 最终主线继续使用 direct no-CoT。
- 如果要保留 CoT，应使用很短的结构化 visual cues，而不是完整推理链。
- CoT 更适合作为低置信样本的 verifier，而不是默认 generator。
