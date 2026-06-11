# Safe MCTS vs Best no-CoT Diagnostic

## Overall

| System | Score Sum | Accuracy |
| --- | ---: | ---: |
| best no-CoT rounds1 | 678.3/1145 | 59.24% |
| safe MCTS n=5 | 658.1/1145 | 57.48% |
| Delta | -20.2 | -1.76 |

## Pairwise Buckets

| Bucket | Count |
| --- | ---: |
| MCTS better | 38 |
| best no-CoT better | 66 |
| same score | 1041 |

## Enhancement Split

| Split | Samples | Delta Sum | Avg Delta |
| --- | ---: | ---: | ---: |
| MCTS actually enhanced | 274 | -14.3 | -0.052 |
| MCTS not enhanced/skipped | 871 | -5.9 | -0.007 |

## By Question Type

| Type | Samples | Enhanced | Delta Sum | MCTS Better | Best Better |
| --- | ---: | ---: | ---: | ---: | ---: |
| what_type_kind | 175 | 108 | -12.8 | 6 | 23 |
| other | 575 | 26 | -7.0 | 16 | 20 |
| activity_event | 94 | 0 | -1.9 | 1 | 3 |
| which | 65 | 64 | -1.2 | 6 | 11 |
| why | 61 | 0 | -0.5 | 1 | 3 |
| text_sign_logo | 47 | 19 | -0.4 | 0 | 2 |
| where | 54 | 0 | +0.0 | 1 | 2 |
| color | 20 | 12 | +0.4 | 1 | 1 |
| what_object | 21 | 14 | +0.9 | 3 | 1 |
| how_many | 33 | 31 | +2.3 | 3 | 0 |

## Strongest MCTS Wins

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 56 | which | 0.0 | 1.0 | new york city | New York | True | The bus is likely driving through which American city? |
| 125 | which | 0.0 | 1.0 | broccoli | carrot | True | Which food on the plate grows in the ground? |
| 194 | how_many | 0.0 | 1.0 | 4 | four | True | How many women are in the picture? |
| 267 | how_many | 0.0 | 1.0 | 1 | one | True | How many vehicles have their lights on? |
| 287 | other | 0.0 | 1.0 | drawer | desk | False | What does the cat have his paw up against? |
| 316 | color | 0.0 | 1.0 | blue | white | True | The most abundant cake has a topping with which color? |
| 516 | what_type_kind | 0.0 | 1.0 | flughafen | airport | False | According to the graphic on the sign what kind of place is nearby? |
| 601 | where | 0.0 | 1.0 | autumn | summer | False | What season is it on the grassland where the elephants are grazing? |
| 642 | which | 0.0 | 1.0 | duck | Donald Duck | True | Which Disney mascot is pictured next to the Christmas Tree? |
| 686 | what_type_kind | 0.0 | 1.0 | coca-cola | coca cola | False | What kind of beverage is the red sign advertising? |
| 339 | what_type_kind | 0.0 | 0.9 | gravel | dirt | True | What type of terrain is beyond the table and grill? |
| 136 | what_object | 0.3 | 1.0 | umbrella | laptop | True | What object should never get wet? |

## Strongest MCTS Losses

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 22 | which | 1.0 | 0.0 | human | cat | True | Which animal usually occupies the position the cat is in right now? |
| 122 | what_type_kind | 1.0 | 0.0 | vegetarian | vegan | False | A person following what kind of diet is least likely to eat this meal? |
| 174 | other | 1.0 | 0.0 | cat | rat | False | What other animal is this animal traditionally an enemy of? |
| 183 | what_type_kind | 1.0 | 0.0 | bus | car | True | What type of vehicle is this? |
| 246 | what_type_kind | 1.0 | 0.0 | kitchen | person | True | What type of region is this likely? |
| 297 | what_type_kind | 1.0 | 0.0 | forehand | overhand shot | True | What type of shot is the woman about to hit? |
| 301 | activity_event | 1.0 | 0.0 | travel | carry | False | The bag which the cat is standing is used for what? |
| 304 | what_type_kind | 1.0 | 0.0 | beer | water | True | What type of bottles are on the table? |
| 360 | other | 1.0 | 0.0 | cows | cattle | True | What animals are in the field? |
| 471 | what_type_kind | 1.0 | 0.0 | rottweiler | black dog | True | What type of dog is it? |
| 485 | what_type_kind | 1.0 | 0.0 | serve | forehand | True | What type of shot is the woman about to hit? |
| 488 | other | 1.0 | 0.0 | tourists | passengers | False | What are the people on the sidewalk likely to be? |

## Enhanced-only Wins

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 56 | which | 0.0 | 1.0 | new york city | New York | True | The bus is likely driving through which American city? |
| 125 | which | 0.0 | 1.0 | broccoli | carrot | True | Which food on the plate grows in the ground? |
| 194 | how_many | 0.0 | 1.0 | 4 | four | True | How many women are in the picture? |
| 267 | how_many | 0.0 | 1.0 | 1 | one | True | How many vehicles have their lights on? |
| 316 | color | 0.0 | 1.0 | blue | white | True | The most abundant cake has a topping with which color? |
| 642 | which | 0.0 | 1.0 | duck | Donald Duck | True | Which Disney mascot is pictured next to the Christmas Tree? |
| 339 | what_type_kind | 0.0 | 0.9 | gravel | dirt | True | What type of terrain is beyond the table and grill? |
| 136 | what_object | 0.3 | 1.0 | umbrella | laptop | True | What object should never get wet? |
| 145 | what_object | 0.0 | 0.6 | remote | speaker | True | What item is on the bottom shelf near the TV? |
| 220 | which | 0.0 | 0.6 | red car | car | True | Which vehicle is closest to the transport hub? |
| 412 | other | 0.0 | 0.6 | purse | nothing | True | What is hanging from the woman's back pocket? |
| 476 | which | 0.6 | 1.0 | vegetarian | vegetarians | True | This dish is suitable for which group of people? |

## Enhanced-only Losses

| ID | Type | Best | MCTS | Best Pred | MCTS Pred | Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 22 | which | 1.0 | 0.0 | human | cat | True | Which animal usually occupies the position the cat is in right now? |
| 183 | what_type_kind | 1.0 | 0.0 | bus | car | True | What type of vehicle is this? |
| 246 | what_type_kind | 1.0 | 0.0 | kitchen | person | True | What type of region is this likely? |
| 297 | what_type_kind | 1.0 | 0.0 | forehand | overhand shot | True | What type of shot is the woman about to hit? |
| 304 | what_type_kind | 1.0 | 0.0 | beer | water | True | What type of bottles are on the table? |
| 360 | other | 1.0 | 0.0 | cows | cattle | True | What animals are in the field? |
| 471 | what_type_kind | 1.0 | 0.0 | rottweiler | black dog | True | What type of dog is it? |
| 485 | what_type_kind | 1.0 | 0.0 | serve | forehand | True | What type of shot is the woman about to hit? |
| 525 | what_type_kind | 1.0 | 0.0 | tennis | tennis racket | True | What type of racket is next to the mans head? |
| 588 | what_type_kind | 1.0 | 0.0 | meat | food | True | The reddish-brown food in the further bowl is what type of food? |
| 705 | which | 1.0 | 0.0 | tusks | trunk | True | Which part of the animals is/are precious? |
| 821 | other | 1.0 | 0.0 | cats | sheep | True | What animals are in the grass behind the woman in yellow? |

## Chinese Takeaway

Safe MCTS 比原始 MCTS 明显更好，但和 best no-CoT 相比仍然净亏。关键不是搜索次数不够，而是增强触发后仍会在一部分问题上改变模型答案，带来额外错误。

最重要的诊断看两处：

- `MCTS actually enhanced` 的净 delta：如果这里为负，说明真正执行 MCTS 的题整体仍在伤害结果。
- `MCTS not enhanced/skipped` 的净 delta：如果这里也有差异，说明即使没有增强，运行随机性、缓存或 prompt 路径也会造成回答波动。

下一步优先改路由和 reward，而不是先盲目提高 `n_simulations`。
