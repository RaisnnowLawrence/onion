# marker_n10 vs outline_n10

## Overall

| System | Score Sum | Accuracy |
| --- | ---: | ---: |
| outline_n10 | 676.6/1145 | 59.09% |
| marker_n10 | 672.8/1145 | 58.76% |
| Delta (marker_n10 - outline_n10) | -3.8 | -0.33 |

## Pairwise Buckets

| Bucket | Count |
| --- | ---: |
| marker_n10 better | 26 |
| outline_n10 better | 34 |
| same score | 1085 |

## By Question Type

| Type | Samples | marker_n10 Enhanced | Delta Sum | marker_n10 Better | outline_n10 Better |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 575 | 26 | -3.0 | 6 | 14 |
| what_type_kind | 175 | 0 | -2.8 | 6 | 8 |
| activity_event | 94 | 0 | -1.0 | 3 | 4 |
| how_many | 33 | 30 | -0.7 | 1 | 1 |
| what_object | 21 | 20 | +0.0 | 2 | 2 |
| color | 20 | 10 | +0.0 | 0 | 0 |
| which | 65 | 0 | +0.2 | 2 | 3 |
| why | 61 | 0 | +0.4 | 1 | 1 |
| text_sign_logo | 47 | 0 | +0.7 | 2 | 1 |
| where | 54 | 0 | +2.4 | 3 | 0 |

## Strongest marker_n10 Wins

| ID | Type | outline_n10 | marker_n10 | outline_n10 Pred | marker_n10 Pred | marker_n10 Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 56 | which | 0.0 | 1.0 | new york city | new york | False | The bus is likely driving through which American city? |
| 630 | where | 0.0 | 1.0 | in bathroom | bathroom | False | Where is the photographer standing? |
| 722 | what_type_kind | 0.0 | 1.0 | living room | kitchen | False | The room with the refrigerator in it appears to be a room of what type? |
| 821 | other | 0.0 | 1.0 | sheep | cats | True | What animals are in the grass behind the woman in yellow? |
| 970 | where | 0.0 | 1.0 | grass covered field | field | False | Where are these zebras located? |
| 733 | text_sign_logo | 0.0 | 0.9 | 18 | 16 | False | The front of the dog is closest to what number on the scale? |
| 882 | other | 0.0 | 0.9 | formal | high | False | The positioning of the horses suggests what level of formality? |
| 136 | what_object | 0.3 | 1.0 | umbrella | laptop | True | What object should never get wet? |
| 403 | text_sign_logo | 0.3 | 1.0 | brand | wilson | False | What does the letter on the racket represent? |
| 496 | what_type_kind | 0.3 | 1.0 | restroom | bathroom | False | This man is likely taking a photo in what type of location? |
| 540 | why | 0.3 | 1.0 | rain | raining | False | Why is the woman holding an umbrella while seated? |
| 614 | other | 0.3 | 1.0 | cell phone | phone | False | What is the woman holding to her ear? |
| 8 | activity_event | 0.0 | 0.6 | riding wave | crouching | False | What is the person on the left doing with their body? |
| 433 | other | 0.0 | 0.6 | baseballs | gloves | False | What sports equipment is on the ground? |
| 591 | other | 0.0 | 0.6 | mist | clouds | False | What is hiding the bridge? |

## Strongest marker_n10 Losses

| ID | Type | outline_n10 | marker_n10 | outline_n10 Pred | marker_n10 Pred | marker_n10 Enhanced | Question |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 122 | what_type_kind | 1.0 | 0.0 | vegetarian | vegan | False | A person following what kind of diet is least likely to eat this meal? |
| 194 | how_many | 1.0 | 0.0 | four | 4 | True | How many women are in the picture? |
| 211 | other | 1.0 | 0.0 | jackets | jacket | True | What is hanging on the right wall next the desk? |
| 297 | what_type_kind | 1.0 | 0.0 | forehand | overhand | False | What type of shot is the woman about to hit? |
| 301 | activity_event | 1.0 | 0.0 | travel | carry | False | The bag which the cat is standing is used for what? |
| 516 | what_type_kind | 1.0 | 0.0 | airport | flughafen | False | According to the graphic on the sign what kind of place is nearby? |
| 998 | what_object | 1.0 | 0.0 | shirt | jacket | True | What item of visible clothing is red? |
| 54 | what_type_kind | 0.9 | 0.0 | party | social gathering | False | They are likely having pizza at what kind of event? |
| 339 | what_type_kind | 0.9 | 0.0 | dirt | gravel | False | What type of terrain is beyond the table and grill? |
| 379 | text_sign_logo | 0.9 | 0.0 | eight | seven | False | What number is closest to the number of people that are pushing the bus? |
| 532 | other | 0.9 | 0.0 | snowboard | surfboard | True | What is on top of the car? |
| 665 | activity_event | 0.9 | 0.0 | plowing | plow | False | What work is the team of horses doing? |
| 455 | other | 1.0 | 0.3 | wagon | cart | False | What vehicle might the thing on the wall have come from? |
| 1054 | what_type_kind | 1.0 | 0.3 | pizza party | party | False | What type of event is happening? |
| 189 | other | 0.6 | 0.0 | kimono | traditional | False | What cultural clothing are the women wearing? |
