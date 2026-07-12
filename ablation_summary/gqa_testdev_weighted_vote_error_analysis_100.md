# GQA testdev weighted_vote Error Analysis - 100 Examples

| # | idx | image_path | question | correct_answer | wrong_answer |
|---:|---:|---|---|---|---|
| 1 | 1 | `/data2/lizhengxue/datasets/gqa/images/n235859.jpg` | Who is wearing the dress? | women | woman |
| 2 | 4 | `/data2/lizhengxue/datasets/gqa/images/n518912.jpg` | How tall is the chair in the bottom of the photo? | short | 3 feet |
| 3 | 5 | `/data2/lizhengxue/datasets/gqa/images/n435808.jpg` | What kind of device is on top of the desk? | keyboard | monitor |
| 4 | 9 | `/data2/lizhengxue/datasets/gqa/images/n23181.jpg` | What is around the open window? | drapes | curtains |
| 5 | 10 | `/data2/lizhengxue/datasets/gqa/images/n23181.jpg` | What's around the window? | drapes | curtains |
| 6 | 11 | `/data2/lizhengxue/datasets/gqa/images/n52544.jpg` | Who is standing at the table? | woman | people |
| 7 | 14 | `/data2/lizhengxue/datasets/gqa/images/n437064.jpg` | Is the cake on a platter? | no | yes |
| 8 | 16 | `/data2/lizhengxue/datasets/gqa/images/n435808.jpg` | What device is sitting next to the mouse pad? | keyboard | remote |
| 9 | 23 | `/data2/lizhengxue/datasets/gqa/images/n578564.jpg` | What is the name of the cooking utensil that is hang from the hook? | pan | pot |
| 10 | 24 | `/data2/lizhengxue/datasets/gqa/images/n52544.jpg` | Where is the skinny person standing? | table | doorway |
| 11 | 26 | `/data2/lizhengxue/datasets/gqa/images/n579256.jpg` | Is the freezer near the wall small or large? | large | small |
| 12 | 27 | `/data2/lizhengxue/datasets/gqa/images/n546616.jpg` | What type of food is to the left of the baby that is sitting atop the woman? | marshmallow | dessert |
| 13 | 30 | `/data2/lizhengxue/datasets/gqa/images/n315887.jpg` | Are both the phone and the coffee cup the same color? | yes | no |
| 14 | 33 | `/data2/lizhengxue/datasets/gqa/images/n293477.jpg` | What color is the book? | white | black |
| 15 | 34 | `/data2/lizhengxue/datasets/gqa/images/n130638.jpg` | What color is the dirt? | red | brown |
| 16 | 36 | `/data2/lizhengxue/datasets/gqa/images/n23181.jpg` | What are the drapes around of? | window | windows |
| 17 | 43 | `/data2/lizhengxue/datasets/gqa/images/n274905.jpg` | Which kind of clothing is not pink? | hat | shorts |
| 18 | 44 | `/data2/lizhengxue/datasets/gqa/images/n250715.jpg` | Is this helicopter on or off? | off | on |
| 19 | 51 | `/data2/lizhengxue/datasets/gqa/images/n207708.jpg` | What is beneath the microwave? | bananas | counter |
| 20 | 52 | `/data2/lizhengxue/datasets/gqa/images/n511913.jpg` | On which side of the picture is the chair? | right | left |
| 21 | 53 | `/data2/lizhengxue/datasets/gqa/images/n71728.jpg` | Is the happy man to the left or to the right of the woman in the center? | right | left |
| 22 | 57 | `/data2/lizhengxue/datasets/gqa/images/n435808.jpg` | On which side is the router? | left | right |
| 23 | 58 | `/data2/lizhengxue/datasets/gqa/images/n518912.jpg` | What is the color of the pants? | gray | black |
| 24 | 59 | `/data2/lizhengxue/datasets/gqa/images/n137182.jpg` | Who is wearing the shirt? | girl | woman |
| 25 | 60 | `/data2/lizhengxue/datasets/gqa/images/n137182.jpg` | Who is wearing a shirt? | girl | woman |
| 26 | 64 | `/data2/lizhengxue/datasets/gqa/images/n214497.jpg` | How do the cars look like, dense or sparse? | dense | sparse |
| 27 | 65 | `/data2/lizhengxue/datasets/gqa/images/n296467.jpg` | What food isn't baked? | cookies | carrot |
| 28 | 68 | `/data2/lizhengxue/datasets/gqa/images/n275148.jpg` | What is hanging from the wall? | picture frame | picture |
| 29 | 69 | `/data2/lizhengxue/datasets/gqa/images/n130464.jpg` | What's the skateboarder jumping off of? | pavement | train |
| 30 | 74 | `/data2/lizhengxue/datasets/gqa/images/n151768.jpg` | Are the boxes to the right of the man full and square? | yes | no |
| 31 | 77 | `/data2/lizhengxue/datasets/gqa/images/n64959.jpg` | What appliance is the refrigerator larger than? | stove | oven |
| 32 | 79 | `/data2/lizhengxue/datasets/gqa/images/n296467.jpg` | What is inside the bowl to the right of the beans? | cookies | rice |
| 33 | 80 | `/data2/lizhengxue/datasets/gqa/images/n317260.jpg` | How clean do you think is the face mask the catcher is wearing? | clean | dirty |
| 34 | 81 | `/data2/lizhengxue/datasets/gqa/images/n570181.jpg` | Where is the catcher standing on? | field | mound |
| 35 | 84 | `/data2/lizhengxue/datasets/gqa/images/n35676.jpg` | What color are the drawers? | light brown | brown |
| 36 | 86 | `/data2/lizhengxue/datasets/gqa/images/n88933.jpg` | Which kind of clothing is bright? | gown | dress |
| 37 | 88 | `/data2/lizhengxue/datasets/gqa/images/n571179.jpg` | What is the woman wearing? | gloves | jacket |
| 38 | 89 | `/data2/lizhengxue/datasets/gqa/images/n571179.jpg` | What do you think is the standing person near the man wearing? | gloves | snowsuit |
| 39 | 90 | `/data2/lizhengxue/datasets/gqa/images/n88933.jpg` | Which type of clothing is pink? | gown | dress |
| 40 | 91 | `/data2/lizhengxue/datasets/gqa/images/n566028.jpg` | What is the person that is sitting down sitting atop? | stairs | steps |
| 41 | 93 | `/data2/lizhengxue/datasets/gqa/images/n531359.jpg` | What items of furniture are to the left of the boy? | tables | table |
| 42 | 94 | `/data2/lizhengxue/datasets/gqa/images/n500209.jpg` | What is in front of the wall that is not short? | shelf | bookshelf |
| 43 | 97 | `/data2/lizhengxue/datasets/gqa/images/n315887.jpg` | What is the device in front of the flat computer? | phone | keyboard |
| 44 | 98 | `/data2/lizhengxue/datasets/gqa/images/n554880.jpg` | What is sitting on the floor? | gift | snowboard |
| 45 | 100 | `/data2/lizhengxue/datasets/gqa/images/n554880.jpg` | Is the gift sitting on the floor? | yes | no |
| 46 | 101 | `/data2/lizhengxue/datasets/gqa/images/n250715.jpg` | Which material makes up the round glasses, glass or wire? | glass | wire |
| 47 | 102 | `/data2/lizhengxue/datasets/gqa/images/n250715.jpg` | What are the glasses made of? | glass | metal |
| 48 | 107 | `/data2/lizhengxue/datasets/gqa/images/n318684.jpg` | What is the man to the left of the glasses doing? | resting | sitting |
| 49 | 109 | `/data2/lizhengxue/datasets/gqa/images/n309148.jpg` | Are there any red fire trucks? | no | yes |
| 50 | 110 | `/data2/lizhengxue/datasets/gqa/images/n208302.jpg` | Which kind of vehicle is waiting for the traffic light? | cars | car |
| 51 | 111 | `/data2/lizhengxue/datasets/gqa/images/n208302.jpg` | What kind of vehicle is waiting for the traffic light? | cars | car |
| 52 | 112 | `/data2/lizhengxue/datasets/gqa/images/n403734.jpg` | The electronic device to the left of the notebook has what color? | blue | green |
| 53 | 114 | `/data2/lizhengxue/datasets/gqa/images/n208302.jpg` | What is waiting for the traffic light? | cars | car |
| 54 | 115 | `/data2/lizhengxue/datasets/gqa/images/n507959.jpg` | What is sitting in front of the table that looks yellow and black? | luggage | backpack |
| 55 | 116 | `/data2/lizhengxue/datasets/gqa/images/n473688.jpg` | Are there both toothbrushes and mats in this picture? | no | yes |
| 56 | 118 | `/data2/lizhengxue/datasets/gqa/images/n208302.jpg` | The parked vehicles are waiting for what? | traffic light | green light |
| 57 | 120 | `/data2/lizhengxue/datasets/gqa/images/n473688.jpg` | The soap dispenser made of chrome is sitting on what? | countertop | sink |
| 58 | 125 | `/data2/lizhengxue/datasets/gqa/images/n342511.jpg` | Who is the jacket worn around? | man | person |
| 59 | 126 | `/data2/lizhengxue/datasets/gqa/images/n351318.jpg` | On which side of the picture are the pens? | right | left |
| 60 | 128 | `/data2/lizhengxue/datasets/gqa/images/n111390.jpg` | What is the item of furniture to the right of the lady that is looking down at the cake called? | table | chair |
| 61 | 129 | `/data2/lizhengxue/datasets/gqa/images/n429883.jpg` | Is the man to the left of the performer brunette or blond? | blond | brunette |
| 62 | 131 | `/data2/lizhengxue/datasets/gqa/images/n471866.jpg` | Is the plastic helmet to the left of a woman? | yes | no |
| 63 | 133 | `/data2/lizhengxue/datasets/gqa/images/n204894.jpg` | Who is wearing jeans? | child | boy |
| 64 | 137 | `/data2/lizhengxue/datasets/gqa/images/n494677.jpg` | Are the trees on the field bare or lush? | lush | bare |
| 65 | 141 | `/data2/lizhengxue/datasets/gqa/images/n67005.jpg` | Is the jacket made of cotton large or small? | small | large |
| 66 | 146 | `/data2/lizhengxue/datasets/gqa/images/n281241.jpg` | What is the picture hanging above? | chair | chairs |
| 67 | 147 | `/data2/lizhengxue/datasets/gqa/images/n281241.jpg` | The framed picture is hanging above what? | chair | chairs |
| 68 | 149 | `/data2/lizhengxue/datasets/gqa/images/n259002.jpg` | Who is running? | soccer player | boys |
| 69 | 150 | `/data2/lizhengxue/datasets/gqa/images/n282436.jpg` | What is the large device called? | keyboard | computer |
| 70 | 151 | `/data2/lizhengxue/datasets/gqa/images/n471866.jpg` | Who is wearing a helmet? | policeman | officer |
| 71 | 152 | `/data2/lizhengxue/datasets/gqa/images/n386682.jpg` | What is beneath the microwave? | dishwasher | cabinet |
| 72 | 154 | `/data2/lizhengxue/datasets/gqa/images/n351318.jpg` | Of what color are the scissors? | gray | black |
| 73 | 159 | `/data2/lizhengxue/datasets/gqa/images/n512257.jpg` | What is the name of the clothing item that is navy? | jacket | suit |
| 74 | 162 | `/data2/lizhengxue/datasets/gqa/images/n546616.jpg` | What kind of food is to the left of the baby? | marshmallow | cake |
| 75 | 164 | `/data2/lizhengxue/datasets/gqa/images/n59147.jpg` | The toilet paper to the right of the toilet is resting on what? | chair | surface |
| 76 | 167 | `/data2/lizhengxue/datasets/gqa/images/n496803.jpg` | What is the person below the crowd bigger than? | sneakers | crowd |
| 77 | 171 | `/data2/lizhengxue/datasets/gqa/images/n352479.jpg` | Who is standing? | snowboarder | woman |
| 78 | 172 | `/data2/lizhengxue/datasets/gqa/images/n181355.jpg` | What piece of furniture is made of wood? | coffee table | table |
| 79 | 174 | `/data2/lizhengxue/datasets/gqa/images/n181355.jpg` | How the piece of furniture that is made of wood is called? | coffee table | table |
| 80 | 175 | `/data2/lizhengxue/datasets/gqa/images/n259002.jpg` | Who is looking up? | spectator | boy in pink |
| 81 | 176 | `/data2/lizhengxue/datasets/gqa/images/n88933.jpg` | Which kind of furniture is blue? | sofa | bench |
| 82 | 177 | `/data2/lizhengxue/datasets/gqa/images/n513429.jpg` | What is that monitor in front of? | poster | laptop |
| 83 | 179 | `/data2/lizhengxue/datasets/gqa/images/n259002.jpg` | What do you think is that spectator doing? | looking up | clapping |
| 84 | 180 | `/data2/lizhengxue/datasets/gqa/images/n88933.jpg` | What piece of furniture is it? | sofa | bench |
| 85 | 181 | `/data2/lizhengxue/datasets/gqa/images/n500209.jpg` | What is the container made of glass sitting on top of? | shelf | wooden stand |
| 86 | 182 | `/data2/lizhengxue/datasets/gqa/images/n433532.jpg` | What is the name of the smooth piece of clothing? | robe | coat |
| 87 | 184 | `/data2/lizhengxue/datasets/gqa/images/n557666.jpg` | What color is the shirt the woman wears? | white | plaid |
| 88 | 185 | `/data2/lizhengxue/datasets/gqa/images/n140421.jpg` | Do the soap bottle and the clock have the same color? | yes | no |
| 89 | 186 | `/data2/lizhengxue/datasets/gqa/images/n500209.jpg` | What is sitting on top of the shelf? | jar | books |
| 90 | 188 | `/data2/lizhengxue/datasets/gqa/images/n23762.jpg` | How tall do you think is the person? | tall | 6 feet |
| 91 | 197 | `/data2/lizhengxue/datasets/gqa/images/n513100.jpg` | Which kind of furniture is in front of the fence? | chair | table |
| 92 | 203 | `/data2/lizhengxue/datasets/gqa/images/n141939.jpg` | What is the sink on? | countertop | wood |
| 93 | 209 | `/data2/lizhengxue/datasets/gqa/images/n98544.jpg` | Does the heater next to the toilet look white and large? | no | yes |
| 94 | 217 | `/data2/lizhengxue/datasets/gqa/images/n480253.jpg` | What is parked alongside the barn? | ambulance | fire truck |
| 95 | 220 | `/data2/lizhengxue/datasets/gqa/images/n480253.jpg` | What vehicle is parked alongside the barn? | ambulance | fire truck |
| 96 | 222 | `/data2/lizhengxue/datasets/gqa/images/n143935.jpg` | Does the calf have brown color and large size? | no | yes |
| 97 | 225 | `/data2/lizhengxue/datasets/gqa/images/n507959.jpg` | Does the blue bag look small? | yes | no |
| 98 | 226 | `/data2/lizhengxue/datasets/gqa/images/n570181.jpg` | Is the baseball mitt bright? | yes | no |
| 99 | 227 | `/data2/lizhengxue/datasets/gqa/images/n235859.jpg` | Who is wearing the watch? | women | man |
| 100 | 228 | `/data2/lizhengxue/datasets/gqa/images/n235859.jpg` | Who is wearing a watch? | women | man |
