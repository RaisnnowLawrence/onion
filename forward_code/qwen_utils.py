import argparse
import torch
import base64
from tqdm import tqdm
import ast

# qwen输出是字符串,所以这是一个配套解决输出的函数
def string_to_list_if_possible(s):
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        else:
            # 如果不是列表，返回原字符串或按需处理
            return s
    except (SyntaxError, ValueError):
        # 如果解析失败，返回原字符串
        return s

# # 示例
# print(string_to_list_if_possible("[1, 2, 3]"))  # 输出: [1, 2, 3]
# print(string_to_list_if_possible("['a', 'b']")) # 输出: ['a', 'b']
# print(string_to_list_if_possible("not a list")) # 输出: "not a list"
# print(string_to_list_if_possible("{'key': 1}")) # 输出: "{'key': 1}" (字典，不转换)

def initialize_qwen(model_name):

    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import AutoTokenizer

    if model_name == "qwen3-VL-2B":
        qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-2B-Instruct"
    elif model_name == "qwen3-VL-4B":
        qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-4B-Instruct"
    elif model_name == "qwen3-VL-8B":
        qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-8B-Instruct"
    elif model_name == "qwen3-VL-30B":
        qwen_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/Qwen/Qwen3-VL-30B-A3B-Instruct"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_path, 
        dtype="auto", 
        device_map="cuda")
    processor = AutoProcessor.from_pretrained(qwen_path)
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    return model, processor, tokenizer

def chat_with_qwen_vl(
    model,
    processor,
    prompt,
    image_path = None,  # 图像路径，可以是单个字符串或字符串列表
    max_new_tokens: int = 512,
    use_images: bool = True,
    history: list = None,  # 新增：历史对话参数
    return_history: bool = False  # 新增：是否返回更新后的历史
) -> str:
    """
    与Qwen3-VL模型进行对话交流，支持多轮对话
    
    Args:
        model: 已加载的Qwen3-VL模型
        processor: 已加载的处理器
        prompt: 用户输入的文本提示
        image_path: 图像路径或URL，可以是单个字符串或字符串列表
        max_new_tokens: 生成的最大新token数量
        use_images: 是否使用图像输入
        history: 历史对话列表，格式为 [{"role": "user", "content": [...]}, ...]
        return_history: 是否返回更新后的历史对话
        
    Returns:
        str: 模型的回复文本
        list: 如果return_history=True，返回(回复文本, 更新后的历史对话)
    """
    # 初始化历史对话
    if history is None:
        messages = []
    else:
        messages = history.copy()  # 复制历史对话，避免修改原列表
    
    # 构建当前用户消息
    current_message = {"role": "user", "content": []}
    
    # 添加图像内容
    if use_images and image_path:
        if isinstance(image_path, str):
            image_path = [image_path]
            
        for path in image_path:
            current_message["content"].append({
                "type": "image",
                "image": path
            })
    
    # 添加文本内容
    current_message["content"].append({
        "type": "text",
        "text": prompt
    })
    
    # 将当前消息添加到对话历史
    messages.append(current_message)
    
    # 准备模型输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    reply = output_text[0] if output_text else ""
    
    # 将模型的回复添加到历史对话
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
    
    if return_history:
        return reply, messages
    else:
        return reply


def chat_with_qwen_vllm(
    client,
    model_name,
    prompt,
    image_path=None,
    max_new_tokens: int = 512,
    use_images: bool = True,
    history: list = None,
    return_history: bool = False
) -> str:
    """
    通过vLLM OpenAI兼容API与Qwen3-VL模型进行对话，支持多轮对话
    """
    if history is None:
        messages = []
    else:
        messages = history.copy()

    current_message = {"role": "user", "content": []}

    if use_images and image_path:
        if isinstance(image_path, str):
            image_path = [image_path]

        for path in image_path:
            with open(path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()
            current_message["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

    current_message["content"].append({
        "type": "text",
        "text": prompt
    })

    messages.append(current_message)

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        extra_body={"top_k": 20},
    )

    reply = chat_response.choices[0].message.content

    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

    if return_history:
        return reply, messages
    else:
        return reply


# 简化版本：假设answer_list中存储的就是正确答案文本（选择题）或列表（非选择题）
def calculate_final_score_simple(pred_answer, answer, pred_answer_list=None, answer_list=None, choice_mode=False):
    """
    简化版本，假设answer_list中存储的就是正确答案
    
    参数:
    - choice_mode: 是否为选择题模式
    """
    
    # 计算当前问题得分
    if choice_mode:
        # 选择题：直接比较文本
        current_score = 1 if pred_answer == answer else 0
    else:
        # 非选择题
        if not isinstance(answer, list):
            answer = [answer]
        
        counter = 0
        for ans in answer:
            if pred_answer == ans:
                counter += 1
        
        current_score = min(1.0, float(counter) * 0.3)

    # 计算所有问题平均得分
    all_scores = [current_score]  # 从当前得分开始
    
    if pred_answer_list is not None and answer_list is not None:
        for i in range(len(pred_answer_list)):
            hist_pred = pred_answer_list[i]
            hist_answer = answer_list[i]
            
            if choice_mode:
                hist_score = 1 if hist_pred == hist_answer else 0
            else:
                if not isinstance(hist_answer, list):
                    hist_answer = [hist_answer]
                
                hist_counter = 0
                for ans in hist_answer:
                    if hist_pred == ans:
                        hist_counter += 1
                
                hist_score = min(1.0, float(hist_counter) * 0.3)
            
            all_scores.append(hist_score)
    
    print('all_scores:', all_scores)
    
    avg_score = sum(all_scores) / len(all_scores)
    
    return current_score, avg_score

# 参数解析器函数
def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--set_name', type=str, default='aokvqa')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--raw_image_dir', type=str, default="/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco17")
    parser.add_argument('--coco_path', type=str, default='/data2/lizhengxue/WorkSpace/huchunning/VisualCoT-data/coco_annotations')
    parser.add_argument('--choice_only', action='store_true')

    args = parser.parse_args()
    
    return args

def main():

    # 加载参数
    args = parser_args()

    # 初始化模型
    model, processor, tokenizer = initialize_qwen('qwen3-VL-8B')

    # 准备数据
    from aokvqa_utils import load_aokvqa_dataset_v1, find_image_path
    val_keys, args.raw_image_dir, answer_dict, question_dict, _, choices_dict, = load_aokvqa_dataset_v1(args)

    # 创建结果字典
    results_dict = {}

    pred_answer_list = []
    answer_list = []

    for idx, key in enumerate(tqdm(val_keys)):

        # # 单个样例-单点检查
        # if '405691' not in key : continue

        # 获取图像ID
        image_key = int(key.split('<->')[0])
        print("Processing index:", idx, "Image ID:", image_key)

        # 准备message
        # image_path = raw_image_dir[idx]
        image_path = find_image_path(args, image_key)
        print(f"Processing image: {image_path}")

        # 获取问题文本
        question = question_dict[key]
        print(f"Question: {question}")

        # 获取选项列表
        choices = choices_dict[key]
        print(f"Choices: {choices}")

        # 获取答案文本
        answer = answer_dict[key]
        print(f"Answer: {answer}")

        
        # # 生成图像的caption
        # # 与模型信息交换+检测
        # outputs = chat_with_qwen_vl(
        #     model, 
        #     processor, 
        #     "Please briefly describe the content of the picture.", 
        #     image_path,
        #     max_new_tokens = 512
        # )
        # print(outputs)


        # # 生成问题的信息提取
        # # 与模型信息交换+检测
        # outputs = chat_with_qwen_vl(
        #     model, 
        #     processor, 
        #     "Please help me extract all the main targets in the question, typically nouns, and output them in the form of a list. If there are no targets, please output an empty list. \nQuestion: " + question, 
        #     use_images = False
        # )
        # print(outputs)
        # # 每十个样本保存一下
        # # 保存到字典
        # results_dict[image_key] = string_to_list_if_possible(outputs)

        # 直接回答问题
        # 与模型信息交换+检测
        outputs = chat_with_qwen_vl(
            model, 
            processor, 
            "Please carefully observe the content of the image and answer the following questions using words or phrases. \nQuestion: " + question, + "\nOptions: " + choices,
            use_images = True,
            image_path = image_path
        )
        print('outputs:', outputs)

        # 每十个样本保存一下
        # 保存到字典
        results_dict[image_key] = string_to_list_if_possible(outputs)

        pred_answer_list.append(outputs)
        answer_list.append(answer)
        current_score, avg_score = calculate_final_score_simple(outputs, answer, pred_answer_list=pred_answer_list, answer_list=answer_list, choice_mode=args.choice_only)
        print(f"Current Score: {current_score}, Average Score: {avg_score}")
        
        # 每十个样本保存一次
        if (idx + 1) % 10 == 0:
            import json
            with open('qwen_vl_results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Saved results for {idx + 1} samples")
    
    # 最终保存所有结果
    import json
    with open('qwen_vl_results_final.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Final results saved. Total samples: {len(results_dict)}")


if __name__ == "__main__":
    main()  # 程序从这里开始执行