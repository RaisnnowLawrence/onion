from abc import ABC, abstractmethod
import shortuuid    # 生成唯一ID
import base64       # 图片base64编解码
import io           # 内存二进制流处理
from PIL import Image  # 图片处理
import numpy as np     # 数值计算
import random          # 随机选择
import math            # 数学计算（UCB公式）
import traceback       # 异常堆栈打印
import re              # 正则表达式，提取问题中的物体
import tempfile        # 临时文件处理
import os              # 文件路径操作
import torch           # 深度学习框架
import cv2
from qwen_utils import chat_with_qwen_vl, chat_with_qwen_vllm, string_to_list_if_possible  # Qwen VL模型调用
from sam_utils import process_langsam_results_to_visualization, combine_masks_max_simple  # SAM可视化

class QuestionSample(ABC):

    def __init__(self, row, args, round_idx=0):
        self.row = row
        self.args = args
        self.round_idx = round_idx
        self.image = row['image']  # base64
        self.question = row['question']
        self.answer = row['answer']


    async def generate(self, prompt, image, max_tokens=1024):
        # Rotate client selection
        # Randomly select client and model
        idx = random.randint(0, len(self.clients) - 1)
        client = self.clients[idx]
        model = self.models[idx]

        # Process image scaling
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes))

        # Get target size from args
        target_size = (self.args.image_size, self.args.image_size)

        # Only scale if image is larger than target size
        if img.width > target_size[0] or img.height > target_size[1]:
            # Calculate scaling ratio
            ratio = min(target_size[0]/img.width, target_size[1]/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))

            # Use bilinear interpolation for scaling
            img = img.resize(new_size, Image.Resampling.BILINEAR)

            # Convert back to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            processed_image = base64.b64encode(buffered.getvalue()).decode()
        else:
            processed_image = image

        chat_completion = await client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{processed_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            model=model,
            max_tokens=max_tokens,
            temperature=self.args.temperature if self.args.temperature > 0 else 0.0,
            extra_body={
                "add_generation_prompt": True
            }
        )

        result = chat_completion.choices[0].message.content
        return result

    async def process(self):
        try:
            return await self._process()
        except Exception as e:
            import traceback
            print(f"Error occurred while processing sample: {e}")
            traceback.print_exc()  # Print full error stack trace

            return {
                "question_id": self.row['index'],
                "round_id": self.round_idx,
                "prompt": "",
                "text": "",  # No longer default to returning 'A'
                "answer_id": shortuuid.uuid(),
                "model_id": self.args.model_path,
                "answer": self.row['answer'],
                "metadata": {"error": str(e)}
            }

    @abstractmethod
    async def _process(self):
        """Abstract method that must be implemented by subclasses"""
        pass


# ====================== MCTS节点类（同步版） ======================
class MCTSNode:
    """蒙特卡洛树搜索（MCTS）的节点类（同步版）"""
    def __init__(self, state, parent=None, available_actions=None):
        self.state = state                  # 节点状态（图片、区域、历史动作等）
        self.parent = parent                # 父节点
        self.children = {}                  # 子节点：key=动作名，value=节点
        self.visits = 0                     # 节点被访问次数
        self.value = 0                      # 节点累计奖励
        self.leaf_reward = 0                # 作为叶子节点时获得的奖励
        self.untried_actions = available_actions.copy() if available_actions else []  # 未尝试的动作
        self.expert_info = None             # 视觉专家返回的检测信息
        self.valid_area_ratio = 1.0         # 当前图片有效区域占原图比例（初始1.0=全图）
        self.region_coords = state.get('region_coords', (0, 0, state['image_width'], state['image_height']))  # 区域坐标
        self.extra_info = {}                # 额外信息存储


# ====================== MCTS策略样本类（同步版） ======================
class MCTSQuestionSample(QuestionSample):
    def __init__(self, row, args, round_idx=0, llm_model=None, llm_processor=None, sam_model=None,
                 clip_model=None, clip_processor=None, use_vllm=False, vllm_client=None, vllm_model_name=None):
        # 调用父类构造函数
        super().__init__(row, args, round_idx)

        # 使用外部传入的模型，如果没有则设为None
        self.llm_model = llm_model
        self.llm_processor = llm_processor
        self.sam_model = sam_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # vLLM API模式配置
        self.use_vllm = use_vllm
        self.vllm_client = vllm_client
        self.vllm_model_name = vllm_model_name

        # 检查模型是否已初始化
        if not self.use_vllm and (self.llm_model is None or self.sam_model is None):
            print("警告：LLM或SAM模型未初始化，请确保外部传入")
        elif self.use_vllm and self.sam_model is None:
            print("警告：SAM模型未初始化，请确保外部传入")

        # 解码图片，获取图片宽高
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        self.image_width, self.image_height = img.size

        # 保存原始图片为临时文件（用于视觉专家调用）
        self.original_image_path = row['image_path']

        # 创建32x32空白白色图片（用于纯文本提问，不需要真实图片）
        blank_image = Image.new('RGB', (32, 32), color='white')
        self.blank_image_path = self._save_temp_image(blank_image)

        # MCTS参数
        self.max_depth = 3          # 最大搜索深度
        self.c_puct = 1.0           # PUCT探索系数
        self.n_simulations = 20      # 模拟次数
        self.use_ensemble = True    # 是否使用集成投票

        # 动作空间（MCTS可选择的操作）— 由_setup_actions动态填充
        self.actions = []             # 动作名列表：["focus_cat", "focus_dog", "zoom_out", ...]
        self.action_objects = {}      # 动作名 -> 物体名映射：{"focus_cat": "cat", ...}
        self.action_prompts = {}      # 动作名 -> 描述文本
        self.detected_objects = []    # SAM实际检测到的物体，便于日志分析
        self.action_marker_ids = {}   # 动作名 -> 轻量标记编号

        # 动作执行器映射：动作名 → 对应函数（focus_* 共用 execute_focus_object_action）
        self.action_executors = {
            "zoom_out": self.execute_zoom_out_action
        }

        # MCTS根节点
        self.root = None


    # ====================== 辅助函数：保存临时图片 ======================
    def _save_temp_image(self, image):
        """将PIL Image保存为临时文件，返回文件路径"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_file, format='PNG')
        temp_file.close()
        return temp_file.name

    def _base64_to_image(self, base64_str):
        """将base64字符串转换为PIL Image"""
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes))

    def _image_to_base64(self, image):
        """将PIL Image转换为base64字符串"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_current_image(self, image_base64, prefix="temp"):
        """保存当前图片为临时文件，返回文件路径"""
        img = self._base64_to_image(image_base64)
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', prefix=prefix, delete=False)
        img.save(temp_file, format='PNG')
        temp_file.close()
        return temp_file.name

    # ====================== 同步调用LLM生成 ======================
    def generate_sync(self, prompt, image_base64, max_tokens=100):
        """
        同步调用本地Qwen-VL LLM或vLLM API生成回答
        """
        temp_image_path = self._save_current_image(image_base64, prefix="llm_input")

        try:
            if self.use_vllm:
                response = chat_with_qwen_vllm(
                    self.vllm_client, self.vllm_model_name,
                    prompt=prompt,
                    image_path=temp_image_path,
                    max_new_tokens=max_tokens,
                    use_images=True,
                    history=None,
                    return_history=False
                )
            else:
                response = chat_with_qwen_vl(
                    model=self.llm_model,
                    processor=self.llm_processor,
                    prompt=prompt,
                    image_path=temp_image_path,
                    max_new_tokens=max_tokens,
                    use_images=True,
                    history=None,
                    return_history=False
                )
            return response

        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    # ====================== 同步调用SAM视觉专家 ======================
    def get_expert_boxes_sync(self, image_base64, text):
        """
        调用本地SAM视觉专家服务进行物体检测和分割
        传入：base64图片 + 文本（要找的物体）
        返回：检测框、标签、mask
        """
        # 将base64图片转换为PIL Image
        image_pil = self._base64_to_image(image_base64)

        # 确保图片是RGB模式
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        try:
            # 调用SAM模型进行预测
            with torch.no_grad():
                text_mask = self.sam_model.predict([image_pil], [text])

            # 提取结果
            if not text_mask or len(text_mask) == 0:
                print(f"SAM未返回结果: {text}")
                return None

            first_result = text_mask[0]
            masks = first_result.get('masks', None)

            if masks is None or len(masks) == 0:
                print(f"SAM未检测到物体: {text}")
                return None

            # 从mask中提取边界框
            boxes = []
            valid_masks = []

            for i, single_mask in enumerate(masks):
                # 转换为numpy数组（如果需要）
                if isinstance(single_mask, np.ndarray):
                    mask_array = single_mask
                else:
                    mask_array = np.array(single_mask)

                # 确保是二值mask
                if mask_array.dtype != bool:
                    mask_array = mask_array > 0

                # 获取非零像素的坐标
                y_indices, x_indices = np.where(mask_array)

                if len(y_indices) > 0 and len(x_indices) > 0:
                    # 计算边界框（添加小padding）
                    x1 = max(0, np.min(x_indices) - 5)
                    y1 = max(0, np.min(y_indices) - 5)
                    x2 = min(image_pil.width, np.max(x_indices) + 5)
                    y2 = min(image_pil.height, np.max(y_indices) + 5)

                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    valid_masks.append(mask_array)

            if not boxes:
                print(f"从mask中提取边界框失败: {text}")
                return None

            # 构造返回结果
            result = {
                'boxes': boxes,
                'labels': [text] * len(boxes),
                'masks': valid_masks
            }

            print(f"SAM检测成功: {text} -> {len(boxes)}个检测框")
            return result

        except Exception as e:
            print(f"调用SAM视觉专家失败: {str(e)}")
            traceback.print_exc()
            return None

    # ====================== 纯文本LLM调用（不传图像） ======================
    def generate_text_sync(self, prompt, max_tokens=50):
        """纯文本调用LLM，用于物体提取等不需要图像的场景"""
        if self.use_vllm:
            response = chat_with_qwen_vllm(
                self.vllm_client, self.vllm_model_name,
                prompt=prompt,
                max_new_tokens=max_tokens,
                use_images=False,
                history=None,
                return_history=False
            )
        else:
            response = chat_with_qwen_vl(
                model=self.llm_model,
                processor=self.llm_processor,
                prompt=prompt,
                max_new_tokens=max_tokens,
                history=None,
                return_history=False
            )
        return response

    # ====================== 从问题中提取关键物体 ======================
    def _filter_key_objects(self, objects):
        """清洗MCTS候选物体，并可选地约束到scene graph已有物体。"""
        generic_terms = {
            'thing', 'things', 'object', 'objects', 'item', 'items', 'photo', 'picture',
            'image', 'event', 'activity', 'period', 'kind', 'type', 'place', 'area',
            'scene', 'background', 'foreground', 'what else', 'here', 'there', 'beverage'
        }
        cleaned = []
        for obj in objects:
            obj = obj.strip().lower()
            if not obj or obj in generic_terms:
                continue
            if obj not in cleaned:
                cleaned.append(obj)

        if not getattr(self.args, "mcts_filter_objects", False):
            return cleaned

        candidates = []
        for obj in self.row.get('candidate_objects', []):
            obj = str(obj).strip().lower()
            if obj and obj not in generic_terms and obj not in candidates:
                candidates.append(obj)

        if not candidates:
            return cleaned

        aligned = []
        for obj in cleaned:
            for cand in candidates:
                if obj == cand or obj in cand or cand in obj:
                    if cand not in aligned:
                        aligned.append(cand)
                    break

        if not aligned:
            aligned = candidates[:3]
            print(f"[extract_key_objects] 对象过滤后为空，回退到scene graph候选: {aligned}")

        return aligned

    def extract_key_objects_sync(self):
        """从问题里提取要检测的关键物体（用于视觉专家定位）"""
        question = self.row['question']

        # 用纯文本调用LLM（不传空白图，避免VL模型困惑）
        prompt = 'Extract the key objects (nouns) from the question below.\n'
        prompt += 'Output ONLY the object names separated by commas, nothing else.\n'
        prompt += f'Question: {question}\n'
        prompt += 'Objects:'

        response = self.generate_text_sync(prompt, max_tokens=50)

        print(f"[extract_key_objects] question: {question}")
        print(f"[extract_key_objects] raw response: {repr(response)}")

        # 清洗并解析
        objects = []
        # 移除常见的多余文字
        response = response.strip().strip('.').strip()
        # 尝试按逗号分割
        for part in response.split(','):
            obj = part.strip().strip('.').strip('"').strip("'").strip().lower()
            if obj and len(obj) >= 2:  # 至少2个字符的有效物体名
                objects.append(obj)

        if not objects:
            # fallback：用简单正则提取英文名词短语
            fallback = re.findall(r'\b([a-z]{2,}(?:\s+[a-z]{2,})?)\b', question.lower())
            # 过滤掉疑问词、介词等
            stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'the',
                          'in', 'on', 'at', 'of', 'to', 'a', 'an', 'this', 'that', 'there',
                          'doing', 'color', 'many', 'much', 'type', 'kind', 'image', 'picture'}
            objects = [w for w in fallback if w not in stop_words]
            print(f"[extract_key_objects] LLM返回为空，使用正则fallback: {objects}")

        objects = self._filter_key_objects(objects)
        print(f"[extract_key_objects] parsed objects: {objects}")
        return objects

    # ====================== MCTS选择阶段 ======================
    def selection(self, node):
        """选择：用UCB公式选最优子节点，一路选到叶子节点"""
        if node.untried_actions:
            return node  # 还有未尝试动作，直接返回当前节点用于扩展
        if not node.children:
            return node  # 没有子节点，返回自身

        total_visits = sum(child.visits for child in node.children.values())

        # UCB得分公式
        def ucb_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(2 * math.log(total_visits) / (child.visits + 1e-8))
            return exploit + self.c_puct * explore

        # 选得分最高的子节点，递归选择
        best_child = max(node.children.values(), key=ucb_score)
        return self.selection(best_child)

    # ====================== 动作：关注指定物体（生成SAM高亮图） ======================
    def execute_focus_object_action(self, node, action_name):
        """执行关注物体动作：用SAM分割指定物体，在原图上叠加mask高亮"""
        object_name = self.action_objects[action_name]

        # 获取当前节点对应的图像
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        try:
            with torch.no_grad():
                text_mask = self.sam_model.predict([image_pil], [object_name])

            first_result = text_mask[0]
            masks = first_result.get('masks', None)

            if masks is None or len(masks) == 0:
                # SAM未检测到该物体，保持原图
                enhanced_base64 = node.state['image']
            else:
                # 生成高亮可视化
                mask_group = [first_result]
                combine_mask = [combine_masks_max_simple(mask_group)]
                visualizations = process_langsam_results_to_visualization(combine_mask, image_pil)

                # 使用 isolate_masked 结果（高亮关注的物体区域）
                overlay_image = Image.fromarray(visualizations[0]['isolate_masked']['result'])
                enhanced_base64 = self._image_to_base64(overlay_image)

        except Exception as e:
            print(f"执行focus动作失败 ({object_name}): {str(e)}")
            traceback.print_exc()
            enhanced_base64 = node.state['image']

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': enhanced_base64,
            'action_history': node.state['action_history'] + [f"Focus on: {object_name}"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': node.state['region_coords'],
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0)
        }

        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== 动作2：缩小视野（zoom out 1.5x） ======================
    def execute_zoom_out_action(self, node):
        """执行缩小动作：把当前区域扩大1.5倍，避免裁掉关键物体"""
        x1, y1, x2, y2 = node.state['region_coords']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # 扩大1.5倍
        new_width = width * 1.5
        new_height = height * 1.5

        # 计算新区域，不超出图片边界
        new_x1 = max(0, center_x - new_width/2)
        new_y1 = max(0, center_y - new_height/2)
        new_x2 = min(node.state['image_width'], center_x + new_width/2)
        new_y2 = min(node.state['image_height'], center_y + new_height/2)

        # 裁剪
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))

        # 转base64
        cropped_image_base64 = self._image_to_base64(cropped_img)

        # 调用专家确认区域
        key_objects = self._ensure_key_objects()
        expert_result = self.get_expert_boxes_sync(cropped_image_base64, ", ".join(key_objects))

        # 新状态
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["zoom_out"]],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': (new_x1, new_y1, new_x2, new_y2)
        }

        # 计算有效区域比例
        total_width = self.image_width
        total_height = self.image_height
        new_width = new_x2 - new_x1
        new_height = new_y2 - new_y1
        new_state['valid_area_ratio'] = (new_width * new_height) / (total_width * total_height)

        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== 动作：裁剪放大物体（SAM分割后直接裁剪，保留真实像素） ======================
    def execute_crop_object_action(self, node, action_name):
        """用SAM分割物体，裁剪到边界框，保留真实像素细节"""
        object_name = self.action_objects[action_name]
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        try:
            with torch.no_grad():
                text_mask = self.sam_model.predict([image_pil], [object_name])
            first_result = text_mask[0]
            masks = first_result.get('masks', None)

            if masks is None or len(masks) == 0:
                enhanced_base64 = node.state['image']
                new_region = node.state['region_coords']
                x1, y1, x2, y2 = 0, 0, image_pil.width, image_pil.height
            else:
                # 从mask提取边界框
                mask_array = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
                if mask_array.dtype != bool:
                    mask_array = mask_array > 0
                y_idx, x_idx = np.where(mask_array)
                x1, y1 = int(np.min(x_idx)), int(np.min(y_idx))
                x2, y2 = int(np.max(x_idx)), int(np.max(y_idx))

                # 添加20% padding
                pad_w = int((x2 - x1) * 0.2)
                pad_h = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(image_pil.width, x2 + pad_w)
                y2 = min(image_pil.height, y2 + pad_h)

                cropped = image_pil.crop((x1, y1, x2, y2))
                enhanced_base64 = self._image_to_base64(cropped)

                # 更新区域坐标（映射回原图坐标系）
                rx1, ry1, rx2, ry2 = node.state['region_coords']
                new_region = (rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2)

        except Exception as e:
            print(f"执行crop动作失败 ({object_name}): {str(e)}")
            traceback.print_exc()
            enhanced_base64 = node.state['image']
            new_region = node.state['region_coords']
            x1, y1, x2, y2 = 0, 0, image_pil.width, image_pil.height

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': enhanced_base64,
            'action_history': node.state['action_history'] + [f"Crop on: {object_name}"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region,
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0)
        }
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = ((x2 - x1) * (y2 - y1)) / (node.state['image_width'] * node.state['image_height'])
        return child

    # ====================== 动作：在原图上画物体边界框（保留全局上下文） ======================
    def execute_outline_object_action(self, node, action_name):
        """用SAM分割物体，在原图上绘制边界框，保留全局上下文"""
        object_name = self.action_objects[action_name]
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        try:
            with torch.no_grad():
                text_mask = self.sam_model.predict([image_pil], [object_name])
            first_result = text_mask[0]
            masks = first_result.get('masks', None)

            if masks is None or len(masks) == 0:
                enhanced_base64 = node.state['image']
            else:
                mask_array = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
                if mask_array.dtype != bool:
                    mask_array = mask_array > 0
                y_idx, x_idx = np.where(mask_array)
                x1, y1 = int(np.min(x_idx)), int(np.min(y_idx))
                x2, y2 = int(np.max(x_idx)), int(np.max(y_idx))

                image_np = np.array(image_pil)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
                cv2.putText(image_np, object_name, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                enhanced_base64 = self._image_to_base64(Image.fromarray(image_np))

        except Exception as e:
            print(f"执行outline动作失败 ({object_name}): {str(e)}")
            traceback.print_exc()
            enhanced_base64 = node.state['image']

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': enhanced_base64,
            'action_history': node.state['action_history'] + [f"Outline on: {object_name}"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': node.state['region_coords'],
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0)
        }
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== 动作：轻量编号提示（尽量少改图像） ======================
    def execute_marker_object_action(self, node, action_name):
        """用小编号点提示目标位置，并在角落给出编号说明，避免大面积覆盖图像。"""
        object_name = self.action_objects[action_name]
        marker_id = self.action_marker_ids.get(action_name, 1)
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        try:
            with torch.no_grad():
                text_mask = self.sam_model.predict([image_pil], [object_name])
            first_result = text_mask[0]
            masks = first_result.get('masks', None)

            if masks is None or len(masks) == 0:
                enhanced_base64 = node.state['image']
            else:
                mask_array = masks[0] if isinstance(masks[0], np.ndarray) else np.array(masks[0])
                if mask_array.dtype != bool:
                    mask_array = mask_array > 0
                y_idx, x_idx = np.where(mask_array)
                x1, y1 = int(np.min(x_idx)), int(np.min(y_idx))
                x2, y2 = int(np.max(x_idx)), int(np.max(y_idx))

                image_np = np.array(image_pil).copy()
                h, w = image_np.shape[:2]
                radius = max(4, min(w, h) // 90)
                thickness = max(1, radius // 3)
                font_scale = max(0.35, min(w, h) / 1400.0)
                font_thickness = max(1, int(round(font_scale * 2)))

                # Put the marker near the bbox corner, clamped inside the image.
                mx = min(max(x1, radius + 2), w - radius - 2)
                my = min(max(y1, radius + 2), h - radius - 2)
                color = (255, 225, 0)
                shadow = (0, 0, 0)
                cv2.circle(image_np, (mx, my), radius + 2, shadow, -1)
                cv2.circle(image_np, (mx, my), radius, color, -1)
                cv2.circle(image_np, (mx, my), radius, shadow, thickness)
                cv2.putText(image_np, str(marker_id), (mx - radius // 2, my + radius // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow, font_thickness)

                # Keep the text in a small corner strip rather than on top of the object.
                label = f"{marker_id}: {object_name}"
                label_x = 6
                label_y = 18 + (marker_id - 1) * 18
                if label_y < h - 6:
                    cv2.putText(image_np, label, (label_x + 1, label_y + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow, font_thickness + 1)
                    cv2.putText(image_np, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

                enhanced_base64 = self._image_to_base64(Image.fromarray(image_np))

        except Exception as e:
            print(f"执行marker动作失败 ({object_name}): {str(e)}")
            traceback.print_exc()
            enhanced_base64 = node.state['image']

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': enhanced_base64,
            'action_history': node.state['action_history'] + [f"Marker {marker_id} on: {object_name}"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': node.state['region_coords'],
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0)
        }
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== 动作：向左平移视野 ======================
    def execute_pan_left_action(self, node):
        """将视野向左平移25%，露出右侧新区域"""
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        w, h = image_pil.size
        shift = int(w * 0.25)
        # 向左平移：裁剪右侧75%，左侧补原图最左边
        left_strip = image_pil.crop((0, 0, shift, h))
        right_part = image_pil.crop((shift, 0, w, h))
        panned = Image.new('RGB', (w, h))
        panned.paste(right_part, (0, 0))
        panned.paste(left_strip, (w - shift, 0))

        x1, y1, x2, y2 = node.state['region_coords']
        new_region = (x1 + shift, y1, min(x2 + shift, node.state['image_width']), y2)

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': self._image_to_base64(panned),
            'action_history': node.state['action_history'] + ["Pan left by 25%"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region,
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0)
        }
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== 动作：中心区域2x放大 ======================
    def execute_zoom_in_center_action(self, node):
        """裁剪图像中心50%区域，拉伸回原尺寸，实现2x中心放大"""
        image_pil = self._base64_to_image(node.state['image'])
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        w, h = image_pil.size
        cx1, cy1 = int(w * 0.25), int(h * 0.25)
        cx2, cy2 = int(w * 0.75), int(h * 0.75)
        cropped = image_pil.crop((cx1, cy1, cx2, cy2))
        zoomed = cropped.resize((w, h), Image.Resampling.BICUBIC)

        # 更新区域坐标
        rx1, ry1, rx2, ry2 = node.state['region_coords']
        new_region = (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2)

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': self._image_to_base64(zoomed),
            'action_history': node.state['action_history'] + ["Zoom in center 2x"],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region,
            'valid_area_ratio': node.state.get('valid_area_ratio', 1.0) * 0.25
        }
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.valid_area_ratio = new_state['valid_area_ratio']
        return child

    # ====================== MCTS扩展阶段 ======================
    def expansion(self, node):
        """扩展：随机选一个未尝试动作，创建子节点"""
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # 按动作类型分发到对应执行器
        if action in self.action_executors:
            child = self.action_executors[action](node)
        elif action.startswith("crop_"):
            child = self.execute_crop_object_action(node, action)
        elif action.startswith("outline_"):
            child = self.execute_outline_object_action(node, action)
        elif action.startswith("marker_"):
            child = self.execute_marker_object_action(node, action)
        else:
            child = self.execute_focus_object_action(node, action)

        node.children[action] = child
        return child

    # ====================== MCTS模拟（评估）阶段 ======================
    def simulation(self, node):
        """模拟：用CLIP计算增强图像与问题文本的对齐分数作为奖励（无GT泄露）"""
        if self.clip_model is not None and self.clip_processor is not None:
            # ===== CLIP对齐分数 =====
            question = self.row['question']
            image_pil = self._base64_to_image(node.state['image'])

            inputs = self.clip_processor(
                text=[question],
                images=image_pil,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)

            # logit_scale将cosine similarity映射为logits，这里还原真实cosine similarity
            logit_scale = self.clip_model.logit_scale.exp().item()
            raw_logit = outputs.logits_per_image[0, 0].item()
            cosine_sim = raw_logit / logit_scale
            reward = float((cosine_sim + 1.0) / 2.0)  # 归一化到 [0, 1]
        else:
            # ===== 回退：模型自评置信度（无CLIP时） =====
            question = self.row['question']
            prompt = 'Answer the following question based on the image using a single word or short phrase.\n'
            prompt += f'Question: {question}\n'
            prompt += 'After answering, rate your confidence in the answer on a scale of 1-5.\n'
            prompt += 'Output format: "Answer: [your answer], Confidence: [1-5]"'

            response = self.generate_sync(prompt, node.state['image'], max_tokens=100)

            try:
                conf_match = re.search(r'[Cc]onfidence:\s*(\d)', response)
                if conf_match:
                    confidence = int(conf_match.group(1))
                    reward = max(0.0, min(1.0, confidence / 5.0))
                else:
                    reward = 0.3 if len(response.strip()) > 0 else 0.0
            except Exception:
                reward = 0.1

        node.leaf_reward = reward
        return reward

    # ====================== MCTS反向传播 ======================
    def backpropagation(self, node, reward):
        """反向传播：把奖励向上更新所有祖先节点"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    # ====================== 单次MCTS流程 ======================
    def single_run(self, root_state):
        """一次完整的MCTS：选择 → 扩展 → 模拟 → 回溯"""
        if not self.root:
            self.root = MCTSNode(root_state, available_actions=self.actions)
            self.root.parent = None

        node = self.selection(self.root)
        node = self.expansion(node)
        reward = self.simulation(node)
        self.backpropagation(node, reward)

        return reward

    # ====================== MCTS搜索最终答案 ======================
    def get_final_answer(self):
        """运行多次MCTS，选出最优区域，给出最终答案"""
        initial_state = {
            'depth': 0,
            'image': self.image,
            'action_history': [],
            'text': self.row['question'],
            'image_width': self.image_width,
            'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height),
            'valid_area_ratio': 1.0
        }

        # 多次模拟
        for _ in range(self.n_simulations):
            self.single_run(initial_state)

        # 收集所有节点
        all_nodes = []
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            all_nodes.append(node)
            nodes_to_visit.extend(node.children.values())

        # 对所有节点对应的区域提问，加权投票得到最终答案
        final_qs = self.row['question']
        answers = []
        best_node = None
        best_score = -1

        for node in all_nodes:
            answer = self.generate_sync(final_qs, node.state['image'])
            weight = node.leaf_reward if node.leaf_reward > 0 else node.value / max(node.visits, 1)

            # 记录最佳节点
            if weight > best_score:
                best_score = weight
                best_node = node

            answers.append((answer.strip(), weight))

        # 加权投票：相同答案（忽略大小写）合并权重
        from collections import defaultdict
        vote_result = defaultdict(float)
        for ans, w in answers:
            if ans:  # 忽略空答案
                vote_result[ans.lower()] += w

        if not vote_result:
            return "", final_qs, "", self.image, None, self.root

        final_answer = max(vote_result, key=vote_result.get)
        # 找到该答案对应的原始大小写形式
        for ans, _ in answers:
            if ans.lower() == final_answer:
                final_answer = ans
                break

        return final_answer, final_qs, answers[-1][0] if answers else "", best_node.state['image'] if best_node else self.image, best_node, self.root

    # ====================== 清理临时文件 ======================
    def __del__(self):
        """析构函数：清理临时文件"""
        try:
            if hasattr(self, 'blank_image_path') and os.path.exists(self.blank_image_path):
                os.remove(self.blank_image_path)
        except:
            pass

    # ====================== 动态构建动作空间 ======================
    def _setup_actions(self):
        """
        根据当前图像中检测到的物体，动态构建MCTS动作空间。
        每个检测到的物体对应一个 focus_{物体名} 动作，外加 zoom_out。
        """
        # 从问题中提取关键物体作为SAM检测候选
        candidate_objects = self._ensure_key_objects()

        # 用SAM在全图上尝试检测每个候选物体，保留能检测到的
        image_pil = self._base64_to_image(self.image)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        detected_objects = []
        for obj in candidate_objects:
            if not obj or not obj.strip():  # 跳过空字符串
                continue
            try:
                with torch.no_grad():
                    text_mask = self.sam_model.predict([image_pil], [obj])
                first_result = text_mask[0]
                masks = first_result.get('masks', None)
                if masks is not None and len(masks) > 0:
                    detected_objects.append(obj)
            except Exception as e:
                print(f"SAM检测物体 '{obj}' 失败: {str(e)}")

        print(f"SAM检测到的物体: {detected_objects}")
        self.detected_objects = detected_objects

        # 为每个检测到的物体创建 focus 动作
        self.actions = []
        self.action_objects = {}
        self.action_prompts = {}
        self.action_marker_ids = {}
        action_mode = getattr(self.args, "mcts_action_mode", "all")

        for marker_id, obj in enumerate(detected_objects, start=1):
            obj_key = obj.replace(' ', '_')
            if action_mode in ("all", "no_crop"):
                focus_name = f"focus_{obj_key}"
                self.actions.append(focus_name)
                self.action_objects[focus_name] = obj
                self.action_prompts[focus_name] = f"Focus on: {obj}"

            if action_mode == "all":
                crop_name = f"crop_{obj_key}"
                self.actions.append(crop_name)
                self.action_objects[crop_name] = obj
                self.action_prompts[crop_name] = f"Crop on: {obj}"

            if action_mode in ("all", "outline_only", "no_crop"):
                outline_name = f"outline_{obj_key}"
                self.actions.append(outline_name)
                self.action_objects[outline_name] = obj
                self.action_prompts[outline_name] = f"Outline on: {obj}"

            if action_mode == "marker_only":
                marker_name = f"marker_{obj_key}"
                self.actions.append(marker_name)
                self.action_objects[marker_name] = obj
                self.action_marker_ids[marker_name] = marker_id
                self.action_prompts[marker_name] = f"Marker {marker_id} on: {obj}"

        if action_mode in ("all", "no_crop"):
            self.actions.append("zoom_out")
            self.action_prompts["zoom_out"] = "Zoom out the region by 1.5x"
            self.actions.append("pan_left")
            self.action_prompts["pan_left"] = "Pan left by 25%"
            self.actions.append("zoom_in_center")
            self.action_prompts["zoom_in_center"] = "Zoom in center 2x"

            # 注册全局动作执行器
            self.action_executors["pan_left"] = self.execute_pan_left_action
            self.action_executors["zoom_in_center"] = self.execute_zoom_in_center_action

        if len(detected_objects) == 0:
            print(f"警告：未检测到任何物体，动作模式 {action_mode} 下actions={self.actions}")

    # ====================== 确保关键物体已提取 ======================
    def _ensure_key_objects(self):
        """延迟初始化key_objects，确保在需要时已提取"""
        if not hasattr(self, 'key_objects') or self.key_objects is None:
            self.key_objects = self.extract_key_objects_sync()
        return self.key_objects

    # ====================== 核心实现：父类要求的_process方法 ======================
    async def _process(self):
        """
        MCTS搜索最优图片区域，再回答问题
        """
        # 1. 提取问题中的关键物体并构建动态动作空间
        self.key_objects = self.extract_key_objects_sync()
        self._setup_actions()

        # 2. 运行MCTS搜索最优图片区域
        final_answer, prompt, full_answer, final_image, best_node, root_node = self.get_final_answer()

        # 3. 返回评测结果
        return {
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
        }
