import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time

import numpy as np
from typing import List, Union, Optional

import numpy as np


def clean_string_basic(text):
    """
    只保留字母和数字,去除所有空格、符号等
    
    示例：
    "Hello, World! 2023" → "HelloWorld2023"
    "A-B_C d@e" → "ABCde"
    """
    # 使用正则表达式只保留字母和数字
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
    return cleaned

def combine_masks_max_simple(data_list):
    """
    从多个data字典中提取所有mask并组合,同一位置取最大值
    
    Args:
        data_list: 包含mask字典的列表
        
    Returns:
        字典 {'mask': 组合后的mask [H, W]}
    """
    # 直接合并所有mask
    all_masks = np.concatenate([d['masks'] for d in data_list], axis=0)
    combined_mask = all_masks.max(axis=0).astype(np.float32)
    
    return {'mask': combined_mask}

def combine_masks_max(results):
    """
    从results列表中提取所有mask并组合,同一位置取最大值
    
    Args:
        results: 包含mask字典的列表
        
    Returns:
        组合后的单个mask [H, W]
    """
    all_masks = []
    
    for result in results:
        if 'masks' in result:
            masks = result['masks']
            if isinstance(masks, np.ndarray):
                if masks.ndim == 3:
                    # [N, H, W] 格式,直接添加
                    all_masks.append(masks)
                elif masks.ndim == 2:
                    # 单个mask [H, W]
                    all_masks.append(masks[np.newaxis, ...])
            elif isinstance(masks, list):
                # list转array
                mask_array = np.array(masks)
                all_masks.append(mask_array)
    
    if not all_masks:
        raise ValueError("没有找到有效的mask")
    
    # 检查所有mask的形状是否一致
    first_shape = all_masks[0].shape[1:]  # 去掉N维度
    for i, masks in enumerate(all_masks):
        if masks.shape[1:] != first_shape:
            raise ValueError(f"mask {i}的形状{masks.shape[1:]}与第一个mask形状{first_shape}不匹配")
    
    # 合并所有mask
    combined_masks = np.concatenate(all_masks, axis=0)  # [总N, H, W]
    
    # 取最大值
    combined_mask = combined_masks.max(axis=0)
    
    return combined_mask.astype(np.float32)

def combine_masks(
    masks: Union[List[np.ndarray], np.ndarray],
    method: str = "union",
    weights: Optional[List[float]] = None,
    threshold: float = 0.5
) -> np.ndarray:
    """
    将多个mask组合成单个mask
    
    Args:
        masks: mask列表,每个mask为[H, W]的numpy数组,或[N, H, W]的数组
        method: 组合方式,可选:
            - "union": 并集（默认）
            - "intersection": 交集
            - "majority": 多数投票
            - "weighted": 加权平均
            - "max": 取最大值
        weights: 当method="weighted"时使用的权重列表
        threshold: 二值化阈值,用于非二值mask
    
    Returns:
        组合后的mask [H, W],值为0-1之间的float或bool
    """
    # 1. 统一输入格式
    if isinstance(masks, list):
        # 转换为numpy数组
        if len(masks) == 0:
            raise ValueError("mask列表不能为空")
        
        # 获取第一个mask的形状作为参考
        first_shape = masks[0].shape
        
        # 确保所有mask形状一致
        mask_array = []
        for i, mask in enumerate(masks):
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            if mask.shape != first_shape:
                raise ValueError(
                    f"mask {i}的形状{mask.shape}与第一个mask形状{first_shape}不匹配"
                )
            
            # 如果是二值mask,确保是bool类型
            if mask.max() <= 1:
                mask = mask.astype(np.float32)
            
            mask_array.append(mask)
        
        masks = np.stack(mask_array, axis=0)  # [N, H, W]
    
    elif isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]  # 单个mask -> [1, H, W]
        elif masks.ndim == 3:
            # 已经是 [N, H, W] 格式
            pass
        else:
            raise ValueError(f"不支持的mask维度: {masks.ndim}")
    
    # 2. 应用组合方法
    if method == "union":
        # 并集：任何mask中有值的位置都保留
        if masks.max() > 1:
            # 非二值mask,使用阈值
            combined = (masks.max(axis=0) > threshold).astype(np.float32)
        else:
            # 二值mask
            combined = masks.max(axis=0).astype(np.float32)
    
    elif method == "intersection":
        # 交集：所有mask都有值的位置才保留
        if masks.max() > 1:
            # 非二值mask
            combined = (masks.min(axis=0) > threshold).astype(np.float32)
        else:
            # 二值mask
            combined = masks.min(axis=0).astype(np.float32)
    
    elif method == "majority":
        # 多数投票：超过一半的mask认为有值的位置
        n_masks = masks.shape[0]
        if masks.max() > 1:
            # 非二值mask,先二值化
            binary_masks = (masks > threshold).astype(np.float32)
        else:
            binary_masks = masks
        
        vote_sum = binary_masks.sum(axis=0)
        combined = (vote_sum > (n_masks / 2)).astype(np.float32)
    
    elif method == "weighted":
        # 加权平均
        if weights is None:
            weights = [1.0] * masks.shape[0]
        
        if len(weights) != masks.shape[0]:
            raise ValueError(f"权重数量{len(weights)}与mask数量{masks.shape[0]}不匹配")
        
        weights = np.array(weights)[:, np.newaxis, np.newaxis]
        weighted_sum = (masks * weights).sum(axis=0)
        weight_sum = weights.sum()
        combined = weighted_sum / weight_sum
        
        # 可选：二值化
        if threshold is not None:
            combined = (combined > threshold).astype(np.float32)
    
    elif method == "max":
        # 取最大值
        combined = masks.max(axis=0)
        if threshold is not None:
            combined = (combined > threshold).astype(np.float32)
    
    else:
        raise ValueError(f"不支持的组合方法: {method}")
    
    return combined

def combine_masks_with_scores(
    masks: List[np.ndarray],
    scores: Optional[List[float]] = None,
    threshold: float = 0.5,
    min_score: float = 0.0,
    method: str = "weighted"
) -> np.ndarray:
    """
    结合置信度分数组合mask
    
    Args:
        masks: mask列表,每个[H, W]
        scores: 每个mask对应的置信度分数
        threshold: 二值化阈值
        min_score: 最小置信度,低于此值的mask将被忽略
        method: 组合方法,可选 "weighted", "union", "intersection"
    
    Returns:
        组合后的mask [H, W]
    """
    if len(masks) == 0:
        raise ValueError("mask列表不能为空")
    
    # 处理置信度
    if scores is None:
        scores = [1.0] * len(masks)
    
    # 过滤低置信度的mask
    valid_masks = []
    valid_scores = []
    
    for mask, score in zip(masks, scores):
        if score >= min_score:
            # 确保mask是numpy数组
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # 归一化到0-1
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            elif mask.max() <= 1 and mask.min() >= 0:
                mask = mask.astype(np.float32)
            
            valid_masks.append(mask)
            valid_scores.append(score)
    
    if len(valid_masks) == 0:
        # 没有有效mask,返回全0
        h, w = masks[0].shape
        return np.zeros((h, w), dtype=np.float32)
    
    # 堆叠mask
    mask_array = np.stack(valid_masks, axis=0)  # [N, H, W]
    
    if method == "weighted":
        # 使用置信度作为权重
        weights = np.array(valid_scores)[:, np.newaxis, np.newaxis]
        weighted_sum = (mask_array * weights).sum(axis=0)
        weight_sum = weights.sum()
        
        if weight_sum > 0:
            combined = weighted_sum / weight_sum
        else:
            combined = np.zeros_like(mask_array[0])
    
    elif method == "union":
        # 带置信度的并集
        score_weights = np.array(valid_scores)[:, np.newaxis, np.newaxis]
        # 置信度加权的最大值
        combined = (mask_array * score_weights).max(axis=0)
        # 归一化
        combined = combined / max(valid_scores) if max(valid_scores) > 0 else combined
    
    elif method == "intersection":
        # 带置信度的交集
        score_weights = np.array(valid_scores)[:, np.newaxis, np.newaxis]
        # 对于低置信度mask,将其值向0.5靠拢（降低影响力）
        adjusted_masks = mask_array.copy()
        for i, score in enumerate(valid_scores):
            if score < 0.5:
                # 低置信度mask：使其更接近不确定状态（0.5）
                adjusted_masks[i] = 0.5 + (adjusted_masks[i] - 0.5) * score * 2
        
        combined = adjusted_masks.min(axis=0)
    
    else:
        raise ValueError(f"不支持的组合方法: {method}")
    
    # 二值化
    if threshold is not None:
        combined = (combined > threshold).astype(np.float32)
    
    return combined

def combine_masks_simple(masks, mode="union"):
    """
    快速组合mask的实用函数
    
    Args:
        masks: 可以是:
            - list of np.ndarray
            - np.ndarray of shape [N, H, W]
            - np.ndarray of shape [H, W, N]
        mode: "union", "intersection", "mean"
    
    Returns:
        np.ndarray of shape [H, W]
    """
    # 转换为numpy数组
    if isinstance(masks, list):
        masks = np.array(masks)  # [N, H, W]
    
    # 处理不同形状
    if masks.ndim == 2:
        # 单个mask
        return masks.astype(np.float32)
    
    elif masks.ndim == 3:
        if masks.shape[0] < masks.shape[2]:
            # 可能是 [H, W, N] 格式
            masks = np.transpose(masks, (2, 0, 1))
    
    # 确保值在0-1之间
    if masks.max() > 1:
        masks = masks.astype(np.float32) / 255.0
    
    # 应用组合模式
    if mode == "union":
        combined = masks.max(axis=0)
    elif mode == "intersection":
        combined = masks.min(axis=0)
    elif mode == "mean":
        combined = masks.mean(axis=0)
    else:
        raise ValueError(f"不支持的mode: {mode}")
    
    return combined

# 方案1：基于距离场的渐变效果（改为白色背景）
def isolate_masked_regions_with_gradient(image_pil, masks, scores=None, max_distance=300, falloff_type='linear'):
    """
    改造版本：mask区域保留原图颜色，离mask越远颜色越淡（基于距离场）
    修改：使用白色背景，实现渐渐变白的效果
    
    Args:
        image_pil: PIL Image对象 (RGB格式)
        masks: numpy数组 [N, H, W] 或 [H, W]
        scores: 置信度（可选）
        max_distance: 最大影响距离（像素）
        falloff_type: 衰减类型 'linear'（线性）或 'exponential'（指数）
    
    Returns:
        包含可视化结果的字典
    """
    # 1. 快速转换为numpy数组
    image = np.asarray(image_pil)
    
    # 确保是RGB格式
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # 直接切片取RGB
    
    h, w = image.shape[:2]
    
    # 2. 处理mask输入格式
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]
    
    n_masks = masks.shape[0]
    
    # 3. 创建空白背景（改为白色背景）
    background_color = (255, 255, 255)  # 白色背景
    result = np.full_like(image, background_color, dtype=np.uint8)
    
    # 4. 创建组合mask（所有mask的并集）
    combined_mask = np.zeros((h, w), dtype=bool)
    
    if n_masks > 0:
        for i in range(n_masks):
            mask = masks[i]
            # 快速二值化
            if mask.max() > 1:
                binary_mask = mask > 0
            else:
                binary_mask = mask.astype(bool)
            combined_mask = combined_mask | binary_mask
    
    # 5. 计算距离场
    from scipy.ndimage import distance_transform_edt
    
    if combined_mask.any():
        # 计算到mask区域的距离
        # 注意：distance_transform_edt计算到False像素的距离，所以我们需要取反
        distances = distance_transform_edt(~combined_mask)
        
        # 对距离进行裁剪和归一化
        distances_clipped = np.clip(distances, 0, max_distance)
        
        if falloff_type == 'linear':
            # 线性衰减：距离越远，alpha值越小
            alpha_map = 1.0 - (distances_clipped / max_distance)
        elif falloff_type == 'exponential':
            # 指数衰减：更自然的过渡
            alpha_map = np.exp(-distances_clipped / (max_distance / 3))
        else:
            alpha_map = 1.0 - (distances_clipped / max_distance)
        
        # 将mask区域本身的alpha设为1
        alpha_map[combined_mask] = 1.0
        
        # 确保alpha值在0-1之间
        alpha_map = np.clip(alpha_map, 0, 1)
        
        # 6. 应用渐变效果
        # 将alpha_map扩展为3通道
        alpha_map_3d = alpha_map[:, :, np.newaxis]
        
        # 计算混合结果（使用白色背景）
        result = (image.astype(np.float32) * alpha_map_3d + 
                  np.array(background_color, dtype=np.float32) * (1 - alpha_map_3d)).astype(np.uint8)
    
    else:
        # 如果没有mask，直接返回白色背景
        result = np.full_like(image, background_color, dtype=np.uint8)
    
    # 7. 保留原函数的其他输出（可选）
    heatmap = combined_mask.astype(np.float32)
    binary_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    if combined_mask.any():
        binary_mask_img[combined_mask] = (255, 255, 255)
    
    # 创建简单的颜色覆盖版本
    color_overlay = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    if n_masks > 0:
        for i in range(min(n_masks, 4)):
            mask = masks[i]
            if mask.max() > 1:
                binary_mask = mask > 0
            else:
                binary_mask = mask.astype(bool)
            rows, cols = np.where(binary_mask)
            if len(rows) > 0:
                color_overlay[rows, cols] = colors[i % len(colors)]
    
    return {
        'original': image,
        'result': result,  # 渐变效果的结果（白色背景）
        'overlay': color_overlay,
        'heatmap': heatmap,
        'binary_mask': binary_mask_img,
        'combined_mask': combined_mask,
        'masks': masks,
        'alpha_map': alpha_map if combined_mask.any() else np.zeros((h, w))
    }

# 有mask的地方保留图像,没有mask的地方变为空白
def isolate_masked_regions_pil_fast(image_pil, masks, scores=None, alpha=0.5):
    """
    专门优化用于PIL Image输入的mask可视化函数
    改造版本：有mask的地方保留图像,没有mask的地方变为空白
    
    Args:
        image_pil: PIL Image对象 (RGB格式)
        masks: numpy数组 [N, H, W] 或 [H, W]
        scores: 置信度（可选）
        alpha: 透明度（在本版本中用于控制mask边缘融合）
    
    Returns:
        包含可视化结果的字典
    """
    # 1. 快速转换为numpy数组
    image = np.asarray(image_pil)
    
    # 确保是RGB格式
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # 直接切片取RGB
    
    h, w = image.shape[:2]
    
    # 2. 处理mask输入格式
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]
    
    n_masks = masks.shape[0]
    
    # 3. 预定义颜色（仅用于热力图显示,不再用于覆盖图像）
    colors = [
        (255, 0, 0),     # 红
        (0, 255, 0),     # 绿
        (0, 0, 255),     # 蓝
        (255, 255, 0),   # 青
        (255, 0, 255),   # 洋红
        (0, 255, 255),   # 黄
        (192, 128, 0),
        (128, 0, 192),
        (0, 192, 128),
    ]
    
    # 4. 创建空白背景（白色或透明色）
    # 可以改为黑色：(0, 0, 0),或透明色：(0, 0, 0, 0) 如果是RGBA
    background_color = (255, 255, 255)  # 白色背景
    result = np.full_like(image, background_color, dtype=np.uint8)
    
    # 5. 创建热力图和组合mask
    heatmap = np.zeros((h, w), dtype=np.float32)
    combined_mask = np.zeros((h, w), dtype=bool)
    
    # 6. 批量处理所有mask
    if n_masks > 0:
        # 预计算所有二值mask
        binary_masks = []
        mask_indices = []  # 存储每个mask的坐标
        
        for i in range(n_masks):
            mask = masks[i]
            # 快速二值化
            if mask.max() > 1:
                binary_mask = mask > 0
            else:
                binary_mask = mask.astype(bool)
            
            if binary_mask.any():
                # 获取非零坐标
                rows, cols = np.where(binary_mask)
                binary_masks.append(binary_mask)
                mask_indices.append((rows, cols))
                
                # 更新热力图
                heatmap += binary_mask.astype(np.float32)
                
                # 更新组合mask（所有mask的并集）
                combined_mask = combined_mask | binary_mask
        
        # 7. 应用mask到结果图像
        if combined_mask.any():
            # 获取所有有mask的像素坐标
            mask_rows, mask_cols = np.where(combined_mask)
            
            # 方案1：直接复制有mask区域的像素
            result[mask_rows, mask_cols] = image[mask_rows, mask_cols]
            
            # 方案2：如果需要边缘平滑,可以使用alpha混合
            if alpha < 1.0:
                # 创建边缘mask（通过腐蚀和膨胀获取边界区域）
                from scipy.ndimage import binary_erosion, binary_dilation
                import cv2
                
                # 腐蚀内部区域,获取边界
                kernel = np.ones((3, 3), np.uint8)
                if hasattr(cv2, 'erode'):
                    # 使用OpenCV
                    eroded = cv2.erode(combined_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                else:
                    # 使用scipy
                    eroded = binary_erosion(combined_mask, structure=np.ones((3, 3)))
                
                # 边界区域 = 原始mask - 腐蚀后的mask
                boundary_mask = combined_mask & ~eroded
                
                if boundary_mask.any():
                    # 获取边界坐标
                    b_rows, b_cols = np.where(boundary_mask)
                    
                    # 在边界区域进行alpha混合
                    result[b_rows, b_cols] = (
                        image[b_rows, b_cols].astype(np.float32) * alpha +
                        np.array(background_color, dtype=np.float32) * (1 - alpha)
                    ).astype(np.uint8)
        
        # 8. 为热力图创建颜色编码（可选）
        color_overlay = image.copy()
        for idx, (rows, cols) in enumerate(mask_indices):
            color = colors[idx % len(colors)]
            color_overlay[rows, cols] = color
    else:
        color_overlay = image.copy()
    
    # 9. 归一化热力图
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # 10. 创建二值mask图像（可视化用）
    binary_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    if combined_mask.any():
        binary_mask_img[combined_mask] = (255, 255, 255)  # mask区域显示为白色
    
    return {
        'original': image,              # 原始图像
        'result': result,               # 主要结果：有mask的地方保留原图,其他地方空白
        'overlay': color_overlay,       # 原始的颜色覆盖版本（与原函数兼容）
        'heatmap': heatmap,             # 热力图
        'binary_mask': binary_mask_img, # 二值mask图像
        'combined_mask': combined_mask, # 组合mask数组
        'masks': masks                  # 原始masks
    }

# mask变为彩色蒙版加到图像上
def visualize_masks_on_image_pil_fast(image_pil, masks, scores=None, alpha=0.5):
    """
    专门优化用于PIL Image输入的mask可视化函数
    
    Args:
        image_pil: PIL Image对象 (RGB格式)
        masks: numpy数组 [N, H, W] 或 [H, W]
        scores: 置信度（可选）
        alpha: 透明度
    
    Returns:
        包含可视化结果的字典
    """
    # 1. 快速转换为numpy数组（避免不必要的复制）
    # 使用np.asarray()而不是np.array(),避免复制数据
    image = np.asarray(image_pil)
    
    # 确保是RGB格式（如果输入是.convert("RGB"),这一步可以省略）
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # 直接切片取RGB,最快的方式
    
    h, w = image.shape[:2]
    
    # 2. 处理mask输入格式
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]
    
    n_masks = masks.shape[0]
    
    # 3. 预定义颜色（使用整数避免类型转换）
    colors = [
        (255, 0, 0),     # 红
        (0, 255, 0),     # 绿
        (0, 0, 255),     # 蓝
        (255, 255, 0),   # 青
        (255, 0, 255),   # 洋红
        (0, 255, 255),   # 黄
        (192, 128, 0),
        (128, 0, 192),
        (0, 192, 128),
    ]
    
    # 4. 准备叠加图像（使用图像的数据类型）
    overlay = image.copy()
    
    # 5. 创建热力图
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 6. 批量处理所有mask（向量化）
    if n_masks > 0:
        # 预计算所有二值mask
        binary_masks = []
        mask_indices = []  # 存储每个mask的坐标
        
        for i in range(n_masks):
            mask = masks[i]
            # 快速二值化（使用布尔索引）
            if mask.max() > 1:
                binary_mask = mask > 0
            else:
                binary_mask = mask.astype(bool)
            
            # 获取非零坐标（减少后续计算量）
            rows, cols = np.where(binary_mask)
            if len(rows) > 0:
                binary_masks.append(binary_mask)
                mask_indices.append((rows, cols))
                heatmap += binary_mask.astype(np.float32)
        
        # 7. 应用颜色叠加（使用整数运算加速）
        if len(mask_indices) > 0:
            # 创建alpha权重矩阵（预计算）
            alpha_weight = alpha
            beta_weight = 1 - alpha
            
            for idx, (rows, cols) in enumerate(mask_indices):
                color = colors[idx % len(colors)]
                
                # 向量化操作：直接修改对应像素
                # 公式：new = original * (1-alpha) + color * alpha
                overlay[rows, cols] = (
                    image[rows, cols].astype(np.int16) * beta_weight + 
                    np.array(color, dtype=np.int16) * alpha_weight
                ).clip(0, 255).astype(np.uint8)
    
    # 8. 归一化热力图
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return {
        'original': image,
        'overlay': overlay,
        'heatmap': heatmap,
        'masks': masks
    }

# 生成热力图加到图像上
def generate_heatmap_visualization_fast(image_pil, masks, colormap=cv2.COLORMAP_JET):
    """
    优化的热力图生成函数,专为PIL Image优化
    
    Args:
        image_pil: PIL Image对象
        masks: [N, H, W] 或 [H, W]
        colormap: OpenCV色彩映射
    
    Returns:
        包含可视化结果的字典
    """
    # 快速转换为numpy
    image = np.asarray(image_pil)
    
    # 确保RGB格式
    if image.shape[2] == 4:
        image = image[:, :, :3]  # 直接切片,最快
    
    h, w = image.shape[:2]
    
    # 处理mask格式
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]
    
    n_masks = masks.shape[0]
    
    # 快速生成热力图
    if n_masks == 0:
        # 没有mask的情况
        return {
            'original': image,
            'heatmap': np.zeros((h, w), dtype=np.float32),
            'heatmap_colored': np.zeros((h, w, 3), dtype=np.uint8),
            'overlay': image.copy()
        }
    
    # 方法1：向量化生成热力图（推荐）
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 批量处理所有mask
    if n_masks > 1:
        # 多个mask：使用向量化操作
        masks_float = masks.astype(np.float32)
        if masks_float.max() > 1:
            masks_float = masks_float > 0
        
        # 直接求和
        heatmap = np.sum(masks_float, axis=0)
    else:
        # 单个mask：快速处理
        mask = masks[0]
        if mask.max() > 1:
            heatmap = (mask > 0).astype(np.float32)
        else:
            heatmap = mask.astype(np.float32)
    
    # 归一化
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val
    
    # 生成彩色热力图
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # 快速叠加（使用addWeighted,OpenCV优化）
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    return {
        'original': image,
        'heatmap': heatmap,
        'heatmap_colored': heatmap_colored,
        'overlay': overlay
    }

# 边界框和mask
def visualize_with_boxes_and_masks_fast(image_pil, boxes, masks, scores=None, 
                                        box_color=(0, 255, 0), box_thickness=2,
                                        mask_alpha=0.3):
    """
    优化的边界框和mask可视化函数
    
    Args:
        image_pil: PIL Image
        boxes: [N, 4] xyxy格式
        masks: [N, H, W]
        scores: 置信度列表（可选）
    """
    # 快速转换
    image = np.asarray(image_pil)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    h, w = image.shape[:2]
    
    # 确保boxes和masks格式正确
    if boxes is not None:
        boxes = np.array(boxes)
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]
    
    # 创建带边界框的图像
    image_with_boxes = image.copy()
    
    # 预定义颜色（用于mask）
    mask_colors = np.array([
        [255, 0, 0],    # 红
        [0, 255, 0],    # 绿
        [0, 0, 255],    # 蓝
        [255, 255, 0],  # 青
        [255, 0, 255],  # 洋红
        [0, 255, 255],  # 黄
    ], dtype=np.uint8)
    
    # 1. 绘制边界框（向量化）
    if boxes is not None and len(boxes) > 0:
        # 批量处理边界框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制矩形框
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), 
                         box_color, box_thickness)
            
            # 如果有置信度,显示在框上方
            if scores is not None and i < len(scores):
                score_text = f"{scores[i]:.2f}"
                # 计算文本位置
                text_x = x1
                text_y = max(y1 - 5, 10)
                
                # 先绘制文本背景（提高可读性）
                text_size = cv2.getTextSize(score_text, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image_with_boxes, 
                             (text_x, text_y - text_size[1] - 2),
                             (text_x + text_size[0], text_y + 2),
                             box_color, -1)
                
                # 绘制文本
                cv2.putText(image_with_boxes, score_text, 
                           (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 255, 255), 1)
    
    # 2. 创建mask叠加层
    mask_overlay = np.zeros_like(image, dtype=np.float32)
    
    # 批量处理mask
    for i in range(masks.shape[0]):
        mask = masks[i]
        
        # 快速二值化
        if mask.max() > 1:
            mask_bool = mask > 0
        else:
            mask_bool = mask.astype(bool)
        
        # 检查是否有有效像素
        if not np.any(mask_bool):
            continue
        
        color = mask_colors[i % len(mask_colors)]
        
        # 向量化着色
        rows, cols = np.where(mask_bool)
        if len(rows) > 0:
            mask_overlay[rows, cols] += color
    
    # 限制范围并转换为uint8
    mask_overlay = np.clip(mask_overlay, 0, 255).astype(np.uint8)
    
    # 3. 合并图像和mask
    combined = cv2.addWeighted(image_with_boxes, 1 - mask_alpha, 
                              mask_overlay, mask_alpha, 0)
    
    # 4. 创建最终组合图像（分屏显示）
    # 左侧：原始图像+边界框,右侧：mask叠加
    combined_display = np.hstack([image_with_boxes, combined])
    
    return {
        'image_with_boxes': image_with_boxes,
        'mask_overlay': mask_overlay,
        'combined': combined,
        'combined_display': combined_display
    }



def process_langsam_results_to_visualization(langsam_results, images_pil):
    """
    处理LangSAM结果并生成可视化
    
    Args:
        langsam_results: LangSAM.predict()的输出
        images_pil: 原始图像列表
        
    Returns:
        可视化结果列表
    """

    visualizations = []
    
    for idx, result in enumerate(langsam_results):
        if type(images_pil) == list:
            image_pil = images_pil[idx]
        else:
            image_pil = images_pil
        masks = result.get('mask', None)
        
        if masks is not None and len(masks) > 0:

            point4 = time.time()

            # 方法4：保留mask区域,其他区域为空白
            # vis4 = isolate_masked_regions_pil_fast(image_pil, masks)
            # 基于距离场的渐变效果
            vis4 = isolate_masked_regions_with_gradient(image_pil, masks, max_distance=200, falloff_type='linear')
            

            point5 = time.time()
            print(f"耗时: {point5 - point4:.2f} 秒")
            
            visualizations.append({
                'idx': idx,
                # 'colored_overlay': vis1,
                # 'heatmap': vis2,
                # 'box_overlay': vis3,
                'isolate_masked': vis4,
                'masks': masks,
                'boxes': result.get('boxes', None),
                'scores': result.get('scores', None)
            })
        else:
            visualizations.append({
                'idx': idx,
                'no_mask': True,
                'original': np.array(image_pil)
            })
    
    return visualizations

def process_single_mask_results(langsam_results, images_pil):
    """
    处理单个mask的LangSAM结果
    
    Args:
        langsam_results: 每个结果应该只包含一个mask
        images_pil: 原始图像列表或单个PIL图像
        
    Returns:
        可视化结果列表
    """
    import numpy as np
    import time
    
    visualizations = []
    
    for idx, result in enumerate(langsam_results):
        # 获取对应的图像
        if isinstance(images_pil, list):
            image_pil = images_pil[idx]
        else:
            image_pil = images_pil
        
        masks = result.get('masks', None)
        
        if masks is not None:
            # 提取单个mask
            if isinstance(masks, list):
                if len(masks) > 0:
                    single_mask = masks[0]
                else:
                    visualizations.append({
                        'idx': idx,
                        'no_mask': True,
                        'original': np.array(image_pil)
                    })
                    continue
            elif isinstance(masks, np.ndarray):
                if masks.ndim == 3:
                    single_mask = masks[0]  # 取第一个
                elif masks.ndim == 2:
                    single_mask = masks
                else:
                    raise ValueError(f"不支持的mask维度: {masks.ndim}")
            else:
                single_mask = masks
            
            # 确保是numpy数组
            if not isinstance(single_mask, np.ndarray):
                single_mask = np.array(single_mask)
            
            # 使用单个mask进行处理
            point4 = time.time()
            
            # 注意：isolate_masked_regions_with_gradient期望mask是[N,H,W]格式
            mask_for_processing = single_mask[np.newaxis, ...]
            vis4 = isolate_masked_regions_with_gradient(image_pil, mask_for_processing)
            
            point5 = time.time()
            print(f"图像 {idx} 处理耗时: {point5 - point4:.2f} 秒")
            
            visualizations.append({
                'idx': idx,
                'isolate_masked': vis4,
                'mask': single_mask,
                'box': result.get('boxes', [None])[0] if result.get('boxes') else None,
                'score': result.get('scores', [None])[0] if result.get('scores') else None,
                'original_image': np.array(image_pil)
            })
        else:
            visualizations.append({
                'idx': idx,
                'no_mask': True,
                'original': np.array(image_pil)
            })
    
    return visualizations


def display_visualizations(visualizations, figsize=(15, 10)):
    """
    显示可视化结果
    """
    for vis in visualizations:
        if 'no_mask' in vis:
            print(f"图像 {vis['idx']}: 没有检测到mask")
            plt.figure(figsize=figsize)
            plt.imshow(vis['original'])
            plt.title(f"Image {vis['idx']} - No masks detected")
            plt.axis('off')
            plt.show()
            continue
        
        idx = vis['idx']
        
        # 创建多子图显示
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        # 1. 原图
        axes[0].imshow(vis['colored_overlay']['original'])
        axes[0].set_title(f"Image {idx} - Original")
        axes[0].axis('off')
        
        # 2. 彩色mask叠加
        axes[1].imshow(vis['colored_overlay']['overlay'])
        axes[1].set_title("Colored Mask Overlay")
        axes[1].axis('off')
        
        # 3. 热力图
        axes[2].imshow(vis['heatmap']['heatmap'], cmap='hot')
        axes[2].set_title("Heatmap (raw)")
        axes[2].axis('off')
        
        # 4. 彩色热力图
        axes[3].imshow(cv2.cvtColor(vis['heatmap']['heatmap_colored'], cv2.COLOR_BGR2RGB))
        axes[3].set_title("Colored Heatmap")
        axes[3].axis('off')
        
        # 5. 热力图叠加
        axes[4].imshow(cv2.cvtColor(vis['heatmap']['overlay'], cv2.COLOR_BGR2RGB))
        axes[4].set_title("Heatmap Overlay")
        axes[4].axis('off')
        
        # 6. 带边界框的可视化
        axes[5].imshow(cv2.cvtColor(vis['box_overlay']['combined'], cv2.COLOR_BGR2RGB))
        axes[5].set_title("Boxes + Masks")
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 显示统计信息
        print(f"图像 {idx} 统计:")
        print(f"  - 检测到的mask数量: {len(vis['masks'])}")
        if 'scores' in vis and vis['scores'] is not None:
            print(f"  - 置信度: {vis['scores']}")
        print()



# from PIL import Image
# from lang_sam import LangSAM

# model = LangSAM(
#     gdino_model_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base", 
#     gdino_processor_ckpt_path="/data2/lizhengxue/WorkSpace/huchunning/Model-Database/grounding-dino-base")
# print('---------------------------1-------------------------')
# image_pil = Image.open("./assets/car.jpeg").convert("RGB")
# text_prompt_1 = "wheel."
# text_prompt_2 = "car."
# results_1 = model.predict([image_pil], [text_prompt_1])
# results_2 = model.predict([image_pil], [text_prompt_2])
# results = []
# results.append(results_1[0])
# results.append(results_2[0])
# print('---------------------------2-------------------------')



# # 生成可视化
# combine_result = [combine_masks_max_simple(results)]
# visualizations = process_langsam_results_to_visualization(combine_result, image_pil)
# print('---------------------------3-------------------------')
# # 显示结果
# # display_visualizations(visualizations)

# # # 也可以单独处理某个图像
# # single_vis = visualize_masks_on_image(
# #     image_pil, 
# #     results[0]['masks'], 
# #     results[0]['scores']
# # )

# # 保存可视化结果
# for i, vis in enumerate(visualizations):
#         # overlay_img = Image.fromarray(vis['colored_overlay']['overlay'])
#         # overlay_img.save(f"result_{i}_overlay.png")

#         overlay_img = Image.fromarray(vis['isolate_masked']['result'])
#         overlay_img.save(f"result_{i}_overlay.png")

        
#         # heatmap_img = Image.fromarray(vis['heatmap']['overlay'])
#         # heatmap_img.save(f"result_{i}_heatmap.png")

# print('---------------------------4-------------------------')