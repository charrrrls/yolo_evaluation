#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COCO数据集关键点可视化脚本
可视化COCO数据集中的图片、边界框、分割轮廓和关键点
"""

import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import cv2
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # Mac优先使用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# COCO关键点定义
KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩",
    "左肘", "右肘", "左腕", "右腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝"
]

# COCO关键点连接关系定义（对，用于画线）
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 面部
    (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (12, 14), (13, 15), (14, 16)  # 腿部
]

# 为关键点设置不同颜色
KEYPOINT_COLORS = [
    'red', 'blue', 'blue', 'cyan', 'cyan',  # 面部
    'yellow', 'yellow', 'green', 'green', 'orange', 'orange',  # 上肢
    'purple', 'purple', 'magenta', 'magenta', 'pink', 'pink'  # 下肢
]

# 连接线颜色
CONNECTION_COLORS = [
    'gray', 'gray', 'gray', 'gray',  # 面部
    'green', 'green', 'green', 'green',  # 手臂
    'red', 'blue', 'blue', 'red',  # 躯干
    'magenta', 'magenta', 'pink', 'pink'  # 腿部
]

def load_coco_annotations(annotation_path):
    """加载COCO标注文件"""
    print(f"加载标注文件: {annotation_path}")
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"标注文件已加载，包含 {len(annotations['annotations'])} 个标注")
    return annotations

def get_image_and_annotations(annotations, image_id=None):
    """获取指定图片ID的图片信息和标注，如未指定则随机选择"""
    # 图片信息字典
    images_dict = {img['id']: img for img in annotations['images']}
    
    # 按图片ID分组的标注
    annotations_by_image = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 如果未指定图片ID，则随机选择一张至少有一个人的图片
    if image_id is None:
        valid_image_ids = list(annotations_by_image.keys())
        if not valid_image_ids:
            raise Exception("未找到有效的图片")
        image_id = random.choice(valid_image_ids)
    elif image_id not in annotations_by_image:
        raise Exception(f"图片ID {image_id} 未找到或没有人体标注")
    
    # 获取图片信息和标注
    image_info = images_dict[image_id]
    image_anns = annotations_by_image[image_id]
    
    print(f"已选择图片 {image_info['file_name']}，包含 {len(image_anns)} 个人体标注")
    return image_info, image_anns

def visualize_annotations(image_path, image_info, annotations):
    """可视化图片、边界框、分割轮廓和关键点"""
    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"无法加载图片: {image_path}")
    
    # 转换为RGB（OpenCV加载为BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # 为每个标注可视化边界框、分割轮廓和关键点
    for i, ann in enumerate(annotations):
        # 随机颜色（保持一致性）
        color = np.random.rand(3,)
        
        # 绘制边界框
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标注ID文本
        ax.text(x, y - 10, f"ID: {ann['id']}", color=color, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # 绘制分割轮廓
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                # 将分割点转换为路径格式
                verts = np.array(seg).reshape(-1, 2)
                codes = np.ones(len(verts), dtype=np.uint8) * Path.LINETO
                codes[0] = Path.MOVETO
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=2, alpha=0.7)
                ax.add_patch(patch)
        
        # 绘制关键点
        if 'keypoints' in ann:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            for j, (x, y, v) in enumerate(keypoints):
                if v > 0:  # 只绘制可见或被标注的关键点
                    # v=1: 被标注但不可见，v=2: 可见
                    alpha = 0.5 if v == 1 else 1.0
                    ax.plot(x, y, 'o', markersize=8, color=KEYPOINT_COLORS[j], alpha=alpha)
                    # 添加关键点名称
                    ax.text(x + 5, y + 5, KEYPOINT_NAMES[j], fontsize=8, color=KEYPOINT_COLORS[j], 
                            bbox=dict(facecolor='white', alpha=0.5))
            
            # 绘制关键点之间的连接
            for k, (i1, i2) in enumerate(KEYPOINT_CONNECTIONS):
                x1, y1, v1 = keypoints[i1]
                x2, y2, v2 = keypoints[i2]
                if v1 > 0 and v2 > 0:  # 只有两个关键点都被标注才绘制连接
                    ax.plot([x1, x2], [y1, y2], '-', linewidth=2, color=CONNECTION_COLORS[k], alpha=0.7)
    
    # 设置标题和其他属性
    ax.set_title(f"图片: {image_info['file_name']} (ID: {image_info['id']})")
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='COCO数据集关键点可视化')
    parser.add_argument('--annotations', type=str, 
                        default='/Users/leion/Charles/work/annotations/person_keypoints_train2017.json',
                        help='COCO标注文件路径')
    parser.add_argument('--images_dir', type=str, 
                        default='/Users/leion/Downloads/annotations/coco_images/train2017',
                        help='COCO图片目录路径')
    parser.add_argument('--image_id', type=int, default=None,
                        help='要可视化的图片ID，不指定则随机选择')
    parser.add_argument('--output', type=str, default=None,
                        help='可视化结果保存路径，不指定则显示在屏幕上')
    
    args = parser.parse_args()
    
    # 加载标注
    annotations = load_coco_annotations(args.annotations)
    
    # 获取图片信息和标注
    image_info, image_anns = get_image_and_annotations(annotations, args.image_id)
    
    # 构建图片路径
    image_path = os.path.join(args.images_dir, image_info['file_name'])
    
    # 可视化
    try:
        fig = visualize_annotations(image_path, image_info, image_anns)
        
        # 保存或显示结果
        if args.output:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {args.output}")
        else:
            plt.show()
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
COCO数据集关键点可视化脚本
可视化COCO数据集中的图片、边界框、分割轮廓和关键点
"""

import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import cv2

# COCO关键点定义
KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩",
    "左肘", "右肘", "左腕", "右腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝"
]

# COCO关键点连接关系定义（对，用于画线）
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 面部
    (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (12, 14), (13, 15), (14, 16)  # 腿部
]

# 为关键点设置不同颜色
KEYPOINT_COLORS = [
    'red', 'blue', 'blue', 'cyan', 'cyan',  # 面部
    'yellow', 'yellow', 'green', 'green', 'orange', 'orange',  # 上肢
    'purple', 'purple', 'magenta', 'magenta', 'pink', 'pink'  # 下肢
]

# 连接线颜色
CONNECTION_COLORS = [
    'gray', 'gray', 'gray', 'gray',  # 面部
    'green', 'green', 'green', 'green',  # 手臂
    'red', 'blue', 'blue', 'red',  # 躯干
    'magenta', 'magenta', 'pink', 'pink'  # 腿部
]

def load_coco_annotations(annotation_path):
    """加载COCO标注文件"""
    print(f"加载标注文件: {annotation_path}")
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"标注文件已加载，包含 {len(annotations['annotations'])} 个标注")
    return annotations

def get_image_and_annotations(annotations, image_id=None):
    """获取指定图片ID的图片信息和标注，如未指定则随机选择"""
    # 图片信息字典
    images_dict = {img['id']: img for img in annotations['images']}
    
    # 按图片ID分组的标注
    annotations_by_image = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 如果未指定图片ID，则随机选择一张至少有一个人的图片
    if image_id is None:
        valid_image_ids = list(annotations_by_image.keys())
        if not valid_image_ids:
            raise Exception("未找到有效的图片")
        image_id = random.choice(valid_image_ids)
    elif image_id not in annotations_by_image:
        raise Exception(f"图片ID {image_id} 未找到或没有人体标注")
    
    # 获取图片信息和标注
    image_info = images_dict[image_id]
    image_anns = annotations_by_image[image_id]
    
    print(f"已选择图片 {image_info['file_name']}，包含 {len(image_anns)} 个人体标注")
    return image_info, image_anns

def visualize_annotations(image_path, image_info, annotations):
    """可视化图片、边界框、分割轮廓和关键点"""
    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"无法加载图片: {image_path}")
    
    # 转换为RGB（OpenCV加载为BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # 为每个标注可视化边界框、分割轮廓和关键点
    for i, ann in enumerate(annotations):
        # 随机颜色（保持一致性）
        color = np.random.rand(3,)
        
        # 绘制边界框
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标注ID文本
        ax.text(x, y - 10, f"ID: {ann['id']}", color=color, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # 绘制分割轮廓
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                # 将分割点转换为路径格式
                verts = np.array(seg).reshape(-1, 2)
                codes = np.ones(len(verts), dtype=np.uint8) * Path.LINETO
                codes[0] = Path.MOVETO
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=2, alpha=0.7)
                ax.add_patch(patch)
        
        # 绘制关键点
        if 'keypoints' in ann:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            for j, (x, y, v) in enumerate(keypoints):
                if v > 0:  # 只绘制可见或被标注的关键点
                    # v=1: 被标注但不可见，v=2: 可见
                    alpha = 0.5 if v == 1 else 1.0
                    ax.plot(x, y, 'o', markersize=8, color=KEYPOINT_COLORS[j], alpha=alpha)
                    # 添加关键点名称
                    ax.text(x + 5, y + 5, KEYPOINT_NAMES[j], fontsize=8, color=KEYPOINT_COLORS[j], 
                            bbox=dict(facecolor='white', alpha=0.5))
            
            # 绘制关键点之间的连接
            for k, (i1, i2) in enumerate(KEYPOINT_CONNECTIONS):
                x1, y1, v1 = keypoints[i1]
                x2, y2, v2 = keypoints[i2]
                if v1 > 0 and v2 > 0:  # 只有两个关键点都被标注才绘制连接
                    ax.plot([x1, x2], [y1, y2], '-', linewidth=2, color=CONNECTION_COLORS[k], alpha=0.7)
    
    # 设置标题和其他属性
    ax.set_title(f"图片: {image_info['file_name']} (ID: {image_info['id']})")
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='COCO数据集关键点可视化')
    parser.add_argument('--annotations', type=str, 
                        default='/Users/leion/Charles/work/annotations/person_keypoints_train2017.json',
                        help='COCO标注文件路径')
    parser.add_argument('--images_dir', type=str, 
                        default='/Users/leion/Downloads/annotations/coco_images/train2017',
                        help='COCO图片目录路径')
    parser.add_argument('--image_id', type=int, default=None,
                        help='要可视化的图片ID，不指定则随机选择')
    parser.add_argument('--output', type=str, default=None,
                        help='可视化结果保存路径，不指定则显示在屏幕上')
    
    args = parser.parse_args()
    
    # 加载标注
    annotations = load_coco_annotations(args.annotations)
    
    # 获取图片信息和标注
    image_info, image_anns = get_image_and_annotations(annotations, args.image_id)
    
    # 构建图片路径
    image_path = os.path.join(args.images_dir, image_info['file_name'])
    
    # 可视化
    try:
        fig = visualize_annotations(image_path, image_info, image_anns)
        
        # 保存或显示结果
        if args.output:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {args.output}")
        else:
            plt.show()
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()