#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO模型评估脚本（基于JSON预测结果）
用于直接使用预测JSON文件评估YOLO模型在人体检测任务上的性能，生成HTML报告
支持标准COCO格式的预测结果和真实标注文件
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
from generate_pdf_report import generate_pdf_report


# ==================== 配置常量 ====================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型配置
MODEL_NAMES = ["YOLOv8-m"]
MODEL_COLORS = ['red']

# 评估参数
IOU_THRESHOLD = 0.5

# 数据路径配置
TRAIN_IMAGES_PATH = "/Users/leion/Downloads/annotations/coco_images/train2017"
GROUND_TRUTH_PATH = "./json/person_keypoints_train2017.json"
PREDICTION_PATH = "./json/model_keypoints_train2017.json"

# 输出目录配置
IMAGES_DIR = "./images"
CSV_DIR = "./csv"
HTML_DIR = "./html"


# ==================== 环境设置函数 ====================

def setup_environment():
    """设置WeasyPrint所需的环境变量"""
    env_vars = {
        "LD_LIBRARY_PATH": "/opt/homebrew/lib:",
        "DYLD_LIBRARY_PATH": "/opt/homebrew/lib:",
        "DYLD_FALLBACK_LIBRARY_PATH": "/opt/homebrew/lib:"
    }

    for var, path in env_vars.items():
        os.environ[var] = path + os.environ.get(var, "")

    print("✅ 环境变量已设置")


def ensure_output_dirs():
    """确保所有输出目录存在"""
    dirs = [IMAGES_DIR, CSV_DIR, HTML_DIR]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 创建目录: {dir_path}")

# ==================== 数据加载和处理函数 ====================

def load_json_data(file_path):
    """加载JSON文件"""
    try:
        print(f"📖 加载文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None


def load_predictions(predictions_path):
    """
    加载模型预测结果数据（标准COCO格式JSON文件）

    Args:
        predictions_path: 预测结果JSON文件路径

    Returns:
        image_to_predictions: 按image_id组织的预测结果字典
        images: 预测文件中的图像信息列表
    """
    data = load_json_data(predictions_path)
    if not data:
        return None, None

    if "images" not in data or "annotations" not in data:
        print("❌ 预测文件格式不正确，缺少images或annotations字段")
        return None, None

    images = data["images"]
    annotations = data["annotations"]

    # 初始化所有图像的预测列表
    image_to_predictions = {}
    for img_data in images:
        img_id = img_data["id"]
        image_to_predictions[img_id] = []

    # 添加预测框
    for anno in annotations:
        img_id = anno["image_id"]
        if "bbox" in anno:
            bbox = anno["bbox"]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            confidence = anno.get("bbox_score", 1.0)

            image_to_predictions[img_id].append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

    # 统计信息
    images_with_predictions = sum(1 for preds in image_to_predictions.values() if len(preds) > 0)
    images_without_predictions = len(image_to_predictions) - images_with_predictions

    print(f"📊 加载了 {len(images)} 张图像信息")
    print(f"📊 加载了 {len(annotations)} 个预测框")
    print(f"📊 其中 {images_with_predictions} 张图像有预测框，{images_without_predictions} 张图像无预测框")

    return image_to_predictions, images


def process_ground_truth(gt_data, target_image_ids=None):
    """
    处理真实标注数据，只保留人体类别的标注

    Args:
        gt_data: COCO格式的标注数据
        target_image_ids: 目标图像ID集合

    Returns:
        image_to_ground_truth: 图像ID到人体检测框的映射
        image_id_to_file: 图像ID到文件名的映射
    """
    images = gt_data.get("images", [])
    annotations = gt_data.get("annotations", [])

    print(f"📊 处理 {len(images)} 张图像和 {len(annotations)} 个标注")

    image_to_ground_truth = {}
    image_id_to_file = {}

    # 初始化目标图像
    if target_image_ids:
        for img_data in images:
            img_id = img_data["id"]
            if img_id in target_image_ids:
                file_name = img_data["file_name"]
                image_id_to_file[img_id] = file_name
                image_to_ground_truth[img_id] = []
    else:
        for img_data in images:
            img_id = img_data["id"]
            file_name = img_data["file_name"]
            image_id_to_file[img_id] = file_name

    # 添加人体类别的标注（category_id = 1）
    person_annotations = 0
    for anno in annotations:
        if anno.get("category_id") != 1:  # 只处理人体类别
            continue

        img_id = anno["image_id"]

        if target_image_ids and img_id not in target_image_ids:
            continue

        if img_id not in image_to_ground_truth:
            image_to_ground_truth[img_id] = []

        if "bbox" in anno:
            bbox = anno["bbox"]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            image_to_ground_truth[img_id].append([x1, y1, x2, y2])
            person_annotations += 1

    # 检查图像文件是否存在
    valid_image_ids = []
    for img_id in image_id_to_file.keys():
        file_name = image_id_to_file[img_id]
        image_path = os.path.join(TRAIN_IMAGES_PATH, file_name)
        if os.path.exists(image_path):
            valid_image_ids.append(img_id)

    # 统计信息
    if target_image_ids:
        images_with_person = sum(1 for boxes in image_to_ground_truth.values() if len(boxes) > 0)
        images_without_person = len(image_to_ground_truth) - images_with_person
        print(f"📊 目标图像总数: {len(target_image_ids)}")
        print(f"📊 实际存在的目标图像数量: {len(valid_image_ids)}")
        print(f"📊 人体标注数量: {person_annotations}")
        print(f"📊 其中 {images_with_person} 张图像有人体，{images_without_person} 张图像无人体")
    else:
        print(f"📊 标注文件中的图像总数: {len(image_id_to_file)}")
        print(f"📊 实际存在的图像数量: {len(valid_image_ids)}")
        print(f"📊 人体标注数量: {person_annotations}")
        print(f"📊 有人体标注的图像数量: {len(image_to_ground_truth)}")

    return image_to_ground_truth, image_id_to_file

# ==================== 计算函数 ====================

def calculate_iou(box1, box2):
    """
    计算两个框的IoU (交并比)

    Args:
        box1: [x1, y1, x2, y2] 格式的框坐标
        box2: [x1, y1, x2, y2] 格式的框坐标

    Returns:
        float: 交并比值 (0~1)
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(precision, recall):
    """
    计算平均精度(AP)

    Args:
        precision: 精确率数组
        recall: 召回率数组

    Returns:
        tuple: (ap, mrec, mpre) - 平均精度值和处理后的召回率、精度数组
    """
    # 添加起点(0,1)和终点(1,0)，确保PR曲线完整
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])

    # 平滑处理（取右侧最大值）
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    # 计算PR曲线下面积
    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices-1]) * mpre[indices])

    return ap, mrec, mpre

# ==================== 绘图函数 ====================

def plot_pr_curves(all_results):
    """绘制PR曲线并保存数据"""
    plt.figure(figsize=(10, 8))

    pr_data_by_model = {}

    for result in all_results:
        name = result['name']
        precision = result['precision']
        recall = result['recall']
        thresholds = result['pr_thresholds']
        ap = result['ap']

        # 收集PR曲线数据
        pr_data = []
        min_len = min(len(precision), len(recall), len(thresholds))
        for i in range(min_len):
            pr_data.append({
                'threshold': thresholds[i],
                'precision': precision[i],
                'recall': recall[i]
            })

        pr_data_by_model[name] = pr_data

        # 绘制PR曲线
        plt.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.4f})')

        # 生成关键点数据
        key_points = _generate_pr_key_points(precision, recall, thresholds)
        result['pr_key_points'] = key_points

    # 设置图表样式
    plt.title('精确率-召回率曲线比较', fontsize=16)
    plt.xlabel('召回率', fontsize=14)
    plt.ylabel('精确率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left', fontsize=12)

    # 保存图像
    plt.savefig(os.path.join(IMAGES_DIR, "pr_curve_comparison.png"), dpi=300, bbox_inches='tight')

    # 导出CSV数据
    for name, pr_data in pr_data_by_model.items():
        df = pd.DataFrame(pr_data)
        csv_name = os.path.join(CSV_DIR, f"{name}_pr_curve_data.csv")
        df.to_csv(csv_name, index=False)
        print(f"✅ {name}的PR曲线数据已保存至: {csv_name}")

    return plt.gcf()


def _generate_pr_key_points(precision, recall, thresholds):
    """生成PR曲线关键点"""
    key_points = []

    # 最佳F1点
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    if f1_scores:
        idx = np.argmax(f1_scores)
        key_points.append({
            'idx': idx,
            'recall': recall[idx],
            'precision': precision[idx],
            'threshold': thresholds[idx] if idx < len(thresholds) else 1.0,
            'f1': f1_scores[idx],
            'special': "最佳F1点"
        })

    # 高精确率点
    high_prec_points = [(i, r, p) for i, (r, p) in enumerate(zip(recall, precision)) if p > 0.9]
    if high_prec_points:
        idx, r, p = max(high_prec_points, key=lambda x: x[1])
        key_points.append({
            'idx': idx,
            'recall': r,
            'precision': p,
            'threshold': thresholds[idx] if idx < len(thresholds) else 1.0,
            'special': "高精确率P>0.9"
        })

    return key_points

def plot_roc_curves(all_results):
    """绘制ROC曲线并保存数据"""
    plt.figure(figsize=(10, 8))

    roc_data_by_model = {}
    has_valid_data = any(not np.isnan(result.get('auc', np.nan)) for result in all_results)

    if has_valid_data:
        for result in all_results:
            name = result['name']
            fpr = result['fpr']
            tpr = result['tpr']
            roc_auc = result['auc']
            thresholds = result['roc_thresholds']

            # 收集ROC曲线数据
            roc_data = []
            min_len = min(len(fpr), len(tpr), len(thresholds))
            for i in range(min_len):
                roc_data.append({
                    'fpr': fpr[i],
                    'tpr': tpr[i],
                    'threshold': thresholds[i]
                })

            roc_data_by_model[name] = roc_data

            # 绘制ROC曲线
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.4f})')

            # 生成关键点数据
            key_points = _generate_roc_key_points(fpr, tpr, thresholds)
            result['roc_key_points'] = key_points

        # 设置图表样式
        plt.title('ROC曲线比较', fontsize=16)
        plt.xlabel('假阳性率 (FPR)', fontsize=14)
        plt.ylabel('真阳性率 (TPR)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right', fontsize=12)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # 对角线

        # 保存图像
        plt.savefig(os.path.join(IMAGES_DIR, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')

        # 导出CSV数据
        for name, roc_data in roc_data_by_model.items():
            df = pd.DataFrame(roc_data)
            csv_name = os.path.join(CSV_DIR, f"{name}_roc_curve_data.csv")
            df.to_csv(csv_name, index=False)
            print(f"✅ {name}的ROC曲线数据已保存至: {csv_name}")
    else:
        plt.title('无有效ROC曲线数据', fontsize=16)
        plt.xlabel('假阳性率', fontsize=14)
        plt.ylabel('真阳性率', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.text(0.5, 0.5, '没有有效的ROC曲线数据\n(可能是因为没有负样本)',
                ha='center', fontsize=14, color='red')

    return plt.gcf()


def _generate_roc_key_points(fpr, tpr, thresholds):
    """生成ROC曲线关键点"""
    key_points = []

    # 约登指数最大点
    youden = [t - f for f, t in zip(fpr, tpr)]
    if youden:
        idx = np.argmax(youden)
        key_points.append({
            'idx': idx,
            'fpr': fpr[idx],
            'tpr': tpr[idx],
            'threshold': thresholds[idx] if idx < len(thresholds) else float('inf'),
            'youden': youden[idx],
            'special': "Best Youden"
        })

    # 高特异性点
    high_spec_points = [(i, f, t) for i, (f, t) in enumerate(zip(fpr, tpr)) if f < 0.01]
    if high_spec_points:
        idx, f, t = max(high_spec_points, key=lambda x: x[2])
        key_points.append({
            'idx': idx,
            'fpr': f,
            'tpr': t,
            'threshold': thresholds[idx] if idx < len(thresholds) else float('inf'),
            'special': "High Spec (FPR<0.01)"
        })

    return key_points

def plot_regression_bias(all_results):
    """绘制回归偏差直方图"""
    plt.figure(figsize=(8.27, 6))

    for result in all_results:
        biases = []
        for pred in result['predictions']:
            if pred['iou'] > 0:
                bias = max(0, int(round((1/pred['iou'] - 1) * 100)))
                if bias <= 60:
                    biases.append(bias)

        if biases:
            plt.hist(biases, bins=range(0, 65, 4), alpha=0.7,
                     label=result['name'], color=result.get('color', 'blue'))

    plt.title('回归偏差分布', fontsize=14)
    plt.xlabel('偏差 (像素)', fontsize=11)
    plt.ylabel('数量', fontsize=11)
    plt.xlim([0, 64])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(IMAGES_DIR, "regression_bias.png"), dpi=300, bbox_inches='tight')


def plot_confidence_distribution(all_results):
    """绘制置信度分布直方图"""
    plt.figure(figsize=(8.27, 6))

    for result in all_results:
        confidences = [pred['confidence'] for pred in result['predictions']]

        if confidences:
            plt.hist(confidences, bins=np.arange(0.25, 1.01, 0.05), alpha=0.7,
                     label=result['name'], color=result.get('color', 'blue'))

    plt.title('置信度分布', fontsize=14)
    plt.xlabel('置信度', fontsize=11)
    plt.ylabel('数量', fontsize=11)
    plt.xlim([0.25, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(IMAGES_DIR, "confidence_distribution.png"), dpi=300, bbox_inches='tight')



def evaluate_with_predictions(predictions, ground_truth, model_name):
    """
    使用预测结果评估模型性能

    Args:
        predictions: 预测结果字典，按image_id组织
        ground_truth: 真实标注字典，按image_id组织
        model_name: 模型名称

    Returns:
        tuple: (precision, recall, fpr, tpr, ap, roc_auc, predictions, pr_thresholds, roc_thresholds)
    """
    print(f"🔍 评估模型: {model_name}")

    try:
        # 计算真实标签总数
        num_gts = sum(len(boxes) for boxes in ground_truth.values())
        print(f"📊 真实标注框总数: {num_gts}")

        # 处理每个图像
        all_image_ids = set(predictions.keys()) & set(ground_truth.keys())
        print(f"📊 共有 {len(all_image_ids)} 张图像同时存在于预测和真实标注中")

        # 存储所有预测结果
        all_predictions = []

        progress_bar = tqdm(all_image_ids, desc=f"处理 {model_name}")
        for img_id in progress_bar:
            gt_boxes = ground_truth[img_id]
            pred_data = predictions[img_id]

            # 为每个预测框寻找最佳匹配的真实框
            for pred in pred_data:
                pred_box = pred["bbox"]
                confidence = pred["confidence"]

                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    current_iou = calculate_iou(pred_box, gt_box)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = j

                all_predictions.append({
                    'img_id': img_id,
                    'confidence': confidence,
                    'iou': best_iou,
                    'matched_gt': best_iou >= IOU_THRESHOLD,
                    'matched_gt_idx': best_gt_idx if best_gt_idx >= 0 else None
                })

        print(f"📊 共有 {len(all_predictions)} 个预测框")
        
        # 按置信度从高到低排序预测结果
        sorted_preds = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
        
        # 准备计算PR曲线和ROC曲线
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))
        matched_gt = set()  # 已匹配的真实标签集合
        
        # 计算TP和FP
        for i, pred in enumerate(sorted_preds):
            img_id = pred['img_id']
            gt_idx = pred['matched_gt_idx']
            iou = pred['iou']
            
            img_gt_key = f"{img_id}_{gt_idx}" if gt_idx is not None else None
            
            if iou >= IOU_THRESHOLD and pred['matched_gt'] and img_gt_key is not None:
                if img_gt_key not in matched_gt:
                    tp[i] = 1  # 真阳性
                    matched_gt.add(img_gt_key)
                else:
                    fp[i] = 1  # 重复检测，假阳性
            else:
                fp[i] = 1  # 未匹配到真实框或IoU不足，假阳性
        
        # 计算累积值
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / num_gts
        
        # 计算AP
        ap, interp_recall, interp_precision = calculate_ap(precision, recall)
        
        # 计算ROC曲线
        confidences = [pred['confidence'] for pred in sorted_preds]
        labels = [1 if pred['matched_gt'] else 0 for pred in sorted_preds]
        
        fpr, tpr, roc_thresholds = roc_curve(labels, confidences)
        roc_auc = auc(fpr, tpr)
        
        # 计算PR曲线的阈值
        _, _, pr_thresholds = precision_recall_curve(labels, confidences, pos_label=1)
        
        print(f"✅ {model_name} 评估完成: AP={ap:.4f}, AUC={roc_auc:.4f}")
        print(f"📊 真阳性总数: {tp_cumsum[-1] if len(tp_cumsum) > 0 else 0}")
        print(f"📊 假阳性总数: {fp_cumsum[-1] if len(fp_cumsum) > 0 else 0}")

        return interp_precision, interp_recall, fpr, tpr, ap, roc_auc, sorted_preds, pr_thresholds, roc_thresholds

    except Exception as e:
        print(f"❌ 评估 {model_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



def generate_key_points_tables(all_results):
    """
    为PR曲线和ROC曲线生成关键点表格并导出到CSV和HTML
    
    参数:
        all_results: 所有模型的评估结果列表
    """
    print("生成关键点表格...")
    
    # 创建CSV文件
    pr_csv_path = os.path.join(CSV_DIR, "pr_key_points_table.csv")
    roc_csv_path = os.path.join(CSV_DIR, "roc_key_points_table.csv")
    
    # 导出PR曲线关键点到CSV
    with open(pr_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'special', 'precision', 'recall', 'threshold', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            model_name = result['name']
            if 'pr_key_points' in result:
                for point in result.get('pr_key_points', []):
                    row = {
                        'model': model_name,
                        'special': point.get('special', ''),
                        'precision': round(point.get('precision', 0), 4),
                        'recall': round(point.get('recall', 0), 4),
                        'threshold': round(point.get('threshold', 0), 4),
                        'f1': round(point.get('f1', 0), 4) if 'f1' in point else ''
                    }
                    writer.writerow(row)
    
    # 导出ROC曲线关键点到CSV
    with open(roc_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'special', 'tpr', 'fpr', 'threshold', 'youden']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            model_name = result['name']
            if 'roc_key_points' in result:
                for point in result.get('roc_key_points', []):
                    row = {
                        'model': model_name,
                        'special': point.get('special', ''),
                        'tpr': round(point.get('tpr', 0), 4),
                        'fpr': round(point.get('fpr', 0), 4),
                        'threshold': round(point.get('threshold', 0), 4),
                        'youden': round(point.get('youden', 0), 4) if 'youden' in point else ''
                    }
                    writer.writerow(row)
    
    print(f"✅ PR曲线关键点已保存至: {pr_csv_path}")
    print(f"✅ ROC曲线关键点已保存至: {roc_csv_path}")
    
    # 生成HTML表格
    html_output_path = os.path.join(HTML_DIR, 'key_points_tables.html')
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>模型评估关键点表格</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            h2 { color: #333; margin-top: 30px; }
            .highlight { background-color: #ffffcc; }
        </style>
    </head>
    <body>
        <h1>模型评估关键点表格</h1>
    """
    
    # 添加PR曲线关键点表格
    html_content += """
        <h2>PR曲线关键点</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>描述</th>
                <th>精确率</th>
                <th>召回率</th>
                <th>阈值</th>
                <th>F1分数</th>
            </tr>
    """
    
    for result in all_results:
        model_name = result['name']
        if 'pr_key_points' in result:
            for point in result.get('pr_key_points', []):
                special = point.get('special', '')
                precision = round(point.get('precision', 0), 4)
                recall = round(point.get('recall', 0), 4)
                threshold = round(point.get('threshold', 0), 4)
                f1 = round(point.get('f1', 0), 4) if 'f1' in point else ''
                
                # 高亮最佳F1点或高精确率点
                highlight = 'class="highlight"' if '最佳F1点' in special or '高精确率' in special else ''
                
                html_content += f"""
                <tr {highlight}>
                    <td>{model_name}</td>
                    <td>{special}</td>
                    <td>{precision}</td>
                    <td>{recall}</td>
                    <td>{threshold}</td>
                    <td>{f1}</td>
                </tr>
                """
    
    html_content += """
        </table>
    """
    
    # 添加ROC曲线关键点表格
    html_content += """
        <h2>ROC曲线关键点</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>描述</th>
                <th>真阳性率(TPR)</th>
                <th>假阳性率(FPR)</th>
                <th>阈值</th>
                <th>约登指数</th>
            </tr>
    """
    
    for result in all_results:
        model_name = result['name']
        if 'roc_key_points' in result:
            for point in result.get('roc_key_points', []):
                special = point.get('special', '')
                tpr = round(point.get('tpr', 0), 4)
                fpr = round(point.get('fpr', 0), 4)
                threshold = round(point.get('threshold', 0), 4)
                youden = round(point.get('youden', 0), 4) if 'youden' in point else ''
                
                # 高亮最佳约登指数点或高特异性点
                highlight = 'class="highlight"' if 'Best Youden' in special or 'High Spec' in special else ''
                
                html_content += f"""
                <tr {highlight}>
                    <td>{model_name}</td>
                    <td>{special}</td>
                    <td>{tpr}</td>
                    <td>{fpr}</td>
                    <td>{threshold}</td>
                    <td>{youden}</td>
                </tr>
                """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 关键点HTML表格已保存至: {html_output_path}")

# ==================== 主函数 ====================

def main():
    """主函数 - 基于JSON文件的YOLO模型评估"""
    global IOU_THRESHOLD, TRAIN_IMAGES_PATH, GROUND_TRUTH_PATH, PREDICTION_PATH

    print("🎯 YOLO模型评估（基于JSON预测结果）")
    print("📋 支持标准COCO格式的预测结果和真实标注文件")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型评估脚本')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD, help='IoU阈值')
    parser.add_argument('--train-path', type=str, default=TRAIN_IMAGES_PATH, help='图像路径')
    parser.add_argument('--gt-path', type=str, default=GROUND_TRUTH_PATH, help='真实标注文件')
    parser.add_argument('--pred-path', type=str, default=PREDICTION_PATH, help='预测结果文件')

    args = parser.parse_args()

    # 更新配置
    IOU_THRESHOLD = args.iou
    TRAIN_IMAGES_PATH = args.train_path
    GROUND_TRUTH_PATH = args.gt_path
    PREDICTION_PATH = args.pred_path

    # 初始化环境
    ensure_output_dirs()
    setup_environment()

    print(f"\n📊 评估配置:")
    print(f"   📂 图像路径: {TRAIN_IMAGES_PATH}")
    print(f"   📝 标注文件: {GROUND_TRUTH_PATH}")
    print(f"   🤖 预测文件: {PREDICTION_PATH}")
    print(f"   🎯 IoU阈值: {IOU_THRESHOLD}")

    # 检查文件存在性
    for file_path, name in [(GROUND_TRUTH_PATH, "标注文件"), (PREDICTION_PATH, "预测文件")]:
        if not os.path.exists(file_path):
            print(f"❌ {name}不存在: {file_path}")
            sys.exit(1)

    # 加载数据
    print("\n🤖 步骤1: 加载预测数据...")
    predictions, _ = load_predictions(PREDICTION_PATH)
    if not predictions:
        print("❌ 加载预测数据失败")
        sys.exit(1)

    pred_image_ids = set(predictions.keys())
    print(f"   📋 预测图像数量: {len(pred_image_ids)}")

    # 加载标注数据
    print("\n📖 步骤2: 加载标注数据...")
    gt_data = load_json_data(GROUND_TRUTH_PATH)
    if not gt_data:
        print("❌ 加载标注数据失败")
        sys.exit(1)

    ground_truth, _ = process_ground_truth(gt_data, target_image_ids=pred_image_ids)

    # 评估模型
    print(f"\n📈 步骤3: 开始性能评估...")
    common_images = len(set(predictions.keys()) & set(ground_truth.keys()))
    print(f"   🔍 共有 {common_images} 张图像将被评估")

    all_results = []
    for i, model_name in enumerate(MODEL_NAMES):
        result = evaluate_with_predictions(predictions, ground_truth, model_name)
        if result:
            precision, recall, fpr, tpr, ap, roc_auc, pred_results, pr_thresholds, roc_thresholds = result

            model_result = {
                'name': model_name,
                'color': MODEL_COLORS[i] if i < len(MODEL_COLORS) else 'blue',
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'tpr': tpr,
                'ap': ap,
                'auc': roc_auc if not np.isnan(roc_auc) else 0.0,
                'predictions': pred_results,
                'pr_thresholds': pr_thresholds,
                'roc_thresholds': roc_thresholds
            }
            all_results.append(model_result)

    if not all_results:
        print("❌ 模型评估失败")
        sys.exit(1)

    # 生成图表和报告
    print(f"\n📊 步骤4: 生成图表和报告...")

    plot_pr_curves(all_results)
    plot_roc_curves(all_results)
    plot_regression_bias(all_results)
    plot_confidence_distribution(all_results)
    generate_key_points_tables(all_results)

    # 生成HTML报告
    html_output_path = os.path.join(HTML_DIR, 'weasy_evaluation_results.html')
    generate_pdf_report(all_results, output_path=html_output_path, save_html=True)

    print(f"\n🎉 评估完成！")
    print(f"📈 图表: {IMAGES_DIR}/")
    print(f"📊 CSV: {CSV_DIR}/")
    print(f"📄 HTML报告: {html_output_path}")
    print(f"📋 关键点表格: {HTML_DIR}/key_points_tables.html")
    print(f"\n✅ HTML评估报告已成功生成！")

if __name__ == "__main__":
    main()