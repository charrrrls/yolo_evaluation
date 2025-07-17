# YOLO模型评估脚本（基于JSON预测结果）🎯

## 概述
本脚本专门用于基于预测JSON文件评估YOLO模型在人体检测任务上的性能，无需进行实时推理。

## 📁 文件要求

### 1. 预测结果文件
- **路径**: `./json/model_keypoints_train2017.json`
- **格式**: 标准COCO格式JSON
- **必需字段**:
  - `images`: 图像信息列表
  - `annotations`: 预测框列表，每个预测框包含：
    - `image_id`: 图像ID
    - `bbox`: 边界框 [x, y, width, height]
    - `bbox_score`: 预测置信度 ⭐

### 2. 真实标注文件
- **路径**: `./json/person_keypoints_train2017.json`
- **格式**: 标准COCO格式JSON
- **必需字段**:
  - `images`: 图像信息列表
  - `annotations`: 真实标注列表，只处理`category_id=1`(人体类别)

### 3. 图像文件
- **路径**: `/Users/leion/Downloads/annotations/coco_images/train2017/`
- **说明**: 根据`image_id`对应的图像文件

## 🚀 使用方法

### 基本运行
```bash
python eval_yolo_coco.py
```

### 自定义参数
```bash
python eval_yolo_coco.py \
    --iou 0.5 \
    --gt-path "./json/person_keypoints_train2017.json" \
    --pred-path "./json/model_keypoints_train2017.json" \
    --train-path "/Users/leion/Downloads/annotations/coco_images/train2017"
```

## 📊 输出结果

脚本将生成以下文件：

### 图表文件 (`./images/`)
- `pr_curve_comparison.png` - PR曲线对比图
- `roc_curve_comparison.png` - ROC曲线对比图
- `regression_bias.png` - 回归偏差分布图
- `confidence_distribution.png` - 置信度分布图

### 数据文件 (`./csv/`)
- `{模型名}_pr_curve_data.csv` - PR曲线原始数据
- `{模型名}_roc_curve_data.csv` - ROC曲线原始数据
- `pr_key_points_table.csv` - PR曲线关键点表格
- `roc_key_points_table.csv` - ROC曲线关键点表格

### 报告文件
- `./html/weasy_evaluation_results.html` - 详细评估报告
- `./html/key_points_tables.html` - 关键点表格

## 🔧 主要特性

1. **📄 JSON文件处理**: 直接处理预测JSON，无需实时推理
2. **📊 全面评估**: 计算AP、AUC、精确率、召回率等指标
3. **📈 可视化图表**: 生成PR曲线、ROC曲线等对比图
4. **🎯 关键点分析**: 自动识别最优工作点
5. **📋 详细报告**: 生成HTML格式的完整评估报告

## 💡 注意事项

- 确保JSON文件格式正确，符合COCO标准
- 预测文件必须包含`bbox_score`字段作为置信度
- 只评估人体检测任务（category_id=1）
- 图像文件路径需要与标注中的`file_name`对应 