#!/bin/bash

# 配置参数 - YOLO评估参数
EVAL_SAMPLES=20          # 评估的图像数量
EVAL_IOU_THRESHOLD=0.5   # 判定正确检测的IoU阈值
TRAIN_IMAGES_PATH="/Users/leion/Downloads/annotations/coco_images/train2017"  # 训练图像路径
ANNOTATIONS_PATH="./json/person_images.json"                                 # 标注文件路径

echo "=== 开始YOLO模型评估 ==="

# 运行模型评估
echo "运行模型评估..."
python eval_yolo_coco.py \
  --samples $EVAL_SAMPLES \
  --iou $EVAL_IOU_THRESHOLD \
  --train-path "$TRAIN_IMAGES_PATH" \
  --annotations "$ANNOTATIONS_PATH"

echo "=== 评估完成 ==="
echo "HTML报告已生成，请查看 ./html/weasy_evaluation_results.html"