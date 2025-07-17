# YOLO Evaluation Tool

一个全面的工具，用于评估和比较不同 YOLO 模型在人体检测任务上的性能。该工具计算各种性能指标（AP、AUC）并生成详细报告。

## 功能特点

- 比较多种 YOLO 模型（YOLOv8-n、YOLOv8-s、YOLOv8-m 等）
- 计算关键指标，如平均精度（AP）和曲线下面积（AUC）
- 生成详细的图表，包括精确率-召回率曲线、ROC 曲线等
- 创建全面的 PDF 报告，包括两种格式：
  - 传统的基于 matplotlib 的报告
  - 增强的基于 HTML/CSS 的报告（需要 WeasyPrint）
- 性能基准测试，评估不同批处理大小下的吞吐量、延迟和内存使用情况

## 项目结构

项目采用清晰的目录结构组织各类文件：

```
yolo_evaluation/
├── images/            # 所有图表和图像文件
├── csv/               # 数据表格和曲线数据点
├── json/              # JSON 格式数据文件
├── pdf/               # 生成的 PDF 报告
├── html/              # HTML 报告和模板
├── models/            # YOLO 模型文件
├── eval_yolo_coco.py  # 主要评估脚本
├── benchmark_yolo.py  # 性能基准测试脚本
└── ...                # 其他脚本和文档
```

## 安装

1. 克隆此仓库：
```
git clone <repository-url>
cd yolo_evaluation
```

2. 安装依赖：
```
pip install -r requirements.txt
```

3. 安装 WeasyPrint（用于增强的 PDF 报告）：
```
pip install weasyprint
```

注意：WeasyPrint 有额外的系统依赖项。请参阅 [WeasyPrint 安装指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation) 获取特定平台的安装说明。

## 使用方法

### 模型评估

1. 准备数据：
   - 将 YOLO 模型文件放入 `models/` 目录
   - 确保标注文件可用（默认：`json/person_images.json`）

2. 运行评估：
```
python eval_yolo_coco.py
```

3. 脚本将：
   - 评估所有模型在数据集上的性能
   - 生成性能图表（保存到 `images/` 目录）
   - 创建 PDF 报告（保存到 `pdf/` 目录）

### 性能基准测试

运行性能基准测试以评估不同批处理大小下的模型性能：

```
python benchmark_yolo.py
```

这将生成有关吞吐量、延迟和内存使用的详细报告和图表。

### 生成合并报告

将评估结果和基准测试合并为一个综合报告：

```
python merge_and_generate_pdf.py
```

这将创建一个包含所有结果的完整 PDF 文档。

## 项目脚本说明

- `eval_yolo_coco.py`：主要评估脚本，计算 AP、AUC 等指标并生成图表
- `benchmark_yolo.py`：性能基准测试脚本，测试不同批处理大小下的性能
- `merge_html_files.py`：合并 HTML 报告文件
- `merge_and_generate_pdf.py`：合并 HTML 并生成最终 PDF 报告
- `generate_pdf_report.py`：使用 WeasyPrint 生成增强的 PDF 报告
- `find_threshold.py`：查找特定召回率或 TPR 值对应的最佳阈值
- `organize_project.py`：整理项目文件到对应目录
- `cleanup.py`：清理无用文件并维护项目结构

## 配置

您可以在 `eval_yolo_coco.py` 中修改以下参数：

- `MODEL_PATHS`：YOLO 模型文件路径
- `MODEL_NAMES`：用于显示的模型名称
- `NUM_SAMPLES`：用于评估的图像数量
- `IOU_THRESHOLD`：真阳性检测的 IoU 阈值

在 `benchmark_yolo.py` 中，您可以设置：

- `batch_sizes`：要测试的批处理大小列表
- `num_warmup`：预热推理次数
- `num_infer`：正式推理次数

## 输出文件

所有输出文件按类型整理在专用目录中：

### images/ 目录
- `pr_curve_comparison.png`：精确率-召回率曲线
- `roc_curve_comparison.png`：ROC 曲线
- `regression_bias.png`：回归偏差分布
- `confidence_distribution.png`：置信度分数分布
- `throughput_fps_comparison.png`：吞吐量比较图
- `latency_per_image_ms_comparison.png`：单图延迟比较图

### pdf/ 目录
- `evaluation_results.pdf`：原始 PDF 报告
- `weasy_evaluation_results.pdf`：使用 WeasyPrint 的增强 PDF 报告
- `merged_evaluation_results.pdf`：合并了评估和基准测试的完整报告

### html/ 目录
- `report_template.html`：报告模板
- `benchmark_report.html`：基准测试 HTML 报告
- `merged_evaluation_results.html`：合并后的评估报告

## 自定义 PDF 报告

要自定义基于 WeasyPrint 的 PDF 报告：

1. 编辑 `html/report_template.html` 更改结构和内容
2. 在 `generate_pdf_report.py` 中修改 CSS 样式（在 `get_css()` 函数内）

## 许可

[在此指定您的许可证]

## 致谢

- [WeasyPrint](https://weasyprint.org/) 用于从 HTML/CSS 生成 PDF
- [Ultralytics](https://github.com/ultralytics/ultralytics) 提供 YOLO 实现 