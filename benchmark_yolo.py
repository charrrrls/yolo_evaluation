#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO模型性能基准测试脚本
测试不同YOLO模型在各个批次大小下的性能表现
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import platform
from ultralytics import YOLO
from datetime import datetime
import argparse
import cpuinfo
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # Mac优先使用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
def get_system_info():
    """
    获取系统硬件和软件信息
    """
    # 获取CPU信息
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_brand = cpu_info.get('brand_raw', platform.processor())
        cpu_arch = cpu_info.get('arch', platform.machine())
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
    except:
        cpu_brand = platform.processor() or "未知"
        cpu_arch = platform.machine() or "未知"
        cpu_cores = psutil.cpu_count(logical=False) or "未知"
        cpu_threads = psutil.cpu_count(logical=True) or "未知"
    
    # 获取内存信息
    mem = psutil.virtual_memory()
    total_memory = f"{mem.total / (1024**3):.2f} GB"
    
    # 获取操作系统信息
    os_info = f"{platform.system()} {platform.release()}"
    
    # 获取Python和PyTorch版本
    python_version = platform.python_version()
    torch_version = torch.__version__
    
    # 获取GPU/MPS信息
    if torch.cuda.is_available():
        device_type = "CUDA (GPU)"
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        device_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_type = "MPS (Apple Silicon)"
        device_name = "Apple Silicon"
        device_count = 1
        # MPS没有直接获取显存的方法
        device_mem = "共享系统内存"
    else:
        device_type = "CPU"
        device_name = cpu_brand
        device_count = 1
        device_mem = total_memory
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "os": os_info,
            "hostname": platform.node()
        },
        "cpu": {
            "brand": cpu_brand,
            "architecture": cpu_arch,
            "physical_cores": cpu_cores,
            "logical_cores": cpu_threads
        },
        "memory": {
            "total": total_memory,
            "available": f"{mem.available / (1024**3):.2f} GB",
            "used_percent": f"{mem.percent}%"
        },
        "compute": {
            "type": device_type,
            "name": device_name,
            "count": device_count,
            "memory": device_mem
        },
        "software": {
            "python": python_version,
            "pytorch": torch_version,
            "cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else "不可用"
        }
    }

def run_benchmark(model_path, batch_sizes=[1, 10, 50, 100], num_warmup=10, num_infer=50, img_size=640):
    """
    对模型进行性能基准测试
    
    参数:
        model_path: 模型路径
        batch_sizes: 要测试的batch size列表
        num_warmup: 预热推理次数
        num_infer: 正式推理次数
        img_size: 输入图像大小
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    model_name = os.path.basename(model_path).split('.')[0]
    model = YOLO(model_path)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                          'cpu')
    print(f"使用设备: {device}")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n测试批处理大小: {batch_size}")
        
        # 创建随机输入数据
        dummy_input = torch.rand(batch_size, 3, img_size, img_size)
        if str(device) == 'mps':
            dummy_input = dummy_input.to('mps')
        elif str(device) == 'cuda':
            dummy_input = dummy_input.to('cuda')
        
        # 预热模型
        print("预热中...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input, verbose=False)
        
        # 测量推理时间
        print(f"执行 {num_infer} 次推理...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 记录开始时间和初始资源使用情况
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        # 执行推理
        for _ in range(num_infer):
            with torch.no_grad():
                _ = model(dummy_input, verbose=False)
        
        # 确保所有GPU操作完成
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 计算总时间和内存使用
        total_time = time.time() - start_time
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        memory_used = end_memory - start_memory
        
        # 计算性能指标
        latency_per_batch = total_time / num_infer * 1000  # 每个批次的延迟(ms)
        latency_per_image = latency_per_batch / batch_size  # 每张图像的延迟(ms)
        throughput = (batch_size * num_infer) / total_time  # 每秒处理图像数量
        
        print(f"批次大小: {batch_size}, 总时间: {total_time:.2f}秒")
        print(f"批次延迟: {latency_per_batch:.2f}毫秒, 单图延迟: {latency_per_image:.2f}毫秒")
        print(f"吞吐量: {throughput:.2f}图像/秒")
        print(f"内存增加: {memory_used:.2f}MB")
        
        # 获取当前CPU和内存使用情况
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # 收集结果
        results.append({
            "batch_size": batch_size,
            "latency_per_batch_ms": latency_per_batch,
            "latency_per_image_ms": latency_per_image,
            "throughput_fps": throughput,
            "memory_increase_mb": memory_used,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        })
    
    return model_name, results

def plot_benchmark_results(all_results, output_dir="./benchmark_results"):
    """
    绘制基准测试结果图表
    """
    # 确保输出目录存在
    images_dir = "./images"
    csv_dir = "./csv"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    # 为每个模型创建DataFrame
    dataframes = {}
    for model_name, results in all_results.items():
        dataframes[model_name] = pd.DataFrame(results)
    
    # 合并所有数据用于保存到CSV
    all_data = []
    for model_name, df in dataframes.items():
        df['model'] = model_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data)
    csv_path = os.path.join(csv_dir, "benchmark_results.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")
    
    # 绘制图表
    metrics = [
        ("throughput_fps", "吞吐量 (图像/秒)"),
        ("latency_per_image_ms", "单图延迟 (毫秒)"),
        ("latency_per_batch_ms", "批次延迟 (毫秒)"),
        ("memory_increase_mb", "内存增加 (MB)")
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(10, 6))
        
        for model_name, df in dataframes.items():
            plt.plot(df['batch_size'], df[metric], marker='o', label=model_name)
        
        plt.title(f"{title} vs. 批次大小")
        plt.xlabel("批次大小")
        plt.ylabel(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 对于延迟图表，使用对数刻度更容易看清差异
        if "latency" in metric:
            plt.yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(images_dir, f"{metric}_comparison.png")
        plt.savefig(plot_path, dpi=300)
        print(f"图表已保存到: {plot_path}")
    
    # 创建吞吐量与批次大小关系的柱状图
    plt.figure(figsize=(12, 7))
    
    # 提取批次大小和不同模型的吞吐量
    batch_sizes = dataframes[list(dataframes.keys())[0]]['batch_size']
    width = 0.2  # 柱子宽度
    
    # 为每个模型创建柱状图
    positions = np.arange(len(batch_sizes))
    for i, (model_name, df) in enumerate(dataframes.items()):
        plt.bar(positions + i*width, df['throughput_fps'], 
                width=width, label=model_name)
    
    plt.title("不同批次大小下的模型吞吐量比较")
    plt.xlabel("批次大小")
    plt.ylabel("吞吐量 (图像/秒)")
    plt.xticks(positions + width, batch_sizes)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    bar_chart_path = os.path.join(images_dir, "throughput_bar_chart.png")
    plt.savefig(bar_chart_path, dpi=300)
    print(f"柱状图已保存到: {bar_chart_path}")

def generate_benchmark_report(system_info, all_results, output_dir="./benchmark_results"):
    """
    生成基准测试HTML报告
    """
    # 确保输出目录存在
    html_dir = "./html"
    json_dir = "./json"
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    # 创建结果摘要
    model_summaries = {}
    for model_name, results in all_results.items():
        # 获取批次大小为1和最大批次的结果
        min_batch = results[0]  # 批次大小1
        max_batch = results[-1]  # 最大批次大小
        
        model_summaries[model_name] = {
            "min_batch": min_batch,
            "max_batch": max_batch,
            "max_throughput": max(results, key=lambda x: x["throughput_fps"]),
            "min_latency": min(results, key=lambda x: x["latency_per_image_ms"])
        }
    
    # 构建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLO模型性能基准测试报告</title>
        <style>
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c5282;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid #e2e8f0;
            }}
            .system-info {{
                background-color: #f8fafc;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            .system-info h2 {{
                margin-top: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            th {{
                background-color: #edf2f7;
                font-weight: 500;
                color: #2d3748;
            }}
            tr:nth-child(even) {{
                background-color: #f7fafc;
            }}
            .charts-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }}
            .chart {{
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
            }}
            .chart img {{
                width: 100%;
                height: auto;
            }}
            .model-summary {{
                margin-bottom: 40px;
            }}
            .highlights {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin: 20px 0;
            }}
            .highlight-card {{
                background-color: #ebf8ff;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
                text-align: center;
            }}
            .highlight-value {{
                font-size: 24px;
                font-weight: 500;
                color: #2b6cb0;
                margin: 10px 0;
            }}
            .highlight-label {{
                font-size: 14px;
                color: #4a5568;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                color: #718096;
                font-size: 14px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
            }}
            @media (max-width: 768px) {{
                .charts-container {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>YOLO模型性能基准测试报告</h1>
            <p>生成时间: {system_info["timestamp"]}</p>
        </div>
        
        <div class="system-info">
            <h2>系统信息</h2>
            <table>
                <tr>
                    <th>操作系统</th>
                    <td>{system_info["system"]["os"]} ({system_info["system"]["hostname"]})</td>
                </tr>
                <tr>
                    <th>CPU</th>
                    <td>{system_info["cpu"]["brand"]} ({system_info["cpu"]["physical_cores"]} 物理核心, {system_info["cpu"]["logical_cores"]} 逻辑核心)</td>
                </tr>
                <tr>
                    <th>内存</th>
                    <td>总计: {system_info["memory"]["total"]}, 可用: {system_info["memory"]["available"]} (使用率: {system_info["memory"]["used_percent"]})</td>
                </tr>
                <tr>
                    <th>计算设备</th>
                    <td>{system_info["compute"]["type"]} - {system_info["compute"]["name"]} (显存: {system_info["compute"]["memory"]})</td>
                </tr>
                <tr>
                    <th>软件环境</th>
                    <td>Python {system_info["software"]["python"]}, PyTorch {system_info["software"]["pytorch"]}, CUDA {system_info["software"]["cuda"]}</td>
                </tr>
            </table>
        </div>
        
        <h2>测试结果概览</h2>
        <p>本测试评估了不同YOLO模型在各批次大小下的性能表现，包括推理延迟、吞吐量和内存使用情况。</p>
        
        <div class="charts-container">
            <div class="chart">
                <h3>吞吐量比较</h3>
                <img src="../images/throughput_fps_comparison.png" alt="吞吐量比较">
            </div>
            <div class="chart">
                <h3>单图延迟比较</h3>
                <img src="../images/latency_per_image_ms_comparison.png" alt="单图延迟比较">
            </div>
            <div class="chart">
                <h3>批次延迟比较</h3>
                <img src="../images/latency_per_batch_ms_comparison.png" alt="批次延迟比较">
            </div>
            <div class="chart">
                <h3>内存使用比较</h3>
                <img src="../images/memory_increase_mb_comparison.png" alt="内存使用比较">
            </div>
        </div>
        
        <div class="chart">
            <h3>吞吐量柱状图比较</h3>
            <img src="../images/throughput_bar_chart.png" alt="吞吐量柱状图比较">
        </div>
    """
    
    # 为每个模型添加详细结果
    for model_name, results in all_results.items():
        summary = model_summaries[model_name]
        html_content += f"""
        <div class="model-summary">
            <h2>{model_name} 模型性能</h2>
            
            <div class="highlights">
                <div class="highlight-card">
                    <div class="highlight-value">{summary["max_throughput"]["throughput_fps"]:.1f}</div>
                    <div class="highlight-label">最大吞吐量 (图像/秒)</div>
                    <div class="highlight-desc">批次大小: {summary["max_throughput"]["batch_size"]}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-value">{summary["min_latency"]["latency_per_image_ms"]:.2f}</div>
                    <div class="highlight-label">最低单图延迟 (毫秒)</div>
                    <div class="highlight-desc">批次大小: {summary["min_latency"]["batch_size"]}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-value">{summary["min_batch"]["latency_per_batch_ms"]:.2f}</div>
                    <div class="highlight-label">单批次延迟 (毫秒)</div>
                    <div class="highlight-desc">批次大小: {summary["min_batch"]["batch_size"]}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-value">{summary["max_batch"]["memory_increase_mb"]:.1f}</div>
                    <div class="highlight-label">最大内存增加 (MB)</div>
                    <div class="highlight-desc">批次大小: {summary["max_batch"]["batch_size"]}</div>
                </div>
            </div>
            
            <h3>详细性能数据</h3>
            <table>
                <tr>
                    <th>批次大小</th>
                    <th>批次延迟 (毫秒)</th>
                    <th>单图延迟 (毫秒)</th>
                    <th>吞吐量 (图像/秒)</th>
                    <th>内存增加 (MB)</th>
                    <th>CPU使用率 (%)</th>
                    <th>内存使用率 (%)</th>
                </tr>
        """
        
        for result in results:
            html_content += f"""
                <tr>
                    <td>{result["batch_size"]}</td>
                    <td>{result["latency_per_batch_ms"]:.2f}</td>
                    <td>{result["latency_per_image_ms"]:.2f}</td>
                    <td>{result["throughput_fps"]:.2f}</td>
                    <td>{result["memory_increase_mb"]:.2f}</td>
                    <td>{result["cpu_percent"]:.1f}</td>
                    <td>{result["memory_percent"]:.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # 添加结论和建议
    html_content += """
        <h2>结论与建议</h2>
        <div class="system-info">
            <p>根据基准测试结果，可以得出以下结论和建议：</p>
            <ul>
                <li><strong>延迟敏感场景</strong>：对于需要低延迟的实时应用，应使用较小的批次大小（通常为1）。YOLOv8n-pose模型通常提供最低的单图延迟。</li>
                <li><strong>吞吐量优先场景</strong>：对于需要高吞吐量的离线处理，应使用较大的批次大小。随着批次大小增加，吞吐量通常会提高，但会达到硬件极限。</li>
                <li><strong>内存受限场景</strong>：对于内存受限的设备，应考虑使用较小的模型（如YOLOv8n-pose）并限制批次大小。</li>
                <li><strong>资源均衡</strong>：YOLOv8s-pose模型在性能和资源消耗之间提供良好的平衡，适合中等配置设备。</li>
                <li><strong>高精度需求</strong>：对于需要高精度的应用，YOLOv8m-pose模型提供最佳性能，但需要更多计算资源。</li>
            </ul>
        </div>
        
        <footer>
            <p>YOLO模型性能基准测试报告 | 生成于 {system_info["timestamp"]}</p>
        </footer>
    </body>
    </html>
    """
    
    # 保存HTML报告
    report_path = os.path.join(html_dir, "benchmark_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML报告已保存到: {report_path}")
    
    # 保存系统信息和结果为JSON
    json_path = os.path.join(json_dir, "benchmark_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "system_info": system_info,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"基准测试数据已保存到: {json_path}")
    
    return report_path

def update_evaluation_report(benchmark_report_path, output_path="./benchmark_evaluation_results.pdf"):
    """
    将基准测试报告添加到评估报告中
    """
    # 这里可以使用现有的generate_pdf_report.py逻辑，将基准测试结果添加到报告中
    # 但由于这需要对原有代码进行修改，这里只返回基准测试报告的路径
    return benchmark_report_path

def main():
    print("="*50)
    print("YOLO模型性能基准测试")
    print("="*50)
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型性能基准测试')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4],
                      help='要测试的批次大小列表，例如：--batch-sizes 1 2 4 8 (默认: [1, 2, 4])')
    parser.add_argument('--warmup', type=int, default=3,
                      help='预热次数 (默认: 3)')
    parser.add_argument('--infer', type=int, default=10,
                      help='推理次数 (默认: 10)')
    parser.add_argument('--models', type=str, nargs='+', default=["./models/yolov8n-pose.pt"],
                      help='要测试的模型列表，例如：--models ./models/yolov8n-pose.pt ./models/yolov8m-pose.pt')
    
    args = parser.parse_args()
    
    # 获取系统信息
    system_info = get_system_info()
    print("\n系统信息:")
    print(f"CPU: {system_info['cpu']['brand']}")
    print(f"内存: {system_info['memory']['total']}")
    print(f"计算设备: {system_info['compute']['type']} - {system_info['compute']['name']}")
    print(f"PyTorch: {system_info['software']['pytorch']}")
    
    # 定义要测试的模型
    models = args.models
    
    # 定义要测试的批次大小
    batch_sizes = args.batch_sizes
    
    # 运行基准测试
    all_results = {}
    for model_path in models:
        if os.path.exists(model_path):
            model_name, results = run_benchmark(
                model_path, 
                batch_sizes=batch_sizes,
                num_warmup=args.warmup,
                num_infer=args.infer
            )
            all_results[model_name] = results
        else:
            print(f"警告: 模型文件不存在 - {model_path}")
    
    # 绘制结果图表
    plot_benchmark_results(all_results)
    
    # 生成HTML报告
    benchmark_report = generate_benchmark_report(system_info, all_results)
    
    print("\n基准测试完成!")
    print(f"HTML报告已生成: {benchmark_report}")
    
    return benchmark_report

if __name__ == "__main__":
    main() 