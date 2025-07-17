#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import numpy as np

def find_closest_threshold(csv_file, target_value, column_name):
    """
    在CSV文件中找到最接近目标值的行，并返回对应的阈值
    
    Args:
        csv_file: CSV文件路径
        target_value: 目标值（召回率或TPR）
        column_name: 列名（'recall'或'tpr'）
    
    Returns:
        最接近的行数据
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 确保列名存在
    if column_name not in df.columns:
        raise ValueError(f"列名 '{column_name}' 不存在于CSV文件中")
    
    # 计算与目标值的差距
    df['distance'] = abs(df[column_name] - target_value)
    
    # 找到最接近的行
    closest_row = df.loc[df['distance'].idxmin()]
    
    # 删除临时列
    df.drop('distance', axis=1, inplace=True)
    
    return closest_row

def analyze_csv_files():
    """分析当前目录下的所有CSV文件，输出统计信息"""
    pr_files = [f for f in os.listdir('.') if f.endswith('_pr_curve_data.csv')]
    roc_files = [f for f in os.listdir('.') if f.endswith('_roc_curve_data.csv')]
    
    print("\n===== CSV文件统计信息 =====")
    
    for f in pr_files:
        df = pd.read_csv(f)
        model_name = f.split('_')[0]
        print(f"\n{model_name} PR曲线数据:")
        print(f"  - 数据点数量: {len(df)}")
        print(f"  - 阈值范围: {df['threshold'].min():.4f} 到 {df['threshold'].max():.4f}")
        print(f"  - 精确率范围: {df['precision'].min():.4f} 到 {df['precision'].max():.4f}")
        print(f"  - 召回率范围: {df['recall'].min():.4f} 到 {df['recall'].max():.4f}")
    
    for f in roc_files:
        df = pd.read_csv(f)
        # 过滤掉inf值
        valid_thresholds = [x for x in df['threshold'] if x != float('inf')]
        model_name = f.split('_')[0]
        print(f"\n{model_name} ROC曲线数据:")
        print(f"  - 数据点数量: {len(df)}")
        if valid_thresholds:
            print(f"  - 阈值范围: {min(valid_thresholds):.4f} 到 {max(valid_thresholds):.4f}")
        else:
            print("  - 阈值范围: 无有效阈值")
        print(f"  - FPR范围: {df['fpr'].min():.4f} 到 {df['fpr'].max():.4f}")
        print(f"  - TPR范围: {df['tpr'].min():.4f} 到 {df['tpr'].max():.4f}")

def compare_thresholds(target_value, column_name):
    """比较所有模型在特定目标值下的阈值"""
    files = []
    if column_name == 'recall':
        files = [f for f in os.listdir('.') if f.endswith('_pr_curve_data.csv')]
    else:  # tpr
        files = [f for f in os.listdir('.') if f.endswith('_roc_curve_data.csv')]
    
    if not files:
        print(f"没有找到包含 {column_name} 列的CSV文件")
        return
    
    print(f"\n===== 所有模型在 {column_name}={target_value} 时的阈值比较 =====")
    print(f"{'模型':<10} | {'阈值':<10} | {'精确率/FPR':<10} | {column_name:<10}")
    print("-" * 50)
    
    for f in files:
        model_name = f.split('_')[0]
        try:
            result = find_closest_threshold(f, target_value, column_name)
            
            if column_name == 'recall':
                print(f"{model_name:<10} | {result['threshold']:.4f} | {result['precision']:.4f} | {result['recall']:.4f}")
            else:  # tpr
                print(f"{model_name:<10} | {result['threshold']:.4f} | {result['fpr']:.4f} | {result['tpr']:.4f}")
        except Exception as e:
            print(f"{model_name:<10} | 错误: {str(e)}")
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='查找特定召回率或TPR值对应的阈值')
    parser.add_argument('--file', type=str, help='CSV文件路径')
    parser.add_argument('--target', type=float, help='目标值（召回率或TPR）')
    parser.add_argument('--column', type=str, choices=['recall', 'tpr'], help='列名（recall或tpr）')
    parser.add_argument('--analyze', action='store_true', help='分析所有CSV文件')
    parser.add_argument('--compare', action='store_true', help='比较所有模型在特定目标值下的阈值')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_csv_files()
        return
    
    if args.compare:
        if not args.target or not args.column:
            print("错误: 使用--compare时必须指定--target和--column参数")
            parser.print_help()
            return
        compare_thresholds(args.target, args.column)
        return
    
    if not args.file or not args.target or not args.column:
        parser.print_help()
        return
    
    try:
        result = find_closest_threshold(args.file, args.target, args.column)
        
        print(f"\n在 {args.file} 中找到最接近 {args.column}={args.target} 的行:")
        if args.column == 'recall':
            print(f"阈值: {result['threshold']:.4f}, 精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}")
        else:  # tpr
            print(f"阈值: {result['threshold']:.4f}, FPR: {result['fpr']:.4f}, TPR: {result['tpr']:.4f}")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 