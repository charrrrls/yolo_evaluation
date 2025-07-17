#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTML合并与PDF生成工具
该脚本将完成以下任务：
1. 将benchmark_report.html的内容添加到weasy_evaluation_results.html文件末尾
2. 将合并后的HTML转换为PDF
"""

import os
import sys
import re
import shutil
import traceback
from bs4 import BeautifulSoup
import copy

def extract_content_without_duplicate_titles(html_content):
    """
    从HTML内容中提取内容，并去除重复标题
    
    参数:
        html_content: HTML内容
    返回:
        处理后的HTML内容
    """
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除页眉部分 (report-header类)
    header = soup.find('div', class_='report-header')
    if header:
        header.decompose()
    
    # 移除页脚部分
    footer = soup.find('footer')
    if footer:
        footer.decompose()
    
    # 返回处理后的HTML内容
    return str(soup.body) if soup.body else ""

def fix_image_paths(soup):
    """
    修复HTML中的图片路径，确保相对路径能正确指向images目录
    
    参数:
        soup: BeautifulSoup对象
    """
    # 查找所有img标签
    for img in soup.find_all('img'):
        # 如果src属性存在且是相对路径
        if img.has_attr('src') and img['src'].startswith('../images/'):
            # 将相对路径转换为绝对路径
            img['src'] = img['src'].replace('../images/', '/Users/leion/Charles/work/annotations/yolo_evaluation/images/')
    
    return soup

def merge_html_files_direct(main_file, benchmark_file, output_file):
    """
    直接合并HTML文件，确保不会有重复内容
    
    参数:
        main_file: 主HTML文件路径
        benchmark_file: 基准测试HTML文件路径
        output_file: 输出合并HTML文件路径
    """
    print(f"正在合并 {main_file} 和 {benchmark_file}...")
    
    try:
        # 读取主HTML文件
        with open(main_file, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # 检查主文件是否已包含基准测试内容
        if "YOLO模型性能基准测试报告" in main_content or "模型性能基准测试" in main_content:
            print("检测到主文件已包含基准测试报告内容，跳过合并...")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(main_content)
            print(f"内容已复制到输出文件：{output_file}")
            return True
        
        # 读取基准测试HTML文件
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_content = f.read()
        
        # 解析主HTML和基准测试HTML内容
        main_soup = BeautifulSoup(main_content, 'html.parser')
        benchmark_soup = BeautifulSoup(benchmark_content, 'html.parser')
        
        if not benchmark_soup.body:
            print("错误：基准测试HTML文件没有<body>标签")
            return False
        
        # 提取并处理benchmark_report的CSS样式
        benchmark_style = benchmark_soup.find('style')
        if benchmark_style:
            # 从基准测试HTML中提取样式
            benchmark_css = benchmark_style.string
            
            # 查找主HTML的style标签
            main_style = main_soup.find('style')
            if main_style:
                # 将基准测试的CSS样式添加到主HTML的style标签中
                combined_css = main_style.string + "\n\n/* 基准测试报告样式 */\n" + benchmark_css
                main_style.string = combined_css
        
        # 提取<body>内的有效内容
        report_header = benchmark_soup.find('div', class_='report-header')
        if report_header:
            # 创建新的标题区域，不包含原始的h1标题
            new_header = BeautifulSoup('<div class="benchmark-header"><h1>模型性能基准测试</h1></div>', 'html.parser')
            report_header.replace_with(new_header)
        
        # 移除页脚
        footer = benchmark_soup.find('footer')
        if footer:
            footer.decompose()
        
        # 修复图片路径
        benchmark_soup = fix_image_paths(benchmark_soup)
        
        # 创建一个新的div来包含基准测试内容
        benchmark_div = main_soup.new_tag('div')
        benchmark_div['style'] = 'page-break-before: always;'
        benchmark_div['class'] = 'benchmark-report-section'
        
        # 将基准测试body内容添加到新div中（不包括body标签本身）
        for child in list(benchmark_soup.body.children):
            # 深拷贝防止原对象被修改
            benchmark_div.append(copy.copy(child))
        
        # 将新div添加到主HTML的body末尾
        main_soup.body.append(benchmark_div)
        
        # 写入合并后的HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(main_soup))
        
        print(f"合并完成！输出文件：{output_file}")
        return True
    
    except Exception as e:
        print(f"合并HTML文件时出错：{str(e)}")
        traceback.print_exc()
        return False

def setup_environment():
    """设置WeasyPrint所需的环境变量"""
    os.environ["LD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    
    print("已设置环境变量:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH')}")
    print(f"DYLD_FALLBACK_LIBRARY_PATH: {os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')}")

def generate_pdf(html_file, output_pdf):
    """
    将HTML文件转换为PDF
    
    参数:
        html_file: 输入HTML文件路径
        output_pdf: 输出PDF文件路径
    """
    try:
        print(f"正在将 {html_file} 转换为 PDF...")
        
        # 确保输入文件存在
        if not os.path.exists(html_file):
            print(f"错误：HTML文件不存在 - {html_file}")
            return False
        
        # 导入WeasyPrint (在设置环境变量后导入)
        from weasyprint import HTML, CSS
        
        # 创建额外的CSS样式，确保在PDF中显示正确
        additional_css = CSS(string="""
            @page {
                size: A4;
                margin: 1cm;
            }
            body {
                font-family: Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.5;
            }
            .benchmark-report-section {
                page-break-before: always;
            }
            .benchmark-header h1 {
                text-align: center;
                margin-bottom: 20px;
            }
            /* 基准测试报告样式修复 */
            .charts-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }
            .chart {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
            }
            .chart img {
                width: 100%;
                height: auto;
                max-width: 100%;
            }
            @media (max-width: 768px) {
                .charts-container {
                    grid-template-columns: 1fr;
                }
            }
            .system-info {
                background-color: #f8fafc;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
            }
            .highlights {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin: 20px 0;
            }
            .highlight-card {
                background-color: #ebf8ff;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
                text-align: center;
            }
        """)
        
        # 转换HTML为PDF
        HTML(html_file).write_pdf(output_pdf, stylesheets=[additional_css])
        
        print(f"PDF生成成功！输出文件：{output_pdf}")
        return True
        
    except Exception as e:
        print(f"生成PDF时出错：{str(e)}")
        traceback.print_exc()
        return False

def clean_merged_files(merged_html):
    """
    清理已存在的合并文件，防止内容重复
    
    参数:
        merged_html: 合并HTML文件路径
    """
    if os.path.exists(merged_html):
        try:
            os.remove(merged_html)
            print(f"已删除旧的合并文件: {merged_html}")
        except Exception as e:
            print(f"删除旧文件时出错: {str(e)}")

def main():
    # 文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_dir = os.path.join(current_dir, 'html')
    pdf_dir = os.path.join(current_dir, 'pdf')
    
    # 确保pdf目录存在
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    main_file = os.path.join(html_dir, 'weasy_evaluation_results.html')
    append_file = os.path.join(html_dir, 'benchmark_report.html')
    merged_html = os.path.join(html_dir, 'merged_evaluation_results.html')
    output_pdf = os.path.join(pdf_dir, 'merged_evaluation_results.pdf')
    
    # 清理已存在的合并文件
    clean_merged_files(merged_html)
    # 同时清理根目录下的文件，确保从头开始
    root_merged_html = os.path.join(current_dir, 'merged_evaluation_results.html')
    root_output_pdf = os.path.join(current_dir, 'merged_evaluation_results.pdf')
    clean_merged_files(root_merged_html)
    clean_merged_files(root_output_pdf)
    
    # 检查文件是否存在
    if not os.path.exists(main_file):
        # 检查是否在根目录
        root_main_file = os.path.join(current_dir, 'weasy_evaluation_results.html')
        if os.path.exists(root_main_file):
            print(f"找到主文件在根目录: {root_main_file}")
            main_file = root_main_file
        else:
            print(f"错误：主文件不存在 - {main_file}")
            sys.exit(1)
    
    if not os.path.exists(append_file):
        print(f"错误：追加文件不存在 - {append_file}")
        sys.exit(1)
    
    # 步骤1：合并HTML文件
    print("步骤1：合并HTML文件...")
    success = merge_html_files_direct(main_file, append_file, merged_html)
    if not success:
        print("HTML文件合并失败，程序终止。")
        sys.exit(1)
    
    # 步骤2：设置环境变量
    print("\n步骤2：设置环境变量...")
    setup_environment()
    
    # 步骤3：生成PDF
    print("\n步骤3：生成PDF文件...")
    success = generate_pdf(merged_html, output_pdf)
    if not success:
        print("PDF生成失败，程序终止。")
        sys.exit(1)
    
    print("\n完成！已成功合并HTML文件并生成PDF。")
    print(f"- 合并的HTML文件：{merged_html}")
    print(f"- 生成的PDF文件：{output_pdf}")
    
    # 将合并后的HTML文件复制到根目录（可选，保持与旧版本兼容）
    try:
        shutil.copy2(merged_html, root_merged_html)
        shutil.copy2(output_pdf, root_output_pdf)
        print("\n为了向后兼容，文件也已复制到根目录：")
        print(f"- {root_merged_html}")
        print(f"- {root_output_pdf}")
    except Exception as e:
        print(f"\n向根目录复制文件时出错：{str(e)}")

if __name__ == "__main__":
    main() 