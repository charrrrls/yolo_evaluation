#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Report Generator for YOLO Evaluation Results
This script generates a PDF report from YOLO evaluation results using WeasyPrint
"""

import os
import json
import base64
import datetime
from jinja2 import Environment, FileSystemLoader

def setup_environment():
    """设置WeasyPrint所需的环境变量"""
    os.environ["LD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    
    print("已设置环境变量:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH')}")
    print(f"DYLD_FALLBACK_LIBRARY_PATH: {os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')}")

setup_environment()
from weasyprint import HTML, CSS
def generate_pdf_report(all_results, output_path='yolo_evaluation_report.pdf', save_html=False):
    """
    Generate a PDF report from YOLO evaluation results
    
    Parameters:
        all_results: List of dictionaries containing model results
        output_path: Path to save the PDF report
        save_html: If True, also save the HTML version of the report
    """
    # Get current directory to locate templates
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(current_dir))
    template = env.get_template('html/report_template.html')
    
    # 读取关键点表格文件
    key_points_html = ""
    try:
        html_dir = os.path.join(current_dir, 'html')
        key_points_file = os.path.join(html_dir, 'key_points_tables.html')
        if os.path.exists(key_points_file):
            with open(key_points_file, 'r', encoding='utf-8') as f:
                key_points_html = f.read()
        else:
            print(f"警告：关键点表格文件不存在: {key_points_file}")
    except Exception as e:
        print(f"读取关键点表格文件时出错: {str(e)}")
        key_points_html = "<p>无法加载关键点表格数据</p>"
    
    # 提取PR和ROC表格，并添加内联样式以确保在PDF中正确显示
    pr_html_table = ""
    roc_html_table = ""
    
    # 基础表格样式 - 直接内联到表格标签中
    table_style = """
    style="width:100%; border-collapse:collapse; margin-top:0.5cm; font-size:9pt;"
    """
    
    th_style = """
    style="border:1px solid #cbd5e0; padding:0.3cm; text-align:center; background-color:#edf2f7; font-weight:500; color:#2d3748;"
    """
    
    td_style = """
    style="border:1px solid #cbd5e0; padding:0.3cm; text-align:center;"
    """
    
    tr_even_style = """
    style="background-color:#f7fafc;"
    """
    
    # 解析HTML内容并提取表格
    if key_points_html:
        # 查找所有表格的开始和结束位置
        table_positions = []
        start_pos = 0
        while True:
            table_start = key_points_html.find('<table', start_pos)
            if table_start == -1:
                break
            table_end = key_points_html.find('</table>', table_start) + 8
            if table_end == 7:  # 没有找到结束标签
                break
            table_positions.append((table_start, table_end))
            start_pos = table_end
        
        print(f"找到 {len(table_positions)} 个表格")
        
        # 提取第一个表格(PR表格)
        if len(table_positions) >= 1:
            start_idx, end_idx = table_positions[0]
            original_table = key_points_html[start_idx:end_idx]
            
            # 替换表格标签，添加内联样式
            enhanced_table = original_table.replace('<table>', f'<table {table_style}>')
            enhanced_table = enhanced_table.replace('<table', f'<table {table_style[:-1]}', 1)  # 处理可能有其他属性的情况
            
            # 替换th标签
            enhanced_table = enhanced_table.replace('<th>', f'<th {th_style}>')
            
            # 替换td标签
            enhanced_table = enhanced_table.replace('<td>', f'<td {td_style}>')
            
            # 为偶数行添加背景色
            rows = enhanced_table.split('<tr>')
            styled_rows = []
            for i, row in enumerate(rows):
                if i > 0 and i % 2 == 0 and 'thead' not in row:  # 跳过表头，只处理tbody中的偶数行
                    styled_rows.append(f'<tr {tr_even_style}>{row}')
                else:
                    styled_rows.append(f'<tr>{row}')
            
            pr_html_table = ''.join(styled_rows).replace('<tr><tr', '<tr')  # 修复可能的标签嵌套
        
        # 提取第二个表格(ROC表格)
        if len(table_positions) >= 2:
            start_idx, end_idx = table_positions[1]
            original_table = key_points_html[start_idx:end_idx]
            
            # 替换表格标签，添加内联样式
            enhanced_table = original_table.replace('<table>', f'<table {table_style}>')
            enhanced_table = enhanced_table.replace('<table', f'<table {table_style[:-1]}', 1)  # 处理可能有其他属性的情况
            
            # 替换th标签
            enhanced_table = enhanced_table.replace('<th>', f'<th {th_style}>')
            
            # 替换td标签
            enhanced_table = enhanced_table.replace('<td>', f'<td {td_style}>')
            
            # 为偶数行添加背景色
            rows = enhanced_table.split('<tr>')
            styled_rows = []
            for i, row in enumerate(rows):
                if i > 0 and i % 2 == 0 and 'thead' not in row:  # 跳过表头，只处理tbody中的偶数行
                    styled_rows.append(f'<tr {tr_even_style}>{row}')
                else:
                    styled_rows.append(f'<tr>{row}')
            
            roc_html_table = ''.join(styled_rows).replace('<tr><tr', '<tr')  # 修复可能的标签嵌套
    
    # Prepare data for the template
    best_ap_model = max(all_results, key=lambda x: x['ap'])['name']
    best_auc_model = max(all_results, key=lambda x: x.get('auc', float('-inf')))['name']
    
    # Encode images as base64 strings
    images = {}
    image_files = [
        'pr_curve_comparison.png',
        'roc_curve_comparison.png',
        'regression_bias.png',
        'confidence_distribution.png'
    ]
    
    # 修正图片路径，使用正确的images目录
    images_dir = os.path.join(current_dir, 'images')
    
    for img_file in image_files:
        try:
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    images[img_file] = f'data:image/png;base64,{img_base64}'
            else:
                print(f"警告: 图片文件不存在: {img_path}")
                # 使用相对路径引用图片，而不是使用空字符串
                images[img_file] = f'../images/{img_file}'
        except Exception as e:
            print(f"处理图片文件时出错: {img_file}, 错误: {str(e)}")
            # 使用相对路径引用图片，而不是使用空字符串
            images[img_file] = f'../images/{img_file}'
    
    # Assign colors to models based on their name
    for model in all_results:
        if 'n' in model['name'].lower():
            model['color'] = 'yolov8-n'
        elif 's' in model['name'].lower():
            model['color'] = 'yolov8-s'
        elif 'm' in model['name'].lower():
            model['color'] = 'yolov8-m'
        else:
            model['color'] = 'default'
    
    # Create template variables
    template_vars = {
        "title": "YOLO模型评估报告",
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "models": all_results,
        "best_ap_model": best_ap_model,
        "best_auc_model": best_auc_model,
        "images": images,
        "pr_key_points_table": pr_html_table,
        "roc_key_points_table": roc_html_table
    }
    
    # Render the template
    html_out = template.render(template_vars)
    
    # 创建增强的CSS，确保表格在PDF中正确显示
    enhanced_css = get_css() + """
    /* 额外的表格样式，确保在PDF中显示 */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 1cm 0 !important;
        page-break-inside: avoid !important;
    }
    
    th, td {
        border: 1px solid #cbd5e0 !important;
        padding: 0.3cm !important;
        text-align: center !important;
    }
    
    th {
        background-color: #edf2f7 !important;
        font-weight: 500 !important;
        color: #2d3748 !important;
    }
    
    tr:nth-child(even) {
        background-color: #f7fafc !important;
    }
    
    .key-points-container {
        page-break-inside: avoid !important;
        margin: 1cm 0 !important;
    }
    """
    
    # 保存带有内联CSS的完整HTML文件
    html_output_path = output_path if output_path.endswith('.html') else output_path.replace('.pdf', '.html')
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n<head>\n')
        f.write('<meta charset="UTF-8">\n')
        f.write('<style>\n')
        f.write(enhanced_css)
        f.write('\n</style>\n')
        f.write('</head>\n<body>\n')
        f.write(html_out)
        f.write('\n</body>\n</html>')
    
    # 如果只需要生成HTML文件或输出路径已经是HTML文件，则不生成PDF
    if save_html and output_path.endswith('.html'):
        print(f"✅ HTML报告已生成: {html_output_path}")
        return html_output_path
    
    # 创建更健壮的完整HTML字符串，包含内联样式
    complete_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
    {enhanced_css}
    </style>
    </head>
    <body>
    {html_out}
    </body>
    </html>
    '''
    
    # 使用更健壮的方式生成PDF
    try:
        # 创建临时HTML文件，确保WeasyPrint能够正确加载所有内容
        temp_html_path = os.path.join(current_dir, 'temp_report.html')
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(complete_html)
        
        # 使用临时HTML文件生成PDF
        HTML(temp_html_path).write_pdf(
            output_path,
            stylesheets=[CSS(string=enhanced_css)]
        )
        
        # 删除临时文件
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
        
        print(f"✅ PDF报告已生成: {output_path}")
        if save_html:
            print(f"✅ HTML报告已生成: {html_output_path}")
        
        return output_path
    except Exception as e:
        print(f"生成PDF时出错: {e}")
        print(f"✅ HTML报告已生成: {html_output_path}")
        return None

def get_css():
    """Return CSS style for the PDF report"""
    return """
    @page {
        size: A4;
        margin: 2cm;
        @top-right {
            content: counter(page);
            font-size: 9pt;
            color: #718096;
        }
        @bottom-center {
            content: "技术评估报告";
            font-size: 9pt;
            color: #718096;
        }
    }
    
    @page :first {
        @top-right {
            content: none;
        }
        @bottom-center {
            content: none;
        }
    }
    
    /* 基础样式 */
    body {
        font-family: 'Helvetica', 'Arial', sans-serif;
        line-height: 1.5;
        color: #333;
        margin: 0;
        padding: 0;
    }
    
    h1 {
        color: #1a365d;
        font-weight: 500;
        margin-bottom: 0.8cm;
    }
    
    h2 {
        color: #2a4365;
        margin-top: 1cm;
        margin-bottom: 0.5cm;
        font-weight: 500;
    }
    
    h3 {
        color: #2c5282;
        margin-top: 0.8cm;
        margin-bottom: 0.4cm;
        font-weight: 500;
    }
    
    /* 表格标题样式 */
    .table-title {
        font-size: 14pt;
        font-weight: 500;
        color: #2c5282;
        margin-bottom: 0.6cm;
        border-bottom: 1px solid #4299e1;
        padding-bottom: 0.2cm;
    }
    
    /* 封面样式 */
    .cover {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 25cm;
        position: relative;
        background-color: #fafafa;
        padding: 3cm;
    }
    
    .cover-header {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1cm;
    }
    
    .logo {
        font-size: 16pt;
        font-weight: 500;
        color: #2b6cb0;
        letter-spacing: 1px;
        text-align: center;
    }
    
    .cover-content {
        text-align: center;
        margin-top: 6cm;
    }
    
    .cover-content h1 {
        font-size: 36pt;
        color: #1a365d;
        margin-bottom: 2cm;
        page-break-before: avoid;
        line-height: 1.3;
        font-weight: 500;
    }
    
    .cover-subtitle {
        font-size: 20pt;
        color: #4a5568;
        margin-bottom: 5cm;
    }
    
    .cover-details {
        display: flex;
        justify-content: center;
        gap: 4cm;
        margin-top: 3cm;
    }
    
    .detail-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5cm;
    }
    
    .detail-label {
        font-size: 12pt;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .detail-value {
        font-size: 14pt;
        font-weight: 500;
        color: #2b6cb0;
    }
    
    .cover-footer {
        position: absolute;
        bottom: 3cm;
        width: 100%;
        text-align: center;
        left: 0;
    }
    
    .company-info {
        font-size: 14pt;
        font-weight: 500;
        color: #2d3748;
        margin-bottom: 0.5cm;
    }
    
    .confidentiality {
        font-size: 9pt;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 目录样式 */
    .toc-section {
        page-break-after: always;
    }
    
    .toc-header {
        margin-bottom: 1.5cm;
    }
    
    .toc {
        margin: 0 1cm;
    }
    
    .toc-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5cm;
        border-bottom: 1px dotted #e2e8f0;
        padding-bottom: 0.3cm;
    }
    
    .toc-number {
        font-weight: 500;
        color: #2b6cb0;
        width: 1.5cm;
    }
    
    .toc-title {
        flex-grow: 1;
        font-size: 11pt;
    }
    
    .toc-page {
        font-weight: 500;
        color: #4a5568;
    }
    
    /* 章节样式 */
    .section {
        margin-bottom: 1cm;
        page-break-before: always;
    }
    
    .section:first-of-type {
        page-break-before: avoid;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1cm;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4cm;
    }
    
    .section-number {
        font-size: 20pt;
        font-weight: 500;
        color: #2b6cb0;
        margin-right: 0.5cm;
        padding: 0.2cm 0.5cm;
        background-color: #ebf8ff;
        border-radius: 50%;
    }
    
    /* 执行摘要样式 */
    .executive-summary {
        background-color: #f7fafc;
        padding: 1cm;
        border-radius: 0.2cm;
    }
    
    .summary-card {
        background-color: white;
        padding: 1cm;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        margin-bottom: 1cm;
    }
    
    .summary-intro {
        font-size: 11pt;
        margin-bottom: 1cm;
        line-height: 1.6;
    }
    
    .key-findings {
        margin-top: 0.8cm;
    }
    
    .findings-header {
        font-size: 13pt;
        color: #2d3748;
        margin-bottom: 0.8cm;
        padding-bottom: 0.2cm;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .finding-item {
        display: flex;
        margin-bottom: 0.8cm;
        align-items: flex-start;
    }
    
    .finding-content {
        flex-grow: 1;
    }
    
    .finding-title {
        font-weight: 500;
        font-size: 11pt;
        margin-bottom: 0.2cm;
        color: #2c5282;
    }
    
    .finding-desc {
        font-size: 10pt;
        color: #4a5568;
    }
    
    .performance-header {
        margin-top: 1.5cm;
        color: #2d3748;
        font-size: 13pt;
        font-weight: 500;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.2cm;
        margin-bottom: 0.8cm;
    }
    
    .performance-grid {
        display: flex;
        justify-content: space-between;
        gap: 0.5cm;
        margin-top: 0.8cm;
        margin-bottom: 1cm;
    }
    
    /* 为单个模型的性能卡片添加样式 */
    .performance-grid.single-model {
        justify-content: center;
    }
    
    .performance-grid.single-model .performance-card {
        width: 60%;
        padding: 1cm;
    }
    
    .performance-grid.single-model .metric-value {
        font-size: 18pt;
    }
    
    .performance-card {
        background-color: white;
        border-radius: 0.2cm;
        padding: 0.8cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        text-align: center;
        position: relative;
        border-top: 2px solid #2b6cb0;
        width: 30%;
    }
    
    .model-yolov8-n {
        border-top-color: #38b2ac;
    }
    
    .model-yolov8-s {
        border-top-color: #4299e1;
    }
    
    .model-yolov8-m {
        border-top-color: #667eea;
    }
    
    .model-name {
        font-weight: 500;
        font-size: 12pt;
        margin-bottom: 0.5cm;
        color: #2d3748;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
    }
    
    .metric {
        text-align: center;
    }
    
    .metric-value {
        font-size: 16pt;
        font-weight: 500;
        color: #2b6cb0;
    }
    
    .metric-label {
        font-size: 9pt;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2cm;
    }
    
    .summary-table-container {
        margin-top: 1.5cm;
    }
    
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5cm;
    }
    
    .summary-table th, .summary-table td {
        border: 1px solid #e2e8f0;
        padding: 0.4cm;
        text-align: center;
    }
    
    .summary-table th {
        background-color: #f7fafc;
        color: #2d3748;
        font-weight: 500;
    }
    
    .summary-table tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    /* 图表样式 */
    .charts-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 1.2cm;
    }
    
    .chart-card {
        background-color: white;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        overflow: hidden;
        margin-bottom: 0.5cm;
    }
    
    .chart-header {
        padding: 0.8cm;
        background-color: #f7fafc;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .chart-header h2 {
        margin: 0 0 0.3cm 0;
        font-size: 14pt;
        color: #2d3748;
        font-weight: 500;
    }
    
    .chart-desc {
        font-size: 10pt;
        color: #4a5568;
    }
    
    .chart-container {
        padding: 1cm;
        text-align: center;
    }
    
    .chart-container img {
        max-width: 100%;
        max-height: 12cm;
        margin-bottom: 0.8cm;
    }
    
    .chart-note {
        background-color: #f7fafc;
        padding: 0.5cm;
        border-radius: 0.2cm;
        text-align: left;
    }
    
    .note-title {
        font-weight: 500;
        color: #2b6cb0;
        margin-bottom: 0.2cm;
        font-size: 10pt;
    }
    
    .note-content {
        font-size: 9pt;
        color: #4a5568;
        line-height: 1.5;
    }
    
    /* 关键点表格样式 */
    .key-points-container {
        margin-top: 1.5cm;
        padding: 1cm;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.2cm;
        page-break-inside: avoid; /* 防止表格被分页 */
    }
    
    .key-points-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5cm;
        font-size: 9pt;
    }
    
    .key-points-table th, .key-points-table td {
        border: 1px solid #cbd5e0;
        padding: 0.3cm;
        text-align: center;
    }
    
    .key-points-table th {
        background-color: #edf2f7;
        font-weight: 500;
        color: #2d3748;
    }
    
    .key-points-table tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    .key-points-table tr:hover {
        background-color: #ebf8ff;
    }
    
    /* 阈值选择指南样式 */
    .threshold-selection-guide {
        background-color: white;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        padding: 1cm;
        margin-top: 1.5cm;
    }
    
    .threshold-selection-guide h3 {
        color: #2d3748;
        margin-top: 0;
        margin-bottom: 0.8cm;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.3cm;
        font-weight: 500;
    }
    
    .guide-content {
        font-size: 10pt;
        color: #4a5568;
    }
    
    .guide-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5cm;
    }
    
    .guide-table th, .guide-table td {
        border: 1px solid #e2e8f0;
        padding: 0.3cm;
        text-align: left;
    }
    
    .guide-table th {
        background-color: #edf2f7;
        color: #2d3748;
        font-weight: 500;
    }
    
    .guide-table tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    /* 分析与建议样式 */
    .analysis-container {
        padding: 0.5cm;
    }
    
    .model-analysis-header {
        margin-bottom: 1cm;
        padding-bottom: 0.3cm;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .models-analysis {
        display: flex;
        flex-wrap: wrap;
        gap: 1cm;
        margin-bottom: 1.5cm;
    }
    
    .model-analysis-card {
        flex: 1;
        min-width: 48%;
        background-color: white;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        overflow: hidden;
    }
    
    .model-header {
        background-color: #2b6cb0;
        color: white;
        padding: 0.5cm;
        font-weight: 500;
        font-size: 12pt;
    }
    
    .model-header.model-yolov8-n {
        background-color: #38b2ac;
    }
    
    .model-header.model-yolov8-s {
        background-color: #4299e1;
    }
    
    .model-header.model-yolov8-m {
        background-color: #667eea;
    }
    
    .model-content {
        padding: 0.8cm;
        display: flex;
        gap: 1cm;
    }
    
    .model-strengths, .model-weaknesses {
        flex: 1;
    }
    
    .model-strengths h3, .model-weaknesses h3 {
        color: #2d3748;
        font-size: 11pt;
        margin-top: 0;
        margin-bottom: 0.5cm;
        padding-bottom: 0.2cm;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .model-strengths ul, .model-weaknesses ul {
        padding-left: 0.8cm;
        margin: 0;
    }
    
    .model-strengths li, .model-weaknesses li {
        margin-bottom: 0.3cm;
        font-size: 10pt;
    }
    
    .recommendations {
        margin-top: 1.5cm;
    }
    
    .recommendation-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8cm;
        margin-top: 0.8cm;
    }
    
    .recommendation-card {
        flex: 1;
        min-width: 30%;
        background-color: white;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        padding: 0.8cm;
    }
    
    .rec-content {
        flex-grow: 1;
    }
    
    .rec-content h3 {
        margin: 0 0 0.3cm 0;
        font-size: 12pt;
        color: #2d3748;
        font-weight: 500;
    }
    
    .rec-content p {
        margin: 0;
        font-size: 10pt;
        color: #4a5568;
    }
    
    /* 结论样式 */
    .conclusion-content {
        padding: 0.5cm;
    }
    
    .conclusion-card {
        background-color: white;
        border-radius: 0.2cm;
        box-shadow: 0 0 0.2cm rgba(0, 0, 0, 0.05);
        padding: 1cm;
    }
    
    .conclusion-text {
        font-size: 11pt;
        line-height: 1.7;
        color: #4a5568;
        margin-bottom: 0.8cm;
    }
    
    .final-recommendation {
        margin-top: 1.5cm;
        background-color: #f7fafc;
        border-radius: 0.2cm;
        padding: 0.8cm;
        width: 80%; /* 固定宽度 */
        max-width: 800px; /* 最大宽度 */
        margin-left: auto; /* 水平居中 */
        margin-right: auto; /* 水平居中 */
        box-sizing: border-box; /* 确保padding不会增加元素总宽度 */
    }
    
    .recommendation-header {
        display: flex;
        align-items: center;
        gap: 0.5cm;
        margin-bottom: 0.8cm;
    }
    
    .recommendation-header h2 {
        margin: 0;
        font-size: 14pt;
        color: #2d3748;
        font-weight: 500;
    }
    
    .recommendation-details {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5cm;
        width: 100%; /* 确保使用全部可用宽度 */
    }
    
    .recommended-model {
        font-size: 16pt;
        font-weight: 500;
        color: #2b6cb0;
        text-align: center; /* 文本居中 */
    }
    
    .recommendation-metrics {
        display: flex;
        gap: 1cm;
        justify-content: center; /* 水平居中 */
        width: 60%; /* 固定宽度 */
        min-width: 300px; /* 最小宽度 */
    }
    
    .rec-metric {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.2cm;
        width: 100px; /* 固定宽度 */
        text-align: center; /* 文本居中 */
    }
    
    .metric-name {
        font-size: 10pt;
        color: #718096;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 14pt;
        font-weight: 500;
        color: #2b6cb0;
    }
    
    .footer {
        margin-top: 2cm;
        text-align: center;
        color: #718096;
    }
    
    .footer-logo {
        font-size: 12pt;
        font-weight: 500;
        color: #2b6cb0;
        margin-bottom: 0.3cm;
    }
    
    .footer-info {
        font-size: 9pt;
    }
    """

if __name__ == "__main__":
    # Example usage
    # This would typically be imported and called from your main evaluation script
    
    # Sample data (for testing)
    sample_results = [
        {
            'name': 'YOLOv8-m',
            'ap': 0.823,
            'auc': 0.912
        },
        {
            'name': 'YOLOv8-n',
            'ap': 0.756,
            'auc': 0.883
        },
        {
            'name': 'YOLOv8-s',
            'ap': 0.795,
            'auc': 0.896
        }
    ]
    
    generate_pdf_report(sample_results, save_html=True)