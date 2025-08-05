#!/usr/bin/env python3
"""
统一的热力图生成器
支持处理训练数据和测试数据，生成散点热力图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import argparse
import os

def parse_json_data(file_path, data_type='auto'):
    """
    解析JSON文件并提取坐标和值
    
    Args:
        file_path: JSON文件路径
        data_type: 数据类型 ('train', 'test', 'auto')
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 用于存储每个坐标点的所有值
    coordinate_values = defaultdict(list)
    
    for key, value in data.items():
        # 跳过统计信息
        if key in ['x_err_std', 'y_err_std']:
            continue
            
        # 根据数据类型或自动检测选择解析方式
        if data_type == 'train' or (data_type == 'auto' and ']-[' in key):
            # 训练数据格式: "[x1,y1]-[x2,y2]-[x3,y3]"
            match = re.match(r'\[([^,]+),\s*([^\]]+)\]-\[([^,]+),\s*([^\]]+)\]-\[([^,]+),\s*([^\]]+)\]', key)
            if match:
                # 取第一个坐标点作为主要坐标
                x = float(match.group(1))
                y = float(match.group(2))
                coordinate_values[(x, y)].append(value)
            else:
                print(f"无法解析训练数据key: {key}")
                
        elif data_type == 'test' or (data_type == 'auto' and ']-' in key and not ']-[' in key):
            # 测试数据格式: "[x,y]-timestamp"
            match = re.match(r'\[([^,]+),\s*([^\]]+)\]-(\d+)', key)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                coordinate_values[(x, y)].append(value)
            else:
                print(f"无法解析测试数据key: {key}")
        else:
            print(f"无法解析key: {key}")
    
    # 计算每个坐标点的平均值
    heatmap_data = {}
    for coord, values in coordinate_values.items():
        heatmap_data[coord] = np.mean(values)
    
    return heatmap_data

def create_scatter_heatmap(heatmap_data, output_file='scatter_heatmap.png', title='Scatter Heatmap'):
    """创建散点图形式的热力图"""
    if not heatmap_data:
        print("没有数据可以绘制")
        return
    
    # 提取坐标和值
    x_coords = [coord[0] for coord in heatmap_data.keys()]
    y_coords = [coord[1] for coord in heatmap_data.keys()]
    values = list(heatmap_data.values())
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 创建散点图，颜色表示值的大小
    scatter = plt.scatter(x_coords, y_coords, 
                         c=values, 
                         cmap='viridis', 
                         s=50,  # 点的大小
                         alpha=0.7,  # 透明度
                         edgecolors='black',  # 边框颜色
                         linewidth=0.5)  # 边框宽度
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Value', fontsize=12)
    
    # 设置标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"散点热力图已保存为: {output_file}")
    
    # 显示图片
    plt.show()

def create_enhanced_scatter_heatmap(heatmap_data, output_file='enhanced_scatter_heatmap.png', title='Enhanced Scatter Heatmap'):
    """创建增强版散点热力图，包含更多信息"""
    if not heatmap_data:
        print("没有数据可以绘制")
        return
    
    # 提取坐标和值
    x_coords = [coord[0] for coord in heatmap_data.keys()]
    y_coords = [coord[1] for coord in heatmap_data.keys()]
    values = list(heatmap_data.values())
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 主散点图
    ax1.invert_yaxis()
    scatter1 = ax1.scatter(x_coords, y_coords, 
                           c=values, 
                           cmap='viridis', 
                           s=60, 
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=0.5)
    
    ax1.set_title(title, fontsize=14)
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Average Value', fontsize=12)
    
    # 值分布直方图
    ax2.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution of Average Values', fontsize=14)
    ax2.set_xlabel('Average Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = np.mean(values)
    std_val = np.std(values)
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax2.axvline(mean_val + std_val, color='orange', linestyle=':', label=f'+1 Std: {mean_val + std_val:.2f}')
    ax2.axvline(mean_val - std_val, color='orange', linestyle=':', label=f'-1 Std: {mean_val - std_val:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"增强版散点热力图已保存为: {output_file}")
    plt.show()

def create_comparison_heatmap(train_data, test_data, output_file='comparison_heatmap.png'):
    """创建训练数据和测试数据的对比热力图"""
    if not train_data or not test_data:
        print("需要训练数据和测试数据才能创建对比图")
        return
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 提取训练数据
    train_x = [coord[0] for coord in train_data.keys()]
    train_y = [coord[1] for coord in train_data.keys()]
    train_values = list(train_data.values())
    
    # 提取测试数据
    test_x = [coord[0] for coord in test_data.keys()]
    test_y = [coord[1] for coord in test_data.keys()]
    test_values = list(test_data.values())
    
    # 训练数据散点图
    ax1.invert_yaxis()
    scatter1 = ax1.scatter(train_x, train_y, 
                           c=train_values, 
                           cmap='viridis', 
                           s=50, 
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=0.5)
    ax1.set_title('Training Data Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Average Value', fontsize=12)
    
    # 测试数据散点图
    ax2.invert_yaxis()
    scatter2 = ax2.scatter(test_x, test_y, 
                           c=test_values, 
                           cmap='viridis', 
                           s=50, 
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=0.5)
    ax2.set_title('Test Data Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Average Value', fontsize=12)
    
    # 训练数据分布
    ax3.hist(train_values, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Training')
    ax3.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Average Value', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 添加训练数据统计信息
    train_mean = np.mean(train_values)
    train_std = np.std(train_values)
    ax3.axvline(train_mean, color='red', linestyle='--', label=f'Train Mean: {train_mean:.2f}')
    ax3.axvline(train_mean + train_std, color='orange', linestyle=':', label=f'Train +1 Std: {train_mean + train_std:.2f}')
    ax3.axvline(train_mean - train_std, color='orange', linestyle=':', label=f'Train -1 Std: {train_mean - train_std:.2f}')
    ax3.legend()
    
    # 测试数据分布
    ax4.hist(test_values, bins=30, alpha=0.7, color='green', edgecolor='black', label='Test')
    ax4.set_title('Test Data Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 添加测试数据统计信息
    test_mean = np.mean(test_values)
    test_std = np.std(test_values)
    ax4.axvline(test_mean, color='red', linestyle='--', label=f'Test Mean: {test_mean:.2f}')
    ax4.axvline(test_mean + test_std, color='orange', linestyle=':', label=f'Test +1 Std: {test_mean + test_std:.2f}')
    ax4.axvline(test_mean - test_std, color='orange', linestyle=':', label=f'Test -1 Std: {test_mean - test_std:.2f}')
    ax4.legend()
    
    # 添加总标题
    fig.suptitle('Training vs Test Data Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"对比热力图已保存为: {output_file}")
    plt.show()

def create_overlay_heatmap(train_data, test_data, output_file='overlay_heatmap.png'):
    """创建训练数据和测试数据的叠加热力图"""
    if not train_data or not test_data:
        print("需要训练数据和测试数据才能创建叠加图")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 提取数据
    train_x = [coord[0] for coord in train_data.keys()]
    train_y = [coord[1] for coord in train_data.keys()]
    train_values = list(train_data.values())
    
    test_x = [coord[0] for coord in test_data.keys()]
    test_y = [coord[1] for coord in test_data.keys()]
    test_values = list(test_data.values())
    
    # 叠加散点图
    ax1.invert_yaxis()
    
    # 绘制训练数据
    scatter_train = ax1.scatter(train_x, train_y, 
                                c=train_values, 
                                cmap='Blues', 
                                s=60, 
                                alpha=0.8,
                                edgecolors='black',
                                linewidth=0.5,
                                label='Training Data')
    
    # 绘制测试数据
    scatter_test = ax1.scatter(test_x, test_y, 
                               c=test_values, 
                               cmap='Reds', 
                               s=60, 
                               alpha=0.8,
                               edgecolors='black',
                               linewidth=0.5,
                               label='Test Data')
    
    ax1.set_title('Training vs Test Data Overlay', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter_train, ax=ax1, label='Training Values')
    cbar2 = plt.colorbar(scatter_test, ax=ax1, label='Test Values')
    
    # 对比直方图
    ax2.hist(train_values, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Training Data')
    ax2.hist(test_values, bins=30, alpha=0.7, color='red', edgecolor='black', label='Test Data')
    ax2.set_title('Value Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加统计信息
    train_mean = np.mean(train_values)
    test_mean = np.mean(test_values)
    ax2.axvline(train_mean, color='blue', linestyle='--', label=f'Train Mean: {train_mean:.2f}')
    ax2.axvline(test_mean, color='red', linestyle='--', label=f'Test Mean: {test_mean:.2f}')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"叠加热力图已保存为: {output_file}")
    plt.show()

def print_statistics(heatmap_data, data_type):
    """打印数据统计信息"""
    if not heatmap_data:
        print("没有数据可以分析")
        return
    
    values = list(heatmap_data.values())
    print(f"\n=== {data_type.upper()} 数据统计 ===")
    print(f"处理了 {len(heatmap_data)} 个唯一坐标点")
    print(f"值的范围: {min(values):.2f} - {max(values):.2f}")
    print(f"平均值: {np.mean(values):.2f}")
    print(f"标准差: {np.std(values):.2f}")
    print(f"中位数: {np.median(values):.2f}")

def process_data(file_path, data_type='auto', output_prefix=''):
    """处理数据并生成热力图"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"正在处理文件: {file_path}")
    print(f"数据类型: {data_type}")
    
    # 解析数据
    heatmap_data = parse_json_data(file_path, data_type)
    
    if not heatmap_data:
        print("没有有效数据可以处理")
        return
    
    # 打印统计信息
    print_statistics(heatmap_data, data_type)
    
    # 生成输出文件名
    if not output_prefix:
        output_prefix = f"{data_type}_"
    
    # 创建散点热力图
    scatter_output = f"{output_prefix}scatter_heatmap.png"
    scatter_title = f"{data_type.title()} Data: Scatter Heatmap of Average Values"
    create_scatter_heatmap(heatmap_data, scatter_output, scatter_title)
    
    # 创建增强版散点热力图
    enhanced_output = f"{output_prefix}enhanced_scatter_heatmap.png"
    enhanced_title = f"{data_type.title()} Data: Enhanced Scatter Heatmap"
    create_enhanced_scatter_heatmap(heatmap_data, enhanced_output, enhanced_title)
    
    return heatmap_data

def process_comparison(train_file, test_file, output_prefix='comparison_'):
    """处理训练数据和测试数据的对比"""
    print("=" * 50)
    print("处理训练数据和测试数据对比")
    print("=" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"训练数据文件不存在: {train_file}")
        return
    if not os.path.exists(test_file):
        print(f"测试数据文件不存在: {test_file}")
        return
    
    # 解析训练数据
    print(f"正在处理训练数据: {train_file}")
    train_data = parse_json_data(train_file, 'train')
    if train_data:
        print_statistics(train_data, 'train')
    
    # 解析测试数据
    print(f"正在处理测试数据: {test_file}")
    test_data = parse_json_data(test_file, 'test')
    if test_data:
        print_statistics(test_data, 'test')
    
    if not train_data or not test_data:
        print("无法处理数据，退出")
        return
    
    # 创建对比图
    comparison_output = f"{output_prefix}heatmap.png"
    create_comparison_heatmap(train_data, test_data, comparison_output)
    
    # 创建叠加图
    overlay_output = f"{output_prefix}overlay.png"
    create_overlay_heatmap(train_data, test_data, overlay_output)
    
    print("对比分析完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一的热力图生成器")
    parser.add_argument("file_path", nargs='?', help="JSON文件路径")
    parser.add_argument("--type", "-t", choices=['train', 'test', 'auto'], default='auto',
                       help="数据类型 (train/test/auto)")
    parser.add_argument("--output-prefix", "-o", default="",
                       help="输出文件前缀")
    parser.add_argument("--no-display", action="store_true",
                       help="不显示图形，只保存文件")
    parser.add_argument("--compare", "-c", nargs=2, metavar=('TRAIN_FILE', 'TEST_FILE'),
                       help="比较训练数据和测试数据")
    
    args = parser.parse_args()
    
    # 如果不显示图形，设置matplotlib后端
    if args.no_display:
        plt.switch_backend('Agg')
    
    print("=" * 50)
    print("统一热力图生成器")
    print("=" * 50)
    
    try:
        if args.compare:
            # 比较模式
            train_file, test_file = args.compare
            process_comparison(train_file, test_file, args.output_prefix)
        elif args.file_path:
            # 单文件处理模式
            heatmap_data = process_data(args.file_path, args.type, args.output_prefix)
            
            if heatmap_data:
                print("\n处理完成！")
            else:
                print("\n处理失败！")
        else:
            print("请提供文件路径或使用 --compare 参数")
            parser.print_help()
            
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main() 