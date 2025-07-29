import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def parse_json_data(file_path):
    """解析JSON文件并提取坐标和值"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 用于存储每个坐标点的所有值
    coordinate_values = defaultdict(list)
    
    for key, value in data.items():
        # 跳过统计信息
        if key in ['x_err_std', 'y_err_std']:
            continue
            
        # 尝试匹配不同的key格式
        # 格式1: "[x,y]-timestamp" (测试结果格式)
        match1 = re.match(r'\[([^,]+),\s*([^\]]+)\]-(\d+)', key)
        if match1:
            x = float(match1.group(1))
            y = float(match1.group(2))
            coordinate_values[(x, y)].append(value)
            continue
            
        # 格式2: "[x1,y1]-[x2,y2]-[x3,y3]" (训练结果格式)
        match2 = re.match(r'\[([^,]+),\s*([^\]]+)\]-\[([^,]+),\s*([^\]]+)\]-\[([^,]+),\s*([^\]]+)\]', key)
        if match2:
            # 取第一个坐标点作为主要坐标
            x = float(match2.group(1))
            y = float(match2.group(2))
            coordinate_values[(x, y)].append(value)
            continue
            
        # 如果都不匹配，打印警告
        print(f"无法解析key: {key}")
    
    # 计算每个坐标点的平均值
    heatmap_data = {}
    for coord, values in coordinate_values.items():
        heatmap_data[coord] = np.mean(values)
    
    return heatmap_data

def create_scatter_heatmap(heatmap_data, output_file='scatter_heatmap.png'):
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
    plt.title('Scatter Heatmap of Average Values by Coordinates', fontsize=16)
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

def create_enhanced_scatter_heatmap(heatmap_data, output_file='enhanced_scatter_heatmap.png'):
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
    scatter1 = ax1.scatter(x_coords, y_coords, 
                           c=values, 
                           cmap='viridis', 
                           s=60, 
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=0.5)
    
    ax1.set_title('Scatter Heatmap of Average Values', fontsize=14)
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

def main():
    # 解析数据
    file_path = 'output/fine_tuned_test_result.json'
    heatmap_data = parse_json_data(file_path)
    
    print(f"处理了 {len(heatmap_data)} 个唯一坐标点")
    
    # 显示一些统计信息
    values = list(heatmap_data.values())
    print(f"值的范围: {min(values):.2f} - {max(values):.2f}")
    print(f"平均值: {np.mean(values):.2f}")
    print(f"标准差: {np.std(values):.2f}")
    
    # 创建散点热力图
    create_scatter_heatmap(heatmap_data)
    
    # 创建增强版散点热力图
    create_enhanced_scatter_heatmap(heatmap_data)

if __name__ == "__main__":
    main() 