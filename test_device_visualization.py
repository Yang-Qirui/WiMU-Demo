#!/usr/bin/env python3
"""
测试脚本：验证按设备分离的t-SNE可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import json

def create_test_device_data():
    """
    创建测试设备数据
    """
    # 创建模拟数据
    n_samples = 200
    embedding_dim = 64
    
    # 生成模拟embeddings
    embeddings = np.random.randn(n_samples, embedding_dim)
    
    # 生成模拟坐标
    coordinates = np.random.uniform(-100, 100, (n_samples, 2))
    
    # 创建模拟的设备信息
    device_info = {
        "355c796ae0f1848f": np.zeros(n_samples, dtype=bool),
        "a6c7c51c90f688c9": np.zeros(n_samples, dtype=bool),
        "b8d9e2f3a4b5c6d7": np.zeros(n_samples, dtype=bool)
    }
    
    # 分配设备（前1/3给设备1，中间1/3给设备2，最后1/3给设备3）
    device_info["355c796ae0f1848f"][:n_samples//3] = True
    device_info["a6c7c51c90f688c9"][n_samples//3:2*n_samples//3] = True
    device_info["b8d9e2f3a4b5c6d7"][2*n_samples//3:] = True
    
    return embeddings, coordinates, device_info

def test_device_tsne_visualization():
    """
    测试按设备分离的t-SNE可视化
    """
    print("=== 测试按设备分离的t-SNE可视化功能 ===")
    
    # 创建测试数据
    embeddings, coordinates, device_info = create_test_device_data()
    
    print(f"创建测试数据:")
    print(f"样本数量: {len(embeddings)}")
    print(f"Embedding维度: {embeddings.shape[1]}")
    print(f"设备数量: {len(device_info)}")
    
    # 执行t-SNE降维
    print("\n执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 提取x和y坐标
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 获取所有设备ID
    device_ids = list(device_info.keys())
    
    # 第一个图：用x坐标作为颜色，按设备分组
    for i, device_id in enumerate(device_ids):
        device_mask = device_info[device_id]
        if np.any(device_mask):
            axes[0, 0].scatter(embeddings_2d[device_mask, 0], embeddings_2d[device_mask, 1], 
                               c=x_coords[device_mask], cmap='viridis', alpha=0.7, s=30,
                               label=f'Device {device_id[:8]}...')
    axes[0, 0].set_title('t-SNE Visualization - Colored by X Coordinate (by Device)', fontsize=14)
    axes[0, 0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0, 0].legend()
    
    # 第二个图：用y坐标作为颜色，按设备分组
    for i, device_id in enumerate(device_ids):
        device_mask = device_info[device_id]
        if np.any(device_mask):
            axes[0, 1].scatter(embeddings_2d[device_mask, 0], embeddings_2d[device_mask, 1], 
                               c=y_coords[device_mask], cmap='plasma', alpha=0.7, s=30,
                               label=f'Device {device_id[:8]}...')
    axes[0, 1].set_title('t-SNE Visualization - Colored by Y Coordinate (by Device)', fontsize=14)
    axes[0, 1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0, 1].legend()
    
    # 第三个图：用设备ID作为颜色
    device_colors = []
    for i in range(len(embeddings)):
        for device_id in device_ids:
            if device_info[device_id][i]:
                device_colors.append(device_ids.index(device_id))
                break
        else:
            device_colors.append(-1)  # 未分配的设备
    
    scatter3 = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=device_colors, cmap='Set3', alpha=0.7, s=30)
    axes[1, 0].set_title('t-SNE Visualization - Colored by Device ID', fontsize=14)
    axes[1, 0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Device ID')
    
    # 第四个图：所有设备混合，用距离作为颜色
    scatter4 = axes[1, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=np.sqrt(x_coords**2 + y_coords**2), cmap='hot', alpha=0.7, s=30)
    axes[1, 1].set_title('t-SNE Visualization - Colored by Distance from Origin', fontsize=14)
    axes[1, 1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Distance from Origin')
    
    plt.tight_layout()
    
    # 保存测试图片
    test_output_dir = "output/test_device_tsne"
    os.makedirs(test_output_dir, exist_ok=True)
    test_path = os.path.join(test_output_dir, "test_device_tsne_visualization.png")
    plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 按设备分离的t-SNE可视化图已保存到: {test_path}")
    
    # 计算相关性
    corr_x_dim1 = np.corrcoef(x_coords, embeddings_2d[:, 0])[0, 1]
    corr_x_dim2 = np.corrcoef(x_coords, embeddings_2d[:, 1])[0, 1]
    corr_y_dim1 = np.corrcoef(y_coords, embeddings_2d[:, 0])[0, 1]
    corr_y_dim2 = np.corrcoef(y_coords, embeddings_2d[:, 1])[0, 1]
    
    print(f"\n相关性分析:")
    print(f"X坐标与t-SNE维度1的相关性: {corr_x_dim1:.4f}")
    print(f"X坐标与t-SNE维度2的相关性: {corr_x_dim2:.4f}")
    print(f"Y坐标与t-SNE维度1的相关性: {corr_y_dim1:.4f}")
    print(f"Y坐标与t-SNE维度2的相关性: {corr_y_dim2:.4f}")
    
    print("\n🎉 按设备分离的t-SNE可视化功能测试通过！")

if __name__ == "__main__":
    test_device_tsne_visualization() 