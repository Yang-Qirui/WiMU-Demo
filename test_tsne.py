#!/usr/bin/env python3
"""
测试脚本：验证t-SNE可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def test_tsne_visualization():
    """
    测试t-SNE可视化功能
    """
    print("=== 测试t-SNE可视化功能 ===")
    
    # 创建模拟数据
    n_samples = 100
    embedding_dim = 64
    
    # 生成模拟embeddings
    embeddings = np.random.randn(n_samples, embedding_dim)
    
    # 生成模拟坐标（在0-100范围内）
    coordinates = np.random.uniform(0, 100, (n_samples, 2))
    
    print(f"创建模拟数据:")
    print(f"样本数量: {n_samples}")
    print(f"Embedding维度: {embedding_dim}")
    print(f"坐标范围: [{coordinates.min():.2f}, {coordinates.max():.2f}]")
    
    # 执行t-SNE降维
    print("\n执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取x和y坐标
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    # 第一个图：用x坐标作为颜色
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=x_coords, cmap='viridis', alpha=0.7, s=30)
    ax1.set_title('t-SNE Visualization - Colored by X Coordinate', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='X Coordinate')
    
    # 第二个图：用y坐标作为颜色
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=y_coords, cmap='plasma', alpha=0.7, s=30)
    ax2.set_title('t-SNE Visualization - Colored by Y Coordinate', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Y Coordinate')
    
    plt.tight_layout()
    
    # 保存测试图片
    test_output_dir = "output/test_tsne"
    os.makedirs(test_output_dir, exist_ok=True)
    test_path = os.path.join(test_output_dir, "test_tsne_visualization.png")
    plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE可视化图已保存到: {test_path}")
    
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
    
    print("\n🎉 t-SNE可视化功能测试通过！")

if __name__ == "__main__":
    test_tsne_visualization() 