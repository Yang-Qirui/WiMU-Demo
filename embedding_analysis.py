import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from torch.utils.data import DataLoader
import pickle

def save_embeddings_and_plot_similarity(model, graph_dataset, train_loader, test_loader, device, save_dir):
    """
    保存train和test loader中的embedding，并绘制相似性与距离关系图
    
    Args:
        model: 训练好的模型
        graph_dataset: 图数据集
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 设备
        save_dir: 保存目录
    """
    model.eval()
    
    # 创建保存目录
    os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    
    # 保存训练集embedding
    print("保存训练集embedding...")
    train_embeddings, train_coordinates, train_weights = extract_embeddings(model, graph_dataset, train_loader, device)
    
    # 保存测试集embedding
    print("保存测试集embedding...")
    test_embeddings, test_coordinates, test_weights = extract_embeddings(model, graph_dataset, test_loader, device)
    
    # 保存embedding数据
    embedding_data = {
        'train_embeddings': train_embeddings.cpu().numpy(),
        'train_coordinates': train_coordinates.cpu().numpy(),
        'train_weights': train_weights.cpu().numpy(),
        'test_embeddings': test_embeddings.cpu().numpy(),
        'test_coordinates': test_coordinates.cpu().numpy(),
        'test_weights': test_weights.cpu().numpy()
    }
    
    with open(f"{save_dir}/embeddings/embedding_data.pkl", 'wb') as f:
        pickle.dump(embedding_data, f)
    
    print(f"Embedding数据已保存到 {save_dir}/embeddings/embedding_data.pkl")
    
    # 绘制相似性与距离关系图
    print("绘制相似性与距离关系图...")
    plot_similarity_distance_relationship(train_embeddings, train_coordinates, test_embeddings, test_coordinates, save_dir)
    
    # 绘制embedding可视化图
    print("绘制embedding可视化图...")
    plot_embedding_visualization(train_embeddings, train_coordinates, test_embeddings, test_coordinates, save_dir)

def extract_embeddings(model, graph_dataset, data_loader, device):
    """
    从数据加载器中提取embedding
    
    Args:
        model: 模型
        graph_dataset: 图数据集
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        embeddings: 提取的embedding
        coordinates: 对应的坐标
        weights: 对应的权重
    """
    embeddings = []
    coordinates = []
    weights = []
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="提取embedding"):
            if len(batch_data) == 4:  # ContrastiveDataset格式
                coords, batch_weights, pos_weights, neg_weights = batch_data
                batch_weights = batch_weights.to(device)
                coords = coords.to(device)
                
                # 获取embedding
                batch_embeddings = model.gen_emb(graph_dataset, batch_weights)
                
                embeddings.append(batch_embeddings.squeeze(-1))
                coordinates.append(coords)
                weights.append(batch_weights)
            else:  # 普通Dataset格式
                batch_weights, coords, timestamps = batch_data
                batch_weights = batch_weights.to(device)
                coords = coords.to(device)
                
                # 获取embedding
                batch_embeddings = model.gen_emb(graph_dataset, batch_weights)
                
                embeddings.append(batch_embeddings.squeeze(-1))
                coordinates.append(coords)
                weights.append(batch_weights)
    
    embeddings = torch.cat(embeddings, dim=0)
    coordinates = torch.cat(coordinates, dim=0)
    weights = torch.cat(weights, dim=0)
    
    return embeddings, coordinates, weights

def plot_similarity_distance_relationship(train_embeddings, train_coordinates, test_embeddings, test_coordinates, save_dir):
    """
    绘制embedding相似性与真实距离的关系图
    
    Args:
        train_embeddings: 训练集embedding
        train_coordinates: 训练集坐标
        test_embeddings: 测试集embedding
        test_coordinates: 测试集坐标
        save_dir: 保存目录
    """
    # 计算embedding相似性
    def compute_similarities(embeddings):
        # 归一化embedding
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # 计算余弦相似性
        similarities = torch.matmul(embeddings_norm, embeddings_norm.T)
        return similarities
    
    # 计算真实距离
    def compute_distances(coordinates):
        distances = torch.zeros(coordinates.shape[0], coordinates.shape[0])
        for i in range(coordinates.shape[0]):
            for j in range(coordinates.shape[0]):
                distances[i, j] = torch.sqrt(torch.sum((coordinates[i] - coordinates[j]) ** 2))
        return distances
    
    # 处理训练集
    print("计算训练集相似性和距离...")
    train_similarities = compute_similarities(train_embeddings)
    train_distances = compute_distances(train_coordinates)
    
    # 处理测试集
    print("计算测试集相似性和距离...")
    test_similarities = compute_similarities(test_embeddings)
    test_distances = compute_distances(test_coordinates)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练集相似性vs距离
    train_sim_flat = train_similarities.flatten().cpu().numpy()
    train_dist_flat = train_distances.flatten().cpu().numpy()
    
    # 只取上三角矩阵（避免重复）
    mask = np.triu(np.ones_like(train_similarities.cpu().numpy()), k=1).astype(bool)
    train_sim_flat = train_similarities.cpu().numpy()[mask]
    train_dist_flat = train_distances.cpu().numpy()[mask]
    
    axes[0, 0].scatter(train_dist_flat, train_sim_flat, alpha=0.6, s=1)
    axes[0, 0].set_xlabel('真实距离')
    axes[0, 0].set_ylabel('Embedding相似性')
    axes[0, 0].set_title('训练集: Embedding相似性 vs 真实距离')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 测试集相似性vs距离
    mask = np.triu(np.ones_like(test_similarities.cpu().numpy()), k=1).astype(bool)
    test_sim_flat = test_similarities.cpu().numpy()[mask]
    test_dist_flat = test_distances.cpu().numpy()[mask]
    
    axes[0, 1].scatter(test_dist_flat, test_sim_flat, alpha=0.6, s=1)
    axes[0, 1].set_xlabel('真实距离')
    axes[0, 1].set_ylabel('Embedding相似性')
    axes[0, 1].set_title('测试集: Embedding相似性 vs 真实距离')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相似性分布
    axes[1, 0].hist(train_sim_flat, bins=50, alpha=0.7, label='训练集', density=True)
    axes[1, 0].hist(test_sim_flat, bins=50, alpha=0.7, label='测试集', density=True)
    axes[1, 0].set_xlabel('Embedding相似性')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('相似性分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 距离分布
    axes[1, 1].hist(train_dist_flat, bins=50, alpha=0.7, label='训练集', density=True)
    axes[1, 1].hist(test_dist_flat, bins=50, alpha=0.7, label='测试集', density=True)
    axes[1, 1].set_xlabel('真实距离')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('距离分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/similarity_distance_relationship.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算相关系数
    train_corr = np.corrcoef(train_sim_flat, train_dist_flat)[0, 1]
    test_corr = np.corrcoef(test_sim_flat, test_dist_flat)[0, 1]
    
    print(f"训练集相似性与距离相关系数: {train_corr:.4f}")
    print(f"测试集相似性与距离相关系数: {test_corr:.4f}")
    
    # 保存相关系数
    correlation_data = {
        'train_correlation': train_corr,
        'test_correlation': test_corr,
        'train_similarities_mean': float(np.mean(train_sim_flat)),
        'train_similarities_std': float(np.std(train_sim_flat)),
        'test_similarities_mean': float(np.mean(test_sim_flat)),
        'test_similarities_std': float(np.std(test_sim_flat)),
        'train_distances_mean': float(np.mean(train_dist_flat)),
        'train_distances_std': float(np.std(train_dist_flat)),
        'test_distances_mean': float(np.mean(test_dist_flat)),
        'test_distances_std': float(np.std(test_dist_flat))
    }
    
    with open(f"{save_dir}/plots/correlation_stats.json", 'w') as f:
        json.dump(correlation_data, f, indent=2)

def plot_embedding_visualization(train_embeddings, train_coordinates, test_embeddings, test_coordinates, save_dir):
    """
    绘制embedding的可视化图（使用t-SNE或PCA）
    
    Args:
        train_embeddings: 训练集embedding
        train_coordinates: 训练集坐标
        test_embeddings: 测试集embedding
        test_coordinates: 测试集坐标
        save_dir: 保存目录
    """
    # 合并训练集和测试集embedding
    all_embeddings = torch.cat([train_embeddings, test_embeddings], dim=0).cpu().numpy()
    all_coordinates = torch.cat([train_coordinates, test_coordinates], dim=0).cpu().numpy()
    
    # 使用PCA降维
    print("使用PCA降维...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 按数据集着色
    train_size = train_embeddings.shape[0]
    test_size = test_embeddings.shape[0]
    
    # PCA可视化
    axes[0].scatter(embeddings_2d[:train_size, 0], embeddings_2d[:train_size, 1], 
                    c='blue', alpha=0.6, s=20, label='训练集')
    axes[0].scatter(embeddings_2d[train_size:, 0], embeddings_2d[train_size:, 1], 
                    c='red', alpha=0.6, s=20, label='测试集')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    axes[0].set_title('Embedding PCA可视化')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 按坐标着色（使用距离原点的距离）
    distances = np.sqrt(np.sum(all_coordinates ** 2, axis=1))
    scatter = axes[1].scatter(embeddings_2d[:train_size, 0], embeddings_2d[:train_size, 1], 
                              c=distances[:train_size], cmap='viridis', alpha=0.6, s=20, label='训练集')
    scatter = axes[1].scatter(embeddings_2d[train_size:, 0], embeddings_2d[train_size:, 1], 
                              c=distances[train_size:], cmap='viridis', alpha=0.6, s=20, label='测试集')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].set_title('Embedding PCA可视化 (按距离着色)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='距离原点距离')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/embedding_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 尝试t-SNE（如果数据量不太大）
    if all_embeddings.shape[0] <= 5000:
        print("使用t-SNE降维...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0]//4))
            embeddings_tsne = tsne.fit_transform(all_embeddings)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # t-SNE可视化
            axes[0].scatter(embeddings_tsne[:train_size, 0], embeddings_tsne[:train_size, 1], 
                           c='blue', alpha=0.6, s=20, label='训练集')
            axes[0].scatter(embeddings_tsne[train_size:, 0], embeddings_tsne[train_size:, 1], 
                           c='red', alpha=0.6, s=20, label='测试集')
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')
            axes[0].set_title('Embedding t-SNE可视化')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 按坐标着色
            scatter = axes[1].scatter(embeddings_tsne[:train_size, 0], embeddings_tsne[:train_size, 1], 
                                     c=distances[:train_size], cmap='viridis', alpha=0.6, s=20, label='训练集')
            scatter = axes[1].scatter(embeddings_tsne[train_size:, 0], embeddings_tsne[train_size:, 1], 
                                     c=distances[train_size:], cmap='viridis', alpha=0.6, s=20, label='测试集')
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
            axes[1].set_title('Embedding t-SNE可视化 (按距离着色)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1], label='距离原点距离')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/plots/embedding_tsne_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"t-SNE可视化失败: {e}")

if __name__ == "__main__":
    # 示例用法
    print("这个文件提供了embedding分析和可视化的功能")
    print("主要函数:")
    print("- save_embeddings_and_plot_similarity(): 保存embedding并绘制相似性关系图")
    print("- extract_embeddings(): 从数据加载器中提取embedding")
    print("- plot_similarity_distance_relationship(): 绘制相似性与距离关系图")
    print("- plot_embedding_visualization(): 绘制embedding可视化图") 