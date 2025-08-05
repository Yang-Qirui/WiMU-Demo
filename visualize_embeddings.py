import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import json
import argparse

def load_embeddings_data(file_path):
    """
    加载embeddings数据
    
    Args:
        file_path: embeddings文件路径
        
    Returns:
        embeddings_data: 包含embeddings和元数据的字典
    """
    if file_path.endswith('.pt'):
        embeddings_data = torch.load(file_path, map_location='cpu', weights_only=False)
    elif file_path.endswith('.npz'):
        embeddings_data = np.load(file_path)
        # 转换为字典格式
        embeddings_data = {key: embeddings_data[key] for key in embeddings_data.files}
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    return embeddings_data

def compute_similarity_and_distance(embeddings):
    """
    计算embeddings之间的cosine similarity和欧式距离
    
    Args:
        embeddings: shape为(n_samples, embedding_dim)的numpy数组
        
    Returns:
        similarities: cosine similarity矩阵
        distances: 欧式距离矩阵
    """
    print("计算cosine similarity...")
    similarities = cosine_similarity(embeddings)
    
    print("计算欧式距离...")
    distances = squareform(pdist(embeddings, metric='euclidean'))
    
    return similarities, distances

def extract_pairs(similarities, distances):
    """
    从相似度矩阵和距离矩阵中提取配对数据
    
    Args:
        similarities: cosine similarity矩阵
        distances: 欧式距离矩阵
        
    Returns:
        sim_values: cosine similarity值
        dist_values: 欧式距离值
    """
    n = similarities.shape[0]
    
    # 获取上三角矩阵的索引（避免重复和自身比较）
    upper_tri_indices = np.triu_indices(n, k=1)
    
    # 提取值
    sim_values = similarities[upper_tri_indices]
    dist_values = distances[upper_tri_indices]
    
    return sim_values, dist_values

def plot_similarity_distance_scatter(sim_values, dist_values, save_path, title="Embedding Similarity vs Distance"):
    """
    绘制相似度-距离散点图
    
    Args:
        sim_values: cosine similarity值
        dist_values: 欧式距离值
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))
    
    # 创建散点图
    plt.scatter(dist_values, sim_values, alpha=0.6, s=20, c='blue', edgecolors='none')
    
    # 添加趋势线
    z = np.polyfit(dist_values, sim_values, 1)
    p = np.poly1d(z)
    plt.plot(dist_values, p(dist_values), "r--", alpha=0.8, linewidth=2, label=f'Trend Line (y={z[0]:.3f}x+{z[1]:.3f})')
    
    # 计算相关系数
    correlation = np.corrcoef(dist_values, sim_values)[0, 1]
    
    plt.xlabel('Euclidean Distance (Denormalized)', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.title(f'{title}\nCorrelation: {correlation:.4f}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置坐标轴范围
    plt.xlim(0, np.percentile(dist_values, 95))  # 排除异常值
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"散点图已保存到: {save_path}")
    print(f"相关系数: {correlation:.4f}")

def plot_device_specific_scatter(embeddings_data, device_groups, output_dir, base_name):
    """
    为每个设备单独绘制散点图和趋势图
    
    Args:
        embeddings_data: embeddings数据字典
        device_groups: 按设备分组的数据
        output_dir: 输出目录
        base_name: 基础文件名
    """
    print("为每个设备绘制散点图和趋势图...")
    
    for device_id, device_data in device_groups.items():
        print(f"处理设备: {device_id}")
        
        # 创建设备特定的输出目录
        device_output_dir = os.path.join(output_dir, f"device_{device_id}")
        os.makedirs(device_output_dir, exist_ok=True)
        
        embeddings = device_data['embeddings']
        
        # 计算该设备的相似度和距离
        similarities, distances = compute_similarity_and_distance(embeddings)
        sim_values, dist_values = extract_pairs(similarities, distances)
        
        # 绘制散点图
        scatter_path = os.path.join(device_output_dir, f"{base_name}_device_{device_id}_similarity_distance_scatter.png")
        plot_similarity_distance_scatter(sim_values, dist_values, scatter_path, 
                                       title=f"Device {device_id}: Similarity vs Distance")
        
        # 保存设备特定的统计信息
        stats = {
            'device_id': device_id,
            'num_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
            'similarity_mean': float(np.mean(sim_values)),
            'similarity_std': float(np.std(sim_values)),
            'distance_mean': float(np.mean(dist_values)),
            'distance_std': float(np.std(dist_values)),
            'similarity_distance_correlation': float(np.corrcoef(sim_values, dist_values)[0, 1])
        }
        
        stats_path = os.path.join(device_output_dir, f"{base_name}_device_{device_id}_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"设备 {device_id} 的散点图已保存到: {device_output_dir}")
        print(f"设备 {device_id} 统计信息: 样本数={stats['num_samples']}, 相似度均值={stats['similarity_mean']:.4f}, 距离均值={stats['distance_mean']:.4f}")

def plot_device_clustering_scatter(embeddings_data, device_groups, output_dir, base_name, n_clusters=5):
    """
    为每个设备进行聚类并绘制散点图
    
    Args:
        embeddings_data: embeddings数据字典
        device_groups: 按设备分组的数据
        output_dir: 输出目录
        base_name: 基础文件名
        n_clusters: 聚类数量
    """
    print("为每个设备进行聚类并绘制散点图...")
    
    for device_id, device_data in device_groups.items():
        print(f"处理设备聚类: {device_id}")
        
        # 创建设备特定的输出目录
        device_output_dir = os.path.join(output_dir, f"device_{device_id}")
        os.makedirs(device_output_dir, exist_ok=True)
        
        embeddings = device_data['embeddings']
        coordinates = device_data['coordinates']
        
        # 如果样本数量少于聚类数量，调整聚类数量
        actual_n_clusters = min(n_clusters, len(embeddings))
        if actual_n_clusters < n_clusters:
            print(f"警告: 设备 {device_id} 的样本数量 ({len(embeddings)}) 少于请求的聚类数量 ({n_clusters})，使用 {actual_n_clusters} 个聚类")
        
        # 进行K-means聚类
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        
        # 使用聚类标签作为颜色，确保颜色一致性
        colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                             c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        
        # 创建legend元素，使用相同的颜色映射
        legend_elements = []
        for i in range(actual_n_clusters):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=colors[i], 
                                            markersize=8, 
                                            label=f'Cluster {i}'))
        
        # 添加聚类中心（如果样本数量足够）
        if len(embeddings) > actual_n_clusters:
            # 计算每个聚类的平均坐标作为聚类中心
            cluster_centers_coords = []
            for i in range(actual_n_clusters):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    center_coord = np.mean(coordinates[cluster_mask], axis=0)
                    cluster_centers_coords.append(center_coord)
            
            if cluster_centers_coords:
                cluster_centers_coords = np.array(cluster_centers_coords)
                plt.scatter(cluster_centers_coords[:, 0], cluster_centers_coords[:, 1], 
                           c='red', marker='x', s=200, linewidths=3)
                legend_elements.append(plt.Line2D([0], [0], marker='x', color='red', 
                                                markersize=10, markeredgewidth=3,
                                                label='Cluster Centers'))
        
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.title(f'Device {device_id}: Embedding Clustering (n_clusters={actual_n_clusters})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图片
        clustering_path = os.path.join(device_output_dir, f"{base_name}_device_{device_id}_clustering_scatter.png")
        plt.savefig(clustering_path, dpi=300, bbox_inches='tight')
    plt.close()
    
        # 保存聚类统计信息
        clustering_stats = {
            'device_id': device_id,
            'num_samples': len(embeddings),
            'n_clusters': actual_n_clusters,
            'cluster_distribution': {}
        }
        
        # 计算每个聚类的样本数量和平均坐标
        for i in range(actual_n_clusters):
            cluster_mask = cluster_labels == i
            cluster_count = np.sum(cluster_mask)
            if cluster_count > 0:
                cluster_coords = coordinates[cluster_mask]
                avg_coord = np.mean(cluster_coords, axis=0)
                clustering_stats['cluster_distribution'][f'cluster_{i}'] = {
                    'count': int(cluster_count),
                    'percentage': float(cluster_count / len(embeddings) * 100),
                    'avg_coordinate': avg_coord.tolist()
                }
        
        # 计算聚类质量指标
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        if len(set(cluster_labels)) > 1:  # 至少需要2个聚类才能计算这些指标
            try:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                calinski_avg = calinski_harabasz_score(embeddings, cluster_labels)
                clustering_stats['silhouette_score'] = float(silhouette_avg)
                clustering_stats['calinski_harabasz_score'] = float(calinski_avg)
            except:
                clustering_stats['silhouette_score'] = None
                clustering_stats['calinski_harabasz_score'] = None
        else:
            clustering_stats['silhouette_score'] = None
            clustering_stats['calinski_harabasz_score'] = None
        
        # 计算聚类内部的余弦相似度
        intra_cluster_similarities = []
        for i in range(actual_n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 1:  # 至少需要2个样本才能计算相似度
                cluster_embeddings = embeddings[cluster_mask]
                # 计算该聚类内所有样本对之间的余弦相似度
                cluster_similarities = cosine_similarity(cluster_embeddings)
                # 获取上三角矩阵的值（避免重复和自身比较）
                upper_tri_indices = np.triu_indices_from(cluster_similarities, k=1)
                cluster_sim_values = cluster_similarities[upper_tri_indices]
                intra_cluster_similarities.extend(cluster_sim_values)
                
                # 保存每个聚类的相似度统计
                clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_mean'] = float(np.mean(cluster_sim_values))
                clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_std'] = float(np.std(cluster_sim_values))
                clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_count'] = int(len(cluster_sim_values))
        
        # 计算聚类中心之间的余弦相似度
        cluster_centers = kmeans.cluster_centers_
        center_similarities = cosine_similarity(cluster_centers)
        # 获取上三角矩阵的值（避免重复和自身比较）
        upper_tri_indices = np.triu_indices_from(center_similarities, k=1)
        center_sim_values = center_similarities[upper_tri_indices]
        
        # 保存总体相似度统计
        if intra_cluster_similarities:
            clustering_stats['intra_cluster_similarity_mean'] = float(np.mean(intra_cluster_similarities))
            clustering_stats['intra_cluster_similarity_std'] = float(np.std(intra_cluster_similarities))
            clustering_stats['intra_cluster_similarity_count'] = int(len(intra_cluster_similarities))
        else:
            clustering_stats['intra_cluster_similarity_mean'] = None
            clustering_stats['intra_cluster_similarity_std'] = None
            clustering_stats['intra_cluster_similarity_count'] = 0
        
        if len(center_sim_values) > 0:
            clustering_stats['inter_cluster_center_similarity_mean'] = float(np.mean(center_sim_values))
            clustering_stats['inter_cluster_center_similarity_std'] = float(np.std(center_sim_values))
            clustering_stats['inter_cluster_center_similarity_count'] = int(len(center_sim_values))
        else:
            clustering_stats['inter_cluster_center_similarity_mean'] = None
            clustering_stats['inter_cluster_center_similarity_std'] = None
            clustering_stats['inter_cluster_center_similarity_count'] = 0
        
        # 保存聚类统计信息
        clustering_stats_path = os.path.join(device_output_dir, f"{base_name}_device_{device_id}_clustering_stats.json")
        with open(clustering_stats_path, 'w') as f:
            json.dump(clustering_stats, f, indent=2)
        
        print(f"设备 {device_id} 的聚类散点图已保存到: {clustering_path}")
        print(f"设备 {device_id} 聚类统计: 样本数={len(embeddings)}, 聚类数={actual_n_clusters}")
        if clustering_stats['silhouette_score'] is not None:
            print(f"设备 {device_id} 聚类质量: Silhouette={clustering_stats['silhouette_score']:.4f}, Calinski-Harabasz={clustering_stats['calinski_harabasz_score']:.4f}")
        
        # 打印相似度统计
        if clustering_stats['intra_cluster_similarity_mean'] is not None:
            print(f"设备 {device_id} 聚类内部相似度: 均值={clustering_stats['intra_cluster_similarity_mean']:.4f}, 方差={clustering_stats['intra_cluster_similarity_std']:.4f}")
        if clustering_stats['inter_cluster_center_similarity_mean'] is not None:
            print(f"设备 {device_id} 聚类中心间相似度: 均值={clustering_stats['inter_cluster_center_similarity_mean']:.4f}, 方差={clustering_stats['inter_cluster_center_similarity_std']:.4f}")
        
        # 打印各聚类内部相似度详情
        print(f"设备 {device_id} 各聚类内部相似度详情:")
        for i in range(actual_n_clusters):
            cluster_key = f'cluster_{i}'
            if cluster_key in clustering_stats['cluster_distribution']:
                cluster_stats = clustering_stats['cluster_distribution'][cluster_key]
                if 'intra_similarity_mean' in cluster_stats:
                    print(f"  聚类 {i}: 均值={cluster_stats['intra_similarity_mean']:.4f}, 方差={cluster_stats['intra_similarity_std']:.4f}, 样本数={cluster_stats['count']}")
        
        # 绘制聚类中心相似度热力图
        plot_cluster_centers_heatmap(cluster_centers, device_output_dir, base_name, device_id)

def plot_cluster_centers_heatmap(cluster_centers, output_dir, base_name, device_id=None):
    """
    绘制聚类中心相似度热力图
    
    Args:
        cluster_centers: 聚类中心数组
        output_dir: 输出目录
        base_name: 基础文件名
        device_id: 设备ID（如果为None则表示所有设备）
    """
    # 计算聚类中心之间的余弦相似度
    center_similarities = cosine_similarity(cluster_centers)
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    
    # 创建热力图
    im = plt.imshow(center_similarities, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # 设置坐标轴标签
    n_clusters = len(cluster_centers)
    plt.xticks(range(n_clusters), [f'Cluster {i}' for i in range(n_clusters)], fontsize=10)
    plt.yticks(range(n_clusters), [f'Cluster {i}' for i in range(n_clusters)], fontsize=10)
    
    # 在热力图上添加数值标签
    for i in range(n_clusters):
        for j in range(n_clusters):
            text = plt.text(j, i, f'{center_similarities[i, j]:.3f}',
                           ha="center", va="center", color="white" if center_similarities[i, j] < 0.5 else "black",
                           fontsize=9, fontweight='bold')
    
    # 设置标题
    if device_id is not None:
        title = f'Device {device_id}: Cluster Centers Similarity Matrix'
    else:
        title = f'All Devices: Cluster Centers Similarity Matrix'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(False)
    
    # 保存图片
    if device_id is not None:
        heatmap_path = os.path.join(output_dir, f"{base_name}_device_{device_id}_cluster_centers_heatmap.png")
    else:
        heatmap_path = os.path.join(output_dir, f"{base_name}_all_devices_cluster_centers_heatmap.png")
    
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"聚类中心相似度热力图已保存到: {heatmap_path}")
    
    # 打印相似度矩阵的统计信息
    print(f"聚类中心相似度矩阵统计:")
    print(f"  最大值: {np.max(center_similarities):.4f}")
    print(f"  最小值: {np.min(center_similarities):.4f}")
    print(f"  均值: {np.mean(center_similarities):.4f}")
    print(f"  标准差: {np.std(center_similarities):.4f}")
    
    # 计算对角线元素（自身相似度）和非对角线元素的统计
    diagonal_elements = np.diag(center_similarities)
    off_diagonal_elements = center_similarities[np.triu_indices_from(center_similarities, k=1)]
    
    print(f"  对角线元素（自身相似度）均值: {np.mean(diagonal_elements):.4f}")
    print(f"  非对角线元素（不同聚类间相似度）均值: {np.mean(off_diagonal_elements):.4f}")

def plot_all_devices_clustering_scatter(embeddings_data, device_groups, output_dir, base_name, n_clusters=5, denormalize_coordinates=True):
    """
    将所有设备的点放在一起进行聚类并绘制散点图
    
    Args:
        embeddings_data: embeddings数据字典
        device_groups: 按设备分组的数据
        output_dir: 输出目录
        base_name: 基础文件名
        n_clusters: 聚类数量
        denormalize_coordinates: 是否对坐标进行反归一化
    """
    print("将所有设备的点放在一起进行聚类...")
    
    # 收集所有设备的embeddings和坐标
    all_embeddings = []
    all_coordinates = []
    all_device_ids = []
    
    for device_id, device_data in device_groups.items():
        embeddings = device_data['embeddings']
        coordinates = device_data['coordinates']
        
        all_embeddings.append(embeddings)
        all_coordinates.append(coordinates)
        all_device_ids.extend([device_id] * len(embeddings))
    
    # 合并所有数据
    all_embeddings = np.vstack(all_embeddings)
    all_coordinates = np.vstack(all_coordinates)
    
    print(f"总样本数: {len(all_embeddings)}")
    print(f"设备数量: {len(device_groups)}")
    
    # 如果需要对坐标进行反归一化
    if denormalize_coordinates:
        norm_params_path = "./output/norm_params.pt"
        if os.path.exists(norm_params_path):
            try:
                norm_params = torch.load(norm_params_path, map_location='cpu')
                pos_range = norm_params['pos_range'].numpy()
                pos_min = norm_params['pos_min'].numpy()
                
                print(f"对坐标进行反归一化:")
                print(f"位置范围: {pos_range}")
                print(f"位置最小值: {pos_min}")
                
                # 反归一化坐标
                all_coordinates = all_coordinates * pos_range + pos_min
                
                print(f"坐标已反归一化")
                
            except Exception as e:
                print(f"加载归一化参数失败: {e}")
                print("使用原始坐标")
        else:
            print("未找到归一化参数文件，使用原始坐标")
    
    # 如果样本数量少于聚类数量，调整聚类数量
    actual_n_clusters = min(n_clusters, len(all_embeddings))
    if actual_n_clusters < n_clusters:
        print(f"警告: 总样本数量 ({len(all_embeddings)}) 少于请求的聚类数量 ({n_clusters})，使用 {actual_n_clusters} 个聚类")
    
    # 进行K-means聚类
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(15, 10))
    
    # 使用聚类标签作为颜色，设备ID作为标记
    unique_devices = list(set(all_device_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, actual_n_clusters))
    
    # 为每个聚类绘制散点图
    legend_elements = []
    for cluster_idx in range(actual_n_clusters):
        cluster_mask = cluster_labels == cluster_idx
        cluster_coords = all_coordinates[cluster_mask]
        cluster_devices = [all_device_ids[i] for i in range(len(all_device_ids)) if cluster_labels[i] == cluster_idx]
        
        # 为每个设备使用不同的标记
        for device_id in unique_devices:
            device_cluster_mask = np.array([(d == device_id and c == cluster_idx) 
                                          for d, c in zip(all_device_ids, cluster_labels)])
            if np.any(device_cluster_mask):
                device_coords = all_coordinates[device_cluster_mask]
                plt.scatter(device_coords[:, 0], device_coords[:, 1], 
                           c=[colors[cluster_idx]], alpha=0.7, s=50,
                           marker='o')
                
                # 为每个聚类-设备组合创建legend元素
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=colors[cluster_idx], 
                                                markersize=8, 
                                                label=f'Cluster {cluster_idx} - Device {device_id}'))
    
    # 添加聚类中心
    if len(all_embeddings) > actual_n_clusters:
        # 计算每个聚类的平均坐标作为聚类中心
        cluster_centers_coords = []
        for i in range(actual_n_clusters):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                center_coord = np.mean(all_coordinates[cluster_mask], axis=0)
                cluster_centers_coords.append(center_coord)
        
        if cluster_centers_coords:
            cluster_centers_coords = np.array(cluster_centers_coords)
            plt.scatter(cluster_centers_coords[:, 0], cluster_centers_coords[:, 1], 
                       c='red', marker='x', s=300, linewidths=4, label='Cluster Centers')
    
    plt.xlabel('X Coordinate (Denormalized)', fontsize=14)
    plt.ylabel('Y Coordinate (Denormalized)', fontsize=14)
    plt.title(f'All Devices: Embedding Clustering (n_clusters={actual_n_clusters})', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 添加聚类中心的legend元素
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='red', 
                                     markersize=10, markeredgewidth=3,
                                     label='Cluster Centers'))
    
    # 显示legend
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 保存图片
    clustering_path = os.path.join(output_dir, f"{base_name}_all_devices_clustering_scatter.png")
    plt.savefig(clustering_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存聚类统计信息
    clustering_stats = {
        'total_samples': len(all_embeddings),
        'n_clusters': actual_n_clusters,
        'n_devices': len(device_groups),
        'denormalized_coordinates': denormalize_coordinates,
        'cluster_distribution': {},
        'device_cluster_distribution': {}
    }
    
    # 计算每个聚类的统计信息
    for i in range(actual_n_clusters):
        cluster_mask = cluster_labels == i
        cluster_count = np.sum(cluster_mask)
        if cluster_count > 0:
            cluster_coords = all_coordinates[cluster_mask]
            cluster_devices = [all_device_ids[j] for j in range(len(all_device_ids)) if cluster_labels[j] == i]
            
            # 统计每个设备在该聚类中的数量
            device_counts = {}
            for device_id in unique_devices:
                device_count = cluster_devices.count(device_id)
                if device_count > 0:
                    device_counts[device_id] = device_count
            
            avg_coord = np.mean(cluster_coords, axis=0)
            clustering_stats['cluster_distribution'][f'cluster_{i}'] = {
                'count': int(cluster_count),
                'percentage': float(cluster_count / len(all_embeddings) * 100),
                'avg_coordinate': avg_coord.tolist(),
                'device_distribution': device_counts
            }
    
    # 计算每个设备在每个聚类中的分布
    for device_id in unique_devices:
        device_mask = np.array([d == device_id for d in all_device_ids])
        device_cluster_labels = cluster_labels[device_mask]
        device_cluster_dist = {}
        for i in range(actual_n_clusters):
            count = np.sum(device_cluster_labels == i)
            if count > 0:
                device_cluster_dist[f'cluster_{i}'] = int(count)
        clustering_stats['device_cluster_distribution'][device_id] = device_cluster_dist
    
    # 计算聚类质量指标
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    if len(set(cluster_labels)) > 1:  # 至少需要2个聚类才能计算这些指标
        try:
            silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
            calinski_avg = calinski_harabasz_score(all_embeddings, cluster_labels)
            clustering_stats['silhouette_score'] = float(silhouette_avg)
            clustering_stats['calinski_harabasz_score'] = float(calinski_avg)
        except:
            clustering_stats['silhouette_score'] = None
            clustering_stats['calinski_harabasz_score'] = None
            else:
        clustering_stats['silhouette_score'] = None
        clustering_stats['calinski_harabasz_score'] = None
    
    # 计算聚类内部的余弦相似度
    print("计算聚类内部余弦相似度...")
    intra_cluster_similarities = []
    for i in range(actual_n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 1:  # 至少需要2个样本才能计算相似度
            cluster_embeddings = all_embeddings[cluster_mask]
            # 计算该聚类内所有样本对之间的余弦相似度
            cluster_similarities = cosine_similarity(cluster_embeddings)
            # 获取上三角矩阵的值（避免重复和自身比较）
            upper_tri_indices = np.triu_indices_from(cluster_similarities, k=1)
            cluster_sim_values = cluster_similarities[upper_tri_indices]
            intra_cluster_similarities.extend(cluster_sim_values)
            
            # 保存每个聚类的相似度统计
            clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_mean'] = float(np.mean(cluster_sim_values))
            clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_std'] = float(np.std(cluster_sim_values))
            clustering_stats['cluster_distribution'][f'cluster_{i}']['intra_similarity_count'] = int(len(cluster_sim_values))
    
    # 计算聚类中心之间的余弦相似度
    print("计算聚类中心之间的余弦相似度...")
    cluster_centers = kmeans.cluster_centers_
    center_similarities = cosine_similarity(cluster_centers)
    # 获取上三角矩阵的值（避免重复和自身比较）
    upper_tri_indices = np.triu_indices_from(center_similarities, k=1)
    center_sim_values = center_similarities[upper_tri_indices]
    
    # 保存总体相似度统计
    if intra_cluster_similarities:
        clustering_stats['intra_cluster_similarity_mean'] = float(np.mean(intra_cluster_similarities))
        clustering_stats['intra_cluster_similarity_std'] = float(np.std(intra_cluster_similarities))
        clustering_stats['intra_cluster_similarity_count'] = int(len(intra_cluster_similarities))
    else:
        clustering_stats['intra_cluster_similarity_mean'] = None
        clustering_stats['intra_cluster_similarity_std'] = None
        clustering_stats['intra_cluster_similarity_count'] = 0
    
    if len(center_sim_values) > 0:
        clustering_stats['inter_cluster_center_similarity_mean'] = float(np.mean(center_sim_values))
        clustering_stats['inter_cluster_center_similarity_std'] = float(np.std(center_sim_values))
        clustering_stats['inter_cluster_center_similarity_count'] = int(len(center_sim_values))
    else:
        clustering_stats['inter_cluster_center_similarity_mean'] = None
        clustering_stats['inter_cluster_center_similarity_std'] = None
        clustering_stats['inter_cluster_center_similarity_count'] = 0
    
    # 保存聚类统计信息
    clustering_stats_path = os.path.join(output_dir, f"{base_name}_all_devices_clustering_stats.json")
    with open(clustering_stats_path, 'w') as f:
        json.dump(clustering_stats, f, indent=2)
    
    print(f"所有设备聚类散点图已保存到: {clustering_path}")
    print(f"聚类统计: 总样本数={len(all_embeddings)}, 聚类数={actual_n_clusters}, 设备数={len(device_groups)}")
    if clustering_stats['silhouette_score'] is not None:
        print(f"聚类质量: Silhouette={clustering_stats['silhouette_score']:.4f}, Calinski-Harabasz={clustering_stats['calinski_harabasz_score']:.4f}")
    
    # 打印相似度统计
    if clustering_stats['intra_cluster_similarity_mean'] is not None:
        print(f"聚类内部相似度: 均值={clustering_stats['intra_cluster_similarity_mean']:.4f}, 方差={clustering_stats['intra_cluster_similarity_std']:.4f}")
    if clustering_stats['inter_cluster_center_similarity_mean'] is not None:
        print(f"聚类中心间相似度: 均值={clustering_stats['inter_cluster_center_similarity_mean']:.4f}, 方差={clustering_stats['inter_cluster_center_similarity_std']:.4f}")
    
        # 打印每个聚类的相似度统计
    print("各聚类内部相似度统计:")
    for i in range(actual_n_clusters):
        cluster_key = f'cluster_{i}'
        if cluster_key in clustering_stats['cluster_distribution']:
            cluster_stats = clustering_stats['cluster_distribution'][cluster_key]
            if 'intra_similarity_mean' in cluster_stats:
                print(f"  聚类 {i}: 均值={cluster_stats['intra_similarity_mean']:.4f}, 方差={cluster_stats['intra_similarity_std']:.4f}, 样本数={cluster_stats['count']}")
    
    # 打印各聚类内部相似度详情（包含设备分布）
    print("各聚类内部相似度详情（包含设备分布）:")
    for i in range(actual_n_clusters):
        cluster_key = f'cluster_{i}'
        if cluster_key in clustering_stats['cluster_distribution']:
            cluster_stats = clustering_stats['cluster_distribution'][cluster_key]
            if 'intra_similarity_mean' in cluster_stats:
                print(f"  聚类 {i}: 均值={cluster_stats['intra_similarity_mean']:.4f}, 方差={cluster_stats['intra_similarity_std']:.4f}, 样本数={cluster_stats['count']}")
                if 'device_distribution' in cluster_stats:
                    print(f"    设备分布: {cluster_stats['device_distribution']}")
    
    # 绘制聚类中心相似度热力图
    plot_cluster_centers_heatmap(cluster_centers, output_dir, base_name, device_id=None)
    
    # 绘制所有设备点的散点图（按设备分组）
    plot_all_devices_scatter(embeddings_data, device_groups, output_dir, base_name, denormalize_coordinates=True)

def plot_all_devices_scatter(embeddings_data, device_groups, output_dir, base_name, denormalize_coordinates=True):
    """
    绘制所有设备点的散点图，按设备分组，使用真实坐标
    
    Args:
        embeddings_data: embeddings数据字典
        device_groups: 按设备分组的数据
        output_dir: 输出目录
        base_name: 基础文件名
        denormalize_coordinates: 是否对坐标进行反归一化
    """
    print("绘制所有设备点的散点图...")
    
    # 收集所有设备的坐标
    all_coordinates = []
    all_device_ids = []
    
    for device_id, device_data in device_groups.items():
        coordinates = device_data['coordinates']
        all_coordinates.append(coordinates)
        all_device_ids.extend([device_id] * len(coordinates))
    
    # 合并所有坐标
    all_coordinates = np.vstack(all_coordinates)
    
    # 如果需要对坐标进行反归一化
    if denormalize_coordinates:
        norm_params_path = "./output/norm_params.pt"
        if os.path.exists(norm_params_path):
            try:
                norm_params = torch.load(norm_params_path, map_location='cpu')
                pos_range = norm_params['pos_range'].numpy()
                pos_min = norm_params['pos_min'].numpy()
                
                print(f"对坐标进行反归一化:")
                print(f"位置范围: {pos_range}")
                print(f"位置最小值: {pos_min}")
                
                # 反归一化坐标
                all_coordinates = all_coordinates * pos_range + pos_min
                
                print(f"坐标已反归一化")
                
            except Exception as e:
                print(f"加载归一化参数失败: {e}")
                print("使用原始坐标")
        else:
            print("未找到归一化参数文件，使用原始坐标")
    
    # 绘制散点图
    plt.figure(figsize=(15, 10))
    
    # 为每个设备使用不同的颜色
    unique_devices = list(set(all_device_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_devices)))
    
    # 创建legend元素
    legend_elements = []
    
    # 为每个设备绘制散点图
    for i, device_id in enumerate(unique_devices):
        device_mask = np.array([d == device_id for d in all_device_ids])
        device_coords = all_coordinates[device_mask]
        
        # 绘制该设备的所有点
        plt.scatter(device_coords[:, 0], device_coords[:, 1], 
                   c=[colors[i]], alpha=0.7, s=50, marker='o')
        
        # 创建legend元素
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], 
                                        markersize=8, 
                                        label=f'Device {device_id}'))
    
    plt.xlabel('X Coordinate (Denormalized)', fontsize=14)
    plt.ylabel('Y Coordinate (Denormalized)', fontsize=14)
    plt.title(f'All Devices: Data Points Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 保存图片
    scatter_path = os.path.join(output_dir, f"{base_name}_all_devices_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有设备散点图已保存到: {scatter_path}")
    print(f"设备数量: {len(unique_devices)}")
    print(f"总样本数: {len(all_coordinates)}")
    
    # 打印每个设备的统计信息
    print("各设备样本统计:")
    for device_id in unique_devices:
        device_mask = np.array([d == device_id for d in all_device_ids])
        device_count = np.sum(device_mask)
        device_coords = all_coordinates[device_mask]
        print(f"  设备 {device_id}: {device_count} 个样本")
        if len(device_coords) > 0:
            print(f"    坐标范围: X[{device_coords[:, 0].min():.2f}, {device_coords[:, 0].max():.2f}], Y[{device_coords[:, 1].min():.2f}, {device_coords[:, 1].max():.2f}]")

def analyze_embeddings(embeddings_file, output_dir="output/embedding_analysis", denormalize_distances=True, n_clusters=5):
    """
    分析embeddings数据
    
    Args:
        embeddings_file: embeddings文件路径
        output_dir: 输出目录
        denormalize_distances: 是否对距离进行反归一化
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载embeddings数据: {embeddings_file}")
    embeddings_data = load_embeddings_data(embeddings_file)
    
    # 提取embeddings
    embeddings = embeddings_data['embeddings']
    embeddings = embeddings.squeeze(-1)
    phase = embeddings_data.get('phase', 'unknown')
    num_samples = embeddings_data.get('num_samples', len(embeddings))
    embedding_dim = embeddings_data.get('embedding_dim', embeddings.shape[1])
    
    print(f"Embeddings形状: {embeddings.shape}")
    print(f"阶段: {phase}")
    print(f"样本数量: {num_samples}")
    print(f"Embedding维度: {embedding_dim}")
    
    # 计算相似度和距离
    similarities, distances = compute_similarity_and_distance(embeddings)
    
    # 提取配对数据
    sim_values, dist_values = extract_pairs(similarities, distances)
    
    # 如果需要进行距离反归一化
    if denormalize_distances:
        # 尝试加载归一化参数
        norm_params_path = "./output/norm_params.pt"
        if os.path.exists(norm_params_path):
            try:
                norm_params = torch.load(norm_params_path, map_location='cpu')
                pos_range = norm_params['pos_range'].numpy()
                pos_min = norm_params['pos_min'].numpy()
                
                print(f"找到归一化参数:")
                print(f"位置范围: {pos_range}")
                print(f"位置最小值: {pos_min}")
                
                # 反归一化距离（将embedding距离转换为实际物理距离）
                # 这里假设embedding距离与物理距离成正比
                # 可以根据实际情况调整转换公式
                dist_values = dist_values * np.mean(pos_range)
                
                print(f"距离已反归一化，使用缩放因子: {np.mean(pos_range):.4f}")
                
            except Exception as e:
                print(f"加载归一化参数失败: {e}")
                print("使用原始距离值")
        else:
            print("未找到归一化参数文件，使用原始距离值")
    
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(embeddings_file))[0]
    
    # 绘制所有设备合在一起的散点图
    scatter_path = os.path.join(output_dir, f"{base_name}_all_devices_similarity_distance_scatter.png")
    plot_similarity_distance_scatter(sim_values, dist_values, scatter_path, 
                                   title=f"{phase.capitalize()} Embeddings: All Devices Similarity vs Distance")
    
    # 检查embeddings数据中是否包含device_ids
    device_ids = embeddings_data.get('device_ids', None)
    if device_ids is not None:
        print(f"从embeddings数据中加载了 {len(set(device_ids))} 个不同的设备")
        print(f"设备列表: {list(set(device_ids))}")
        
        # 按设备分组数据
        device_groups = {}
        for i, device_id in enumerate(device_ids):
            if device_id not in device_groups:
                device_groups[device_id] = {
                    'embeddings': [],
                    'coordinates': [],
                    'indices': []
                }
            device_groups[device_id]['embeddings'].append(embeddings[i])
            device_groups[device_id]['coordinates'].append(embeddings_data['coordinates'][i])
            device_groups[device_id]['indices'].append(i)
        
        # 转换为numpy数组
        for device_id in device_groups:
            device_groups[device_id]['embeddings'] = np.array(device_groups[device_id]['embeddings'])
            device_groups[device_id]['coordinates'] = np.array(device_groups[device_id]['coordinates'])
        
        print(f"按设备分组完成，共 {len(device_groups)} 个设备")
        
        # 为每个设备绘制单独的散点图
        plot_device_specific_scatter(embeddings_data, device_groups, output_dir, base_name)
        
        # 为每个设备进行聚类并绘制散点图
        plot_device_clustering_scatter(embeddings_data, device_groups, output_dir, base_name, n_clusters)
        
        # 将所有设备的点放在一起进行聚类
        plot_all_devices_clustering_scatter(embeddings_data, device_groups, output_dir, base_name, n_clusters, denormalize_coordinates=True)
    
    # 保存统计信息
    stats = {
        'phase': phase,
        'num_samples': num_samples,
        'embedding_dim': embedding_dim,
        'similarity_mean': float(np.mean(sim_values)),
        'similarity_std': float(np.std(sim_values)),
        'distance_mean': float(np.mean(dist_values)),
        'distance_std': float(np.std(dist_values)),
        'similarity_distance_correlation': float(np.corrcoef(sim_values, dist_values)[0, 1]),
        'denormalized': denormalize_distances
    }
    
    stats_path = os.path.join(output_dir, f"{base_name}_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息已保存到: {stats_path}")
    print(f"相似度均值: {stats['similarity_mean']:.4f}")
    print(f"相似度标准差: {stats['similarity_std']:.4f}")
    print(f"距离均值: {stats['distance_mean']:.4f}")
    print(f"距离标准差: {stats['distance_std']:.4f}")
    print(f"相似度-距离相关系数: {stats['similarity_distance_correlation']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='分析embeddings数据')
    parser.add_argument('--embeddings_file', type=str, required=True,
                       help='embeddings文件路径 (.pt 或 .npz)')
    parser.add_argument('--output_dir', type=str, default='output/embedding_analysis',
                       help='输出目录')
    parser.add_argument('--no_denormalize', action='store_true',
                       help='不对距离进行反归一化')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='聚类数量 (默认: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings_file):
        print(f"错误: 文件不存在: {args.embeddings_file}")
        return
    
    analyze_embeddings(args.embeddings_file, args.output_dir, 
                      denormalize_distances=not args.no_denormalize,
                      n_clusters=args.n_clusters)

if __name__ == "__main__":
    main() 