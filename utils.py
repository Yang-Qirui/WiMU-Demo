import colorlog
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

log_colors = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}    
AP_FILTER_THRESHOLD = 10

def init_logger(logger_name, log_name="app.log"):
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'log'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors=log_colors
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def LDPL(rssi, band='5G', r0_5g=32, r0_2g=38, n_5g=2.2, n_2g=2):
    """
    Log-Distance Path Loss model for different frequency bands
    
    Args:
        rssi: Received signal strength in dBm
        band: Frequency band, '5G' or '2.4G'
        r0_5g: Reference distance for 5GHz in dBm
        r0_2g: Reference distance for 2.4GHz in dBm
        n_5g: Path loss exponent for 5GHz
        n_2g: Path loss exponent for 2.4GHz
    """
    if band == '5G':
        return np.power(10, (-rssi - r0_5g) / (10 * n_5g))
    else:  # 2.4G
        return np.power(10, (-rssi - r0_2g) / (10 * n_2g))

def longest_common_substring(str1:str, str2:str):
    str1 = str1.replace(":", "")
    str2 = str2.replace(":", "")
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest_length = 0
    end_index_str1 = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_length:
                    longest_length = dp[i][j]
                    end_index_str1 = i
                    
    longest_common_substr = str1[end_index_str1 - longest_length:end_index_str1]
    return longest_common_substr

def heatmap(data, savepath):
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.title('Heatmap of 2D Array')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.savefig(savepath)
    

def prune_adjacency_topk_min(A_orig, k=5):
    """
    Prune (or sparsify) the adjacency/similarity matrix A_orig by:
      1) Keeping up to k largest off-diagonal entries per row (top-k).
      2) Enforcing symmetry by taking max(A, A^T).
      3) Ensuring every node has at least 1 edge by restoring the single
         largest edge (from the original matrix) if a node is isolated.

    Args:
        A_orig (np.ndarray): Original NxN adjacency or similarity matrix (non-negative).
        k (int): Keep up to k largest entries per row (excluding the diagonal).

    Returns:
        np.ndarray: The pruned, symmetric adjacency matrix with no isolated nodes,
                    unless a node truly had no non-zero edges in A_orig.
    """
    A_orig = A_orig.copy()
    N = A_orig.shape[0]

    # 1) Zero out diagonal so diagonal doesn't get picked in top-k.
    np.fill_diagonal(A_orig, 0.0)

    # We'll build A_pruned from scratch
    A_pruned = np.zeros_like(A_orig)

    # 2) For each row, keep top-k largest entries
    for i in range(N):
        row = A_orig[i, :]
        # Find indices of top-k largest entries
        # argsort sorts ascending => reverse to get descending
        top_k_indices = np.argsort(row)[::-1][:k]
        # Keep those entries in A_pruned
        A_pruned[i, top_k_indices] = row[top_k_indices]

    # 3) Enforce symmetry: if i->j is kept, j->i is also kept
    #    We'll take the max so that if one side is bigger, we keep that.
    A_pruned = np.maximum(A_pruned, A_pruned.T)

    # 4) Ensure no node is completely isolated.
    #    For each node i, if the entire row is zero after symmetry, 
    #    we restore the single largest edge from the original adjacency A_orig.
    #    (If the original adjacency truly has no non-zero edges for i,
    #     then that node remains isolated.)
    for i in range(N):
        if np.all(A_pruned[i, :] == 0):
            # Node i is isolated; restore the single largest edge from A_orig
            # (which is guaranteed to be zero if there are truly no edges for i).
            # Let's find the single largest off-diagonal entry in row i:
            row = A_orig[i, :]
            j = np.argmax(row)  # index of max
            max_val = row[j]
            if max_val > 0:
                # Restore i->j and j->i
                A_pruned[i, j] = max_val
                A_pruned[j, i] = max_val

    return A_pruned

def plot_pretrain_losses(train_recon_losses, train_dist_losses, val_recon_losses, val_dist_losses, train_contrast_losses, val_contrast_losses):
    """
    Plot the training and validation losses during pre-training
    """
    plt.figure(figsize=(12, 5))
    
    # Plot reconstruction losses
    plt.subplot(1, 3, 1)
    plt.plot(train_recon_losses, label='Train Reconstruction Loss')
    plt.plot(val_recon_losses, label='Validation Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss vs Epoch')
    plt.legend()
    
    # Plot distance losses
    plt.subplot(1, 3, 2)
    plt.plot(train_dist_losses, label='Train Distance Loss')
    plt.plot(val_dist_losses, label='Validation Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Distance Loss vs Epoch')
    plt.legend()
    
    # Plot contrast losses
    plt.subplot(1, 3, 3)
    plt.plot(train_contrast_losses, label='Train Contrast Loss')
    plt.plot(val_contrast_losses, label='Validation Contrast Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrast Loss vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/pre_train_plots/training_curves.png')
    plt.close()
        
def plot_adjacency_matrices(original_A, recon_A):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_A.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Original Adjacency Matrix')
    
    plt.subplot(1, 2, 2)
    plt.imshow(recon_A.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Reconstructed Adjacency Matrix')
    
    plt.tight_layout()
    plt.savefig('output/pre_train_adjacency/adjacency_comparison.png')
    plt.close()

def plot_fine_tune_losses(train_losses, test_errors):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_errors, label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Test Error vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/fine_tune_plots/training_curves.png')
    plt.close()
    
def params_to_traj(params):
    """将参数向量转换为轨迹坐标"""
    traj = params.reshape(-1, 2)
    return np.vstack([traj, traj[0]])  # 闭环处理

def residuals(params, pdr_deltas):
    """
    计算残差：
    - 相邻点之间的位移与PDR测量值的差异
    - 闭环约束
    """
    traj = params_to_traj(params)
    res = []
    
    # 相邻节点位移残差
    for i in range(len(traj)-1):
        dx_meas = pdr_deltas[i,0]
        dy_meas = pdr_deltas[i,1]
        dx_actual = traj[i+1,0] - traj[i,0]
        dy_actual = traj[i+1,1] - traj[i,1]
        res.extend([dx_actual - dx_meas, dy_actual - dy_meas])
    
    return np.array(res)

def optimize_trajectory(points, closure_weight=1e3, smooth_weight=0.1):
    """
    轨迹优化核心函数
    :param points: 原始轨迹点列表，起点必须是(0,0)
    :param closure_weight: 起点-终点闭合权重（建议1e3-1e5）
    :param smooth_weight: 轨迹平滑度权重（建议0.1-1.0）
    :return: 优化后的轨迹点
    """
    # 转换输入为NumPy数组便于计算
    original = np.array(points)
    n_points = original.shape[0]
    
    # 将待优化点展开为一维数组（排除起点）
    x0 = original[1:].flatten()  # 形状: [2*(n_points-1)]
    
    def loss_function(x):
        """多目标损失函数"""
        # 重构完整轨迹（包含起点）
        optimized = np.vstack([(0,0), x.reshape(-1,2)])
        
        # 闭合误差（起点-终点距离）
        closure_error = np.sum(optimized[-1]**2)
        
        # 形状保持误差（相对于原始轨迹）
        shape_error = np.sum((optimized[1:] - original[1:])**2)
        
        # 平滑度约束（相邻点加速度最小化）
        smooth_error = 0.0
        for i in range(1, n_points-1):
            delta1 = optimized[i] - optimized[i-1]
            delta2 = optimized[i+1] - optimized[i]
            smooth_error += np.sum((delta2 - delta1)**2)
            
        return (
            closure_weight * closure_error +
            shape_error +
            smooth_weight * smooth_error
        )
    
    # 设置优化器参数
    result = minimize(
        loss_function,
        x0,
        method='L-BFGS-B',
        options={
            'maxiter': 1000,
            'ftol': 1e-8,
            'disp': True
        }
    )
    
    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")
    
    # 重构最终轨迹
    final_points = np.vstack([(0,0), result.x.reshape(-1,2)])
    return final_points


def calculate_step_errors(traj, ground_truth):
    """
    计算轨迹相邻点间距与真实值的误差
    
    Parameters:
    -----------
    traj : numpy.ndarray
        待评估的轨迹
    ground_truth : numpy.ndarray
        真实轨迹
    
    Returns:
    --------
    float
        平均步长误差
    """
    traj_deltas = np.diff(traj, axis=0)
    gt_deltas = np.diff(ground_truth, axis=0)
    step_errors = np.linalg.norm(traj_deltas - gt_deltas, axis=1)
    return np.mean(step_errors)
    
    

