import torch_geometric
import torch
from torch_geometric.data.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_dataset = torch.load("output/graph_dataset.pt", weights_only=False)
graph_dataset = graph_dataset.to(device)

graph_dataset.x = graph_dataset.x.to(device)

def solve_A_for_Ax_eq_0(x_tensor: torch.Tensor):
    """
    求解 Ax = 0 中的矩阵 A。
    
    参数:
    x_tensor (torch.Tensor): 一个形状为 (N, D) 的张量，
                             其中 N 是向量的数量，D 是每个向量的维度。
                             我们假设每个 x 向量是 tensor 的一行。
    
    返回:
    torch.Tensor: 一个矩阵 A，它的行构成了 x_tensor 行空间的零空间的一组基。
                  因此，对于 x_tensor 中的每一行 x，都有 A @ x.T = 0。
                  如果不存在非平凡解，则返回一个空矩阵。
    """
    if x_tensor.ndim != 2:
        raise ValueError("输入张量 x_tensor 必须是二维的 (N, D)。")

    N, D = x_tensor.shape
    print(f"输入张量 X 的形状: {N} 个 {D} 维向量")

    # 使用 SVD 分解 X = UΣVᵀ
    # 注意：在 PyTorch 中，torch.linalg.svd 返回 U, S, Vh，其中 Vh 就是 V.T
    try:
        U, S, Vh = torch.linalg.svd(x_tensor, full_matrices=False)
    except torch.linalg.LinAlgError as e:
        print(f"SVD 计算失败: {e}")
        return torch.empty((0, D), device=x_tensor.device, dtype=x_tensor.dtype)

    print(f"奇异值 (S): \n{S}")

    # 确定哪些奇异值可以被认为是“零”
    # 通常设置一个很小的阈值，而不是直接与0比较，以处理浮点数精度问题
    # 阈值可以根据奇异值的量级来定，例如 max(S) * 1e-6
    if S.numel() > 0:
        tolerance = S.max() * 1e-6
    else:
        tolerance = 1e-6

    # 找到奇异值小于阈值的索引
    zero_singular_value_indices = torch.where(S < tolerance)[0]

    if len(zero_singular_value_indices) == 0:
        print("\n没有发现零奇异值（在容忍度范围内）。")
        print("这意味着 x 向量可能是线性无关的，不存在非平凡解 A。")
        # 返回一个形状为 (0, D) 的空矩阵
        return torch.empty((0, D), device=x_tensor.device, dtype=x_tensor.dtype)
    else:
        print(f"\n发现在索引 {zero_singular_value_indices.tolist()} 处的奇异值足够小。")
        
        # 矩阵 A 的行由 Vh 中对应于零奇异值的行构成
        A = Vh[zero_singular_value_indices, :]
        
        print(f"解出的矩阵 A 的形状: {A.shape}")
        return A

solve_A_for_Ax_eq_0(graph_dataset.x)


