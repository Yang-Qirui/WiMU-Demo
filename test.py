# import numpy as np
# from scipy.optimize import minimize

# def simple_optimizer(original_points, closure_weight=1e4):
#     """
#     简化版轨迹优化器（正确实现梯度计算）
#     :param original_points: 原始轨迹点列表，起点必须是(0,0)
#     :param closure_weight: 闭合约束权重
#     :return: 优化后的轨迹点列表
#     """
#     # 转换为numpy数组
#     original = np.array(original_points, dtype=np.float64)
#     n_points = original.shape[0]
    
#     # 初始化优化变量（排除起点）
#     x0 = original[1:].flatten().copy()
    
#     def compute_loss(x):
#         """损失函数计算"""
#         current_points = np.vstack([(0,0), x.reshape(-1,2)])
        
#         # 闭合误差（终点到起点的距离平方）
#         closure_error = np.sum(current_points[-1]**2)
        
#         # 形状保持误差（各点与原始位置的偏差）
#         shape_error = np.sum((current_points[1:] - original[1:])**2)
        
#         # 平滑误差（加速度最小化）
#         smooth_error = 0.0
#         if n_points >= 3:
#             # 计算二阶差分（加速度）
#             acc = np.diff(np.diff(current_points, axis=0), axis=0)
#             smooth_error = np.sum(acc**2)
        
#         return closure_weight*closure_error + shape_error + 0.1*smooth_error
    
#     def compute_grad(x):
#         """梯度计算（确保边界安全）"""
#         grad = np.zeros_like(x)
#         current_points = np.vstack([(0,0), x.reshape(-1,2)])
#         n = current_points.shape[0]
        
#         # 闭合项梯度（仅影响最后一个点）
#         if n >= 2:
#             grad[-2] = 2 * closure_weight * current_points[-1, 0]  # x分量
#             grad[-1] = 2 * closure_weight * current_points[-1, 1]  # y分量
        
#         # 形状保持梯度（所有中间点）
#         grad += 2 * (x - original[1:].flatten())
        
#         # 平滑项梯度（安全处理边界）
#         if n >= 3:
#             # 计算速度（一阶差分）和加速度（二阶差分）
#             vel = np.diff(current_points, axis=0)
#             acc = np.diff(vel, axis=0)
            
#             # 每个加速度项影响三个点
#             for i in range(acc.shape[0]):
#                 # 加速度项对应的三个点索引
#                 p1 = i    # 对应x[i+1]
#                 p2 = i+1  # 对应x[i+2]
#                 p3 = i+2  # 对应x[i+3]
                
#                 # 梯度分量系数
#                 grad_coeff = 0.2  # 0.1*2 (因为对平方项求导)
                
#                 # 更新每个点的梯度分量
#                 # 影响点p1的x[i+1]
#                 if p1 < x.shape[0]//2:
#                     grad[2*p1:2*p1+2] += grad_coeff * (2*acc[i])
                
#                 # 影响点p2的x[i+2]
#                 if p2 < x.shape[0]//2:
#                     grad[2*p2:2*p2+2] += grad_coeff * (-4*acc[i])
                
#                 # 影响点p3的x[i+3]
#                 if p3 < x.shape[0]//2:
#                     grad[2*p3:2*p3+2] += grad_coeff * (2*acc[i])
        
#         return grad
    
#     # 执行优化
#     res = minimize(
#         compute_loss,
#         x0,
#         jac=compute_grad,
#         method='L-BFGS-B',
#         options={
#             'maxiter': 1000,
#             'ftol': 1e-8,
#             'disp': True
#         }
#     )
    
#     if not res.success:
#         raise RuntimeError(f"优化失败: {res.message}")
    
#     # 重构轨迹点
#     optimized = np.vstack([(0,0), res.x.reshape(-1,2)])
#     return optimized.tolist()

# # 测试用例
# if __name__ == "__main__":
#     # 测试最小轨迹（起点 + 2个中间点）
#     test_case = [
#         [0.0, 0.0],
#         [1.0, 0.1],  # 中间点1
#         [0.9, 0.2]   # 终点（应闭合到0,0）
#     ]
    
#     try:
#         result = simple_optimizer(test_case)
#         print("优化结果:")
#         for i, (x, y) in enumerate(result):
#             print(f"点{i}: {x:.6f}, {y:.6f}")
#         print(f"闭合误差: {np.linalg.norm(result[-1]):.6f}")
#     except Exception as e:
#         print(f"错误发生: {str(e)}")
from utils import LDPL

a = [-75, -82, -75, -63, -80, -56]
b = [-72, -71, -76, -65, -75, -50]

_a = [1 / (LDPL(x)) for x in a]
_b = [1 / (LDPL(x)) for x in b]
_a /= sum(_a)
_b /= sum(_b)

print(_a)
print(_b)