import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# 读取测试结果
with open('output/fine_tuned_test_result.json', 'r') as f:
    data = json.load(f)

# 使用字典存储每个位置点的所有误差值
position_errors = defaultdict(list)
for key, value in data.items():
    if key not in ['x_err_std', 'y_err_std']:  # 跳过标准差数据
        # 使用正则表达式提取坐标
        match = re.search(r'\[([-\d.]+),\s*([-\d.]+)\]', key)
        if match:
            x, y = map(float, match.groups())
            position_errors[(x, y)].append(value)

# 计算每个位置点的均值和标准差
coordinates = []
mean_errors = []
std_errors = []
for coords, errors in position_errors.items():
    coordinates.append(coords)
    mean_errors.append(np.mean(errors))
    std_errors.append(np.std(errors))

# 转换为numpy数组
coordinates = np.array(coordinates)
mean_errors = np.array(mean_errors)
std_errors = np.array(std_errors)

# 创建热力图
plt.figure(figsize=(12, 10))
scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=mean_errors, 
                     cmap='viridis_r', s=100)  # 使用viridis_r颜色映射，颜色越深表示平均误差越大
plt.colorbar(scatter, label='Mean Localization Error (m)')

# 在每个点旁边添加均值和标准差标注
for i, (x, y) in enumerate(coordinates):
    plt.annotate(f'{mean_errors[i]:.2f}±{std_errors[i]:.2f}', 
                (x, y), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

# 添加标题和标签
plt.title('Localization Error Heatmap\n(Color: Mean Error, Text: Mean±Std)')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')

# 保存图像
plt.savefig('output/error_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Heatmap has been saved to output/error_heatmap.png")