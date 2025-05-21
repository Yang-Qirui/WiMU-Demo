import matplotlib.pyplot as plt
import numpy as np

# 数据配置
models = ['WiMu', 'GraFin', 'MYRCJ', 'WePos']
rp_counts = ['25', '50', '100']
colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']

# 从表格中提取中等规模建筑的数据
mae_data = {
    'WiMu':   [5.293, 3.875, 3.158],
    'GraFin': [8.075, 5.901, 4.448],
    'MYRCJ':  [13.324, 8.657, 4.387],
    'WePos':  [30.352, 29.294, 25.539]
}
# Calculate performance boost percentage for WiMu compared to others
boost_data = {}
for model in models[1:]:  # Skip WiMu itself
    boost_percentages = []
    for i in range(len(rp_counts)):
        wimu_error = mae_data['WiMu'][i]
        other_error = mae_data[model][i]
        boost = ((other_error - wimu_error) / other_error) * 100
        boost_percentages.append(boost)
    boost_data[model] = boost_percentages

# Print performance boost results
print("\nWiMu Performance Boost:")
print("-" * 50)
for model, boosts in boost_data.items():
    print(f"\nCompared to {model}:")
    for i, boost in enumerate(boosts):
        print(f"  {rp_counts[i]} RPs: {boost:.1f}% better")


# 创建画布
plt.figure(figsize=(10, 6), dpi=100)
ax = plt.gca()

# 柱状图参数设置
bar_width = 0.2
x = np.arange(len(rp_counts))  # x轴刻度位置

# 绘制每个模型的柱状图
for i, (model, values) in enumerate(mae_data.items()):
    offset = (i - 1.5) * bar_width  # 居中排列
    bars = ax.bar(x + offset, values, bar_width, 
                 label=model, color=colors[i],
                 edgecolor='white', linewidth=1.2)
    
    # 在柱子上方添加数值标签
    ax.bar_label(bars, padding=3, fmt='%.1f', fontsize=16)

# 图表装饰
ax.set_ylabel('Localization Error (m)', fontsize=16)
ax.set_xlabel('Number of Labeled Reference Points (RPs)', fontsize=16)
# ax.set_title('MAE Comparison for Larg Buildings (> 10,000-50,000 m²)', 
#             fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(rp_counts)
ax.set_ylim(0, 35)  # 基于最大误差值设置y轴范围
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=True, fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
# 显示图表
plt.tight_layout()
plt.savefig('mae_comparison.png')