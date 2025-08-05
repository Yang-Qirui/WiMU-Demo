# data_process.py 输出目录修改总结

## 修改目标
将`data_process.py`中的硬编码输出路径修改为根据输入数据路径自动创建的专用输出目录，避免不同数据集的处理结果相互覆盖。

## 主要修改内容

### 1. 添加全局变量
```python
# Global variable to store the current output directory
CURRENT_OUTPUT_DIR = "output/data_process"
```

### 2. 主函数修改
在`if __name__ == "__main__":`部分：

#### 原始代码结构：
```python
# 固定输出路径
sampling_counter_path = 'output/data_process/sampling_counter.json'
# 固定数据路径
data_path = "data/JD_large_scale_test_low"
```

#### 修改后结构：
```python
# 从数据路径提取数据集名称
data_path = "data/JD_large_scale_test_low"
data_name = os.path.basename(data_path.rstrip('/'))
if not data_name:
    data_name = "default_dataset"

# 创建数据特定的输出目录
output_base_dir = f"output_{data_name}"
output_data_process_dir = os.path.join(output_base_dir, "data_process")

# 设置全局输出目录
global CURRENT_OUTPUT_DIR
CURRENT_OUTPUT_DIR = output_data_process_dir

# 使用数据特定的路径
sampling_counter_path = os.path.join(output_data_process_dir, 'sampling_counter.json')
```

### 3. 函数参数修改

#### `process_all_new_format`函数：
```python
# 添加output_dir参数
def process_all_new_format(data_path, output_dir=None):
    if output_dir is None:
        output_dir = "output/data_process"
```

#### `record_distinct_aps_unlabeled_only`函数：
```python
# 添加output_dir参数
def record_distinct_aps_unlabeled_only(unlabeled_dir, valid_aps, output_dir=None):
    if output_dir is None:
        output_dir = "output/data_process"
```

#### `create_ap_unions`函数：
```python
# 添加output_dir参数
def create_ap_unions(merged_aps, output_dir=None):
    if output_dir is None:
        output_dir = "output/data_process"
```

#### `process_wifi`函数：
```python
# 添加sampling_counter_path参数
def process_wifi(wifi_file_path, sampling_counter_path='output/data_process/sampling_counter.json'):
```

### 4. 路径修改

#### 使用全局变量的文件：
- `read_waypoint`函数中的`ap_unions.json`路径
- 所有保存文件的路径都改为使用传入的`output_dir`参数

#### 规范化参数保存：
```python
# 原来
torch.save(norm_params, "output/norm_params.pt")

# 修改后
torch.save(norm_params, os.path.join(output_base_dir, "norm_params.pt"))
```

## 输出目录结构

### 修改前（所有数据集共用）：
```
output/
├── data_process/
│   ├── sampling_counter.json
│   ├── ap_unions.json
│   ├── distinct_aps.json
│   ├── merged_aps.json
│   ├── ap_dist_count.npy
│   ├── ap_dist.npy
│   ├── ap_dist_sq.npy
│   ├── ap_mean.npy
│   └── ap_std.npy
└── norm_params.pt
```

### 修改后（每个数据集独立）：
```
output_JD_large_scale_test_low/
├── data_process/
│   ├── sampling_counter.json
│   ├── ap_unions.json
│   ├── distinct_aps.json
│   ├── merged_aps.json
│   ├── ap_dist_count.npy
│   ├── ap_dist.npy
│   ├── ap_dist_sq.npy
│   ├── ap_mean.npy
│   └── ap_std.npy
└── norm_params.pt

output_hilton_dataset/
├── data_process/
│   └── (同样的文件结构)
└── norm_params.pt
```

## 使用方法

### 修改数据路径
在`data_process.py`的主函数中修改：
```python
# 修改这一行来处理不同的数据集
data_path = "data/your_dataset_name"
```

### 输出目录命名规则
- 输出目录格式：`output_{数据集名称}`
- 数据集名称从数据路径的最后一个目录名提取
- 如果无法提取，使用`default_dataset`

### 示例
```python
# 数据路径 -> 输出目录
"data/JD_large_scale_test_low" -> "output_JD_large_scale_test_low"
"data/hilton_meeting" -> "output_hilton_meeting"
"data/experiment_20241201" -> "output_experiment_20241201"
```

## 优势

1. **数据隔离**：不同数据集的处理结果分别存储
2. **避免覆盖**：新数据集处理不会覆盖之前的结果
3. **清晰标识**：输出目录名直接反映数据来源
4. **向后兼容**：保持现有代码逻辑不变
5. **易于管理**：便于比较和管理不同数据集的结果

## 注意事项

1. 确保有足够的磁盘空间存储多个数据集的结果
2. 清理旧结果时注意区分不同数据集
3. 如果需要处理新数据集，只需修改`data_path`变量即可
4. 全局变量`CURRENT_OUTPUT_DIR`确保所有函数使用一致的输出路径 