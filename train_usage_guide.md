# train_two_stage.py 新文件结构使用指南

## 概述

现在`train_two_stage.py`已经更新为支持从数据特定的文件夹中加载数据，这与`data_process.py`的新输出结构保持一致。

## 新的数据加载逻辑

### Pre-training阶段数据加载
```python
# 原来的固定路径
wifi_inputs = np.load("./data/pre_training/wifi_inputs.npy")
distance_labels = np.load("./data/pre_training/distance_labels.npy")
id_mask = np.load("./data/pre_training/id_mask.npy")

# 现在的数据特定路径
pre_training_dir = f"./{args.save_dir}/pre_training"
wifi_inputs = np.load(f"{pre_training_dir}/wifi_inputs.npy")
distance_labels = np.load(f"{pre_training_dir}/distance_labels.npy")
id_mask = np.load(f"{pre_training_dir}/id_mask.npy")
```

### Fine-tuning阶段数据加载
```python
# 原来的固定路径
train_path = "./data/train_json"
test_path = "./data/test_json"

# 现在的数据特定路径
train_path = f"./{args.save_dir}/train_json"
test_path = f"./{args.save_dir}/test_json"
```

## 使用方法

### 方法1：使用默认的数据名称检测
```bash
# 步骤1：运行data_process.py生成数据
python data_process.py

# 步骤2：运行预训练（会自动检测数据名称）
python train_two_stage.py --pre_train --data_name JD_langfang_large_scale_low_left

# 步骤3：运行fine-tuning（使用相同的数据名称）
python train_two_stage.py --fine_tune --data_name JD_langfang_large_scale_low_left
```

### 方法2：手动指定数据集名称
```bash
# 直接指定数据集名称和保存目录
python train_two_stage.py --pre_train --data_name custom_dataset --save_dir output_custom_dataset

python train_two_stage.py --fine_tune --data_name custom_dataset --save_dir output_custom_dataset
```

### 方法3：使用完整的训练流程
```bash
# 完整流程：预训练 + fine-tuning
python train_two_stage.py --pre_train --fine_tune --data_name JD_langfang_large_scale_low_left
```

## 文件结构对应关系

### data_process.py输出结构
```
output_JD_langfang_large_scale_low_left/
├── data_process/         # 图数据
├── train_json/           # 训练数据
├── test_json/            # 测试数据
├── pre_training/         # 预训练数据
└── norm_params.pt        # 归一化参数
```

### train_two_stage.py使用路径
```python
args.save_dir = "output_JD_langfang_large_scale_low_left"

# Pre-training数据路径
pre_training_dir = f"./{args.save_dir}/pre_training"
# -> "./output_JD_langfang_large_scale_low_left/pre_training"

# Fine-tuning数据路径
train_path = f"./{args.save_dir}/train_json"
test_path = f"./{args.save_dir}/test_json"
# -> "./output_JD_langfang_large_scale_low_left/train_json"
# -> "./output_JD_langfang_large_scale_low_left/test_json"

# 图数据路径
graph_dataset = torch.load(f"./{args.save_dir}/graph_dataset.pt")
# -> "./output_JD_langfang_large_scale_low_left/graph_dataset.pt"
```

## 数据处理 + 训练完整流程

### 步骤1：修改数据路径
在`data_process.py`中修改：
```python
data_path = "data/your_dataset_name"
```

### 步骤2：运行数据处理
```bash
python data_process.py
```
这会创建：`output_your_dataset_name/`目录及其所有子文件

### 步骤3：运行预训练
```bash
python train_two_stage.py --pre_train --data_name your_dataset_name
```

### 步骤4：运行fine-tuning
```bash
python train_two_stage.py --fine_tune --data_name your_dataset_name
```

### 步骤5：生成embeddings和可视化
```bash
python visualize_embeddings.py --data_name your_dataset_name
```

## 多数据集并行处理

现在可以同时处理多个数据集而不会相互干扰：

```bash
# 处理数据集1
# 修改data_process.py: data_path = "data/dataset1"
python data_process.py
python train_two_stage.py --pre_train --fine_tune --data_name dataset1

# 处理数据集2
# 修改data_process.py: data_path = "data/dataset2"  
python data_process.py
python train_two_stage.py --pre_train --fine_tune --data_name dataset2

# 结果分别保存在：
# output_dataset1/
# output_dataset2/
```

## 重要注意事项

1. **数据名称一致性**：确保`data_process.py`生成的输出目录名与`train_two_stage.py`中的`--data_name`参数一致

2. **目录结构**：确保运行`train_two_stage.py`之前已经运行了相应的`data_process.py`生成必要的数据文件

3. **路径检查**：如果遇到文件未找到错误，检查：
   - `args.save_dir`是否正确设置
   - 相应的数据文件是否存在
   - 数据集名称是否匹配

4. **向后兼容**：如果仍有旧格式的数据，可以通过符号链接或复制到新结构中使用

## 故障排除

### 问题1：找不到预训练数据文件
```
FileNotFoundError: ./output_dataset_name/pre_training/wifi_inputs.npy
```
**解决方案**：确保先运行`data_process.py`生成预训练数据

### 问题2：找不到训练/测试JSON文件
```
FileNotFoundError: ./output_dataset_name/train_json/
```
**解决方案**：确保`data_process.py`成功处理了labeled数据并生成了train/test split

### 问题3：数据名称不匹配
```
数据目录不存在或为空
```
**解决方案**：检查`--data_name`参数是否与`data_process.py`生成的目录名匹配

### 问题4：图数据文件缺失
```
FileNotFoundError: ./output_dataset_name/graph_dataset.pt
```
**解决方案**：确保先运行预训练阶段生成图数据，再运行fine-tuning 