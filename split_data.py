import os
import shutil
import random
from pathlib import Path

def split_data(train_dir, test_dir, test_ratio=0.2):
    """
    将训练数据随机分割成训练集和测试集
    :param train_dir: 原始训练数据目录
    :param test_dir: 测试数据目录
    :param test_ratio: 测试集比例
    """
    # 确保测试目录存在
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有训练数据目录
    train_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # 计算需要移动到测试集的数量
    n_test = int(len(train_dirs) * test_ratio)
    
    # 随机选择要移动的目录
    test_dirs = random.sample(train_dirs, n_test)
    
    # 移动选中的目录到测试集
    for dir_name in test_dirs:
        src = os.path.join(train_dir, dir_name)
        dst = os.path.join(test_dir, dir_name)
        shutil.move(src, dst)
        print(f"Moved {dir_name} to test set")

if __name__ == "__main__":
    # 设置路径
    base_dir = "data/JD_simple_warehouse_data"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # 执行分割
    split_data(train_dir, test_dir) 