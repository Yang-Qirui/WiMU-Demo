import os
from pathlib import Path

def count_subdirectories(directory):
    """
    统计目录中的子文件夹数量
    :param directory: 目录路径
    :return: 子文件夹数量
    """
    count = 0
    for root, dirs, files in os.walk(directory):
        # 只统计直接子文件夹
        if root == directory:
            count = len(dirs)
            break
    return count

def main():
    base_dir = "data/JD_warehouse_data"
    directories = ["train", "test", "unlabeled"]
    
    print("子文件夹统计结果：")
    print("-" * 30)
    
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            dir_count = count_subdirectories(dir_path)
            print(f"{dir_name} 目录中的子文件夹数量: {dir_count}")
        else:
            print(f"{dir_name} 目录不存在")
    
    print("\nAP统计结果：")
    print("-" * 30)
    
    # 统计distinct_aps
    distinct_aps_path = os.path.join(base_dir, "distinct_aps")
    if os.path.exists(distinct_aps_path):
        distinct_count = count_subdirectories(distinct_aps_path)
        print(f"distinct_aps 中的AP数量: {distinct_count}")
    else:
        print("distinct_aps 目录不存在")
    
    # 统计merged_aps
    merged_aps_path = os.path.join(base_dir, "merged_aps")
    if os.path.exists(merged_aps_path):
        merged_count = count_subdirectories(merged_aps_path)
        print(f"merged_aps 中的AP数量: {merged_count}")
    else:
        print("merged_aps 目录不存在")

if __name__ == "__main__":
    main() 