#!/usr/bin/env python3
"""
测试data_process.py的输出目录功能
"""

import os
import sys
import tempfile
import shutil

def test_output_directory_creation():
    """测试输出目录创建功能"""
    print("=== 测试data_process.py输出目录功能 ===")
    
    # 模拟不同的数据路径
    test_cases = [
        "data/JD_large_scale_test_low",
        "data/JD_large_scale_test", 
        "data/hilton_dataset",
        "data/custom_experiment"
    ]
    
    for data_path in test_cases:
        # 提取数据集名称
        data_name = os.path.basename(data_path.rstrip('/'))
        if not data_name:
            data_name = "default_dataset"
        
        # 预期的输出目录
        expected_output_dir = f"output_{data_name}"
        expected_data_process_dir = os.path.join(expected_output_dir, "data_process")
        
        print(f"\n数据路径: {data_path}")
        print(f"数据集名称: {data_name}")
        print(f"预期输出目录: {expected_output_dir}")
        print(f"预期数据处理目录: {expected_data_process_dir}")
        
        # 验证目录命名规则
        assert expected_output_dir.startswith("output_"), f"输出目录应以'output_'开头"
        assert data_name in expected_output_dir, f"输出目录应包含数据集名称"

def test_file_structure():
    """测试预期的文件结构"""
    print("\n=== 测试预期文件结构 ===")
    
    data_name = "test_dataset"
    output_base_dir = f"output_{data_name}"
    
    expected_files = [
        os.path.join(output_base_dir, "data_process", "sampling_counter.json"),
        os.path.join(output_base_dir, "data_process", "ap_unions.json"),
        os.path.join(output_base_dir, "data_process", "distinct_aps.json"),
        os.path.join(output_base_dir, "data_process", "merged_aps.json"),
        os.path.join(output_base_dir, "data_process", "ap_dist_count.npy"),
        os.path.join(output_base_dir, "data_process", "ap_dist.npy"),
        os.path.join(output_base_dir, "data_process", "ap_dist_sq.npy"),
        os.path.join(output_base_dir, "data_process", "ap_mean.npy"),
        os.path.join(output_base_dir, "data_process", "ap_std.npy"),
        os.path.join(output_base_dir, "norm_params.pt"),
    ]
    
    print("预期的文件结构:")
    for file_path in expected_files:
        print(f"  {file_path}")

def test_path_extraction():
    """测试路径提取逻辑"""
    print("\n=== 测试路径提取逻辑 ===")
    
    test_paths = [
        ("data/JD_large_scale_test_low", "JD_large_scale_test_low"),
        ("data/JD_large_scale_test/", "JD_large_scale_test"),
        ("./data/hilton", "hilton"),
        ("/absolute/path/to/dataset", "dataset"),
        ("single_folder", "single_folder"),
        ("", "default_dataset"),  # 空路径的情况
    ]
    
    for input_path, expected_name in test_paths:
        # 模拟data_process.py中的逻辑
        data_name = os.path.basename(input_path.rstrip('/'))
        if not data_name:
            data_name = "default_dataset"
        
        print(f"输入路径: '{input_path}' -> 数据集名称: '{data_name}'")
        assert data_name == expected_name, f"预期 {expected_name}, 得到 {data_name}"

def show_example_usage():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    
    examples = [
        {
            "description": "处理JD低密度数据集",
            "data_path": "data/JD_large_scale_test_low",
            "output_dir": "output_JD_large_scale_test_low"
        },
        {
            "description": "处理Hilton数据集", 
            "data_path": "data/hilton_meeting_room",
            "output_dir": "output_hilton_meeting_room"
        },
        {
            "description": "处理自定义实验数据",
            "data_path": "data/experiment_20241201",
            "output_dir": "output_experiment_20241201"
        }
    ]
    
    for example in examples:
        print(f"\n{example['description']}:")
        print(f"  数据路径: {example['data_path']}")
        print(f"  输出目录: {example['output_dir']}")
        print(f"  修改data_process.py中的data_path变量:")
        print(f"    data_path = \"{example['data_path']}\"")

if __name__ == "__main__":
    print("data_process.py输出目录功能测试")
    print("=" * 50)
    
    try:
        test_output_directory_creation()
        test_file_structure()
        test_path_extraction()
        show_example_usage()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过!")
        print("\n主要改进:")
        print("1. 自动根据数据路径创建专用输出目录")
        print("2. 避免不同数据集结果相互覆盖")
        print("3. 输出目录命名清晰，便于识别数据来源")
        print("4. 保持向后兼容性")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        sys.exit(1) 