import requests
import json
import time
import numpy as np

# 推理服务地址
INFERENCE_SERVICE_URL = "http://localhost:5002"

def test_particle_filter():
    """测试粒子滤波器功能"""
    print("=== 测试PyTorch粒子滤波器 ===")
    
    # 模拟传感器数据
    sensor_data = []
    for i in range(5):
        accel = np.random.normal(0, 1, 3).tolist()
        gyro = np.random.normal(0, 0.1, 3).tolist()
        sensor_data.append({
            "accel": accel,
            "gyro": gyro,
            "timestamp": time.time() + i
        })
    
    device_id = "1207472511626099"
    
    # 连续发送多个请求，测试粒子滤波器的更新
    for i in range(5):
        print(f"\n--- 第 {i+1} 次推理请求 ---")
        
        payload = {
            "device_id": device_id,
            "data": {
                "sensor_data": sensor_data
            }
        }
        
        try:
            response = requests.post(
                f"{INFERENCE_SERVICE_URL}/inference",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"推理结果: {result.get('inference_result')}")
                print(f"粒子滤波器估计: {result.get('particle_filter_estimate')}")
                print(f"状态: {result.get('status')}")
            else:
                print(f"请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except Exception as e:
            print(f"请求异常: {str(e)}")
        
        time.sleep(1)  # 等待1秒再发送下一个请求

def test_particle_filter_status():
    """测试粒子滤波器状态查询"""
    print("\n=== 查询粒子滤波器状态 ===")
    
    try:
        response = requests.get(f"{INFERENCE_SERVICE_URL}/particle_filters")
        if response.status_code == 200:
            result = response.json()
            for device_id, info in result.items():
                print(f"设备 {device_id}:")
                print(f"  初始化状态: {info.get('is_initialized')}")
                print(f"  是否超时: {info.get('is_timeout')}")
                print(f"  粒子数量: {info.get('particle_count')}")
                print(f"  有效粒子数: {info.get('effective_particle_size'):.2f}")
                print(f"  当前估计: {info.get('current_estimate')}")
        else:
            print(f"获取状态失败: {response.status_code}")
    except Exception as e:
        print(f"获取状态异常: {str(e)}")

def test_timeout_scenario():
    """测试超时场景"""
    print("\n=== 测试超时场景 ===")
    
    device_id = "test_device_timeout"
    
    # 第一次请求
    print("发送第一次请求...")
    payload = {
        "device_id": device_id,
        "data": {
            "sensor_data": [
                {"accel": [0.1, 0.2, 9.8], "gyro": [0.01, 0.02, 0.03]}
            ]
        }
    }
    
    try:
        response = requests.post(f"{INFERENCE_SERVICE_URL}/inference", json=payload)
        if response.status_code == 200:
            print("第一次请求成功")
    except Exception as e:
        print(f"第一次请求失败: {str(e)}")
    
    # 检查状态
    print("\n检查粒子滤波器状态...")
    test_particle_filter_status()
    
    # 等待超时
    print("\n等待12秒（超过10秒超时）...")
    time.sleep(12)
    
    # 再次检查状态
    print("\n再次检查粒子滤波器状态...")
    test_particle_filter_status()
    
    # 发送第二次请求
    print("\n发送第二次请求...")
    try:
        response = requests.post(f"{INFERENCE_SERVICE_URL}/inference", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("第二次请求成功")
            print(f"推理结果: {result.get('inference_result')}")
            print(f"粒子滤波器估计: {result.get('particle_filter_estimate')}")
    except Exception as e:
        print(f"第二次请求失败: {str(e)}")

if __name__ == "__main__":
    # 1. 测试粒子滤波器
    test_particle_filter()
    
    # 2. 测试状态查询
    test_particle_filter_status()
    
    # 3. 测试超时场景
    test_timeout_scenario() 