import requests
import json
import time
import numpy as np

# 推理服务地址
INFERENCE_SERVICE_URL = "http://localhost:5002"

def test_wifi_inference():
    """测试WiFi推理"""
    print("=== 测试WiFi推理 ===")
    
    # 模拟WiFi数据
    wifi_data = {
        "device_id": "1207472511626099",
        "timestamp": time.time(),
        "wifiEntries": [
            {
                "bssid": "00:11:22:33:44:55",
                "ssid": "AP1",
                "frequency": 2412,
                "rssi": -45
            },
            {
                "bssid": "00:11:22:33:44:56",
                "ssid": "AP2",
                "frequency": 5180,
                "rssi": -55
            },
            {
                "bssid": "00:11:22:33:44:57",
                "ssid": "AP3",
                "frequency": 2417,
                "rssi": -65
            }
        ],
        "system_noise_scale": 1.0,
        "obs_noise_scale": 3.0
    }
    
    try:
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/inference",
            json=wifi_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("WiFi推理成功!")
            print(f"WiFi结果: {result.get('wifi_result')}")
            print(f"粒子滤波器估计: {result.get('particle_filter_estimate')}")
            print(f"状态: {result.get('status')}")
        else:
            print(f"请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"请求异常: {str(e)}")

def test_imu_data():
    """测试IMU数据"""
    print("\n=== 测试IMU数据 ===")
    
    device_id = "1207472511626099"
    
    # 发送多个IMU数据
    for i in range(5):
        imu_data = {
            "device_id": device_id,
            "timestamp": time.time() + i * 1000,  # 每秒一个数据
            "yaw": 45 + i * 10,  # 逐渐转向
            "stride": 0.5  # 步长
        }
        
        try:
            response = requests.post(
                f"{INFERENCE_SERVICE_URL}/sendimu",
                json=imu_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"IMU数据 {i+1} 发送成功: {result}")
            else:
                print(f"IMU数据发送失败: {response.status_code}")
                
        except Exception as e:
            print(f"IMU数据发送异常: {str(e)}")
        
        time.sleep(0.5)

def test_inference_with_imu():
    """测试带IMU数据的推理"""
    print("\n=== 测试带IMU数据的推理 ===")
    
    device_id = "1207472511626099"
    
    # 先发送一些IMU数据
    print("发送IMU数据...")
    for i in range(3):
        imu_data = {
            "device_id": device_id,
            "timestamp": time.time() + i * 1000,
            "yaw": 30 + i * 15,
            "stride": 0.3
        }
        
        try:
            requests.post(f"{INFERENCE_SERVICE_URL}/sendimu", json=imu_data)
        except:
            pass
        
        time.sleep(0.2)
    
    # 然后进行推理
    print("进行推理...")
    wifi_data = {
        "device_id": device_id,
        "timestamp": time.time() + 3000,  # 稍后的时间戳
        "wifiEntries": [
            {
                "bssid": "00:11:22:33:44:55",
                "ssid": "AP1",
                "frequency": 2412,
                "rssi": -50
            },
            {
                "bssid": "00:11:22:33:44:56",
                "ssid": "AP2",
                "frequency": 5180,
                "rssi": -60
            }
        ],
        "system_noise_scale": 1.0,
        "obs_noise_scale": 3.0
    }
    
    try:
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/inference",
            json=wifi_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("推理成功!")
            print(f"WiFi结果: {result.get('wifi_result')}")
            print(f"粒子滤波器估计: {result.get('particle_filter_estimate')}")
            print(f"状态: {result.get('status')}")
        else:
            print(f"推理失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"推理异常: {str(e)}")

def test_reset():
    """测试重置功能"""
    print("\n=== 测试重置功能 ===")
    
    device_id = "1207472511626099"
    
    reset_data = {
        "device_id": device_id,
        "wifiEntries": [
            {
                "bssid": "00:11:22:33:44:55",
                "ssid": "AP1",
                "frequency": 2412,
                "rssi": -40
            }
        ],
        "obs_noise_scale": 2.0
    }
    
    try:
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/reset",
            json=reset_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("重置成功!")
            print(f"重置位置: {result.get('reset_position')}")
            print(f"状态: {result.get('status')}")
        else:
            print(f"重置失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"重置异常: {str(e)}")

def test_particle_filters_status():
    """测试粒子滤波器状态查询"""
    print("\n=== 查询粒子滤波器状态 ===")
    
    try:
        response = requests.get(f"{INFERENCE_SERVICE_URL}/particle_filters")
        if response.status_code == 200:
            result = response.json()
            for device_id, info in result.items():
                print(f"设备 {device_id}:")
                print(f"  初始化状态: {info.get('is_initialized')}")
                print(f"  粒子数量: {info.get('particle_count')}")
                print(f"  有效粒子数: {info.get('effective_particle_size'):.2f}")
                print(f"  当前估计: {info.get('current_estimate')}")
                print(f"  最后位置: {info.get('last_location')}")
        else:
            print(f"获取状态失败: {response.status_code}")
    except Exception as e:
        print(f"获取状态异常: {str(e)}")

if __name__ == "__main__":
    print("=== WiFi + IMU 推理服务测试 ===")
    
    # 1. 测试WiFi推理
    test_wifi_inference()
    
    # 2. 测试IMU数据
    test_imu_data()
    
    # 3. 测试带IMU的推理
    test_inference_with_imu()
    
    # 4. 测试重置
    test_reset()
    
    # 5. 查询状态
    test_particle_filters_status() 