#!/usr/bin/env python3
"""
设备命令发送工具
支持向所有连接的设备发送关闭采集和关闭定位命令
"""

import requests
import json
import logging
import time
import sys
import argparse
from typing import List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 后端配置
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

def get_all_devices() -> Optional[List[dict]]:
    """获取所有连接的设备"""
    try:
        url = f"{BACKEND_BASE_URL}/devices"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            devices = response.json()
            logger.info(f"成功获取到 {len(devices)} 个设备")
            return devices
        else:
            logger.error(f"获取设备列表失败，状态码: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"请求设备列表时出错: {e}")
        return None

def send_command_to_device(device_id: str, command: str, verbose: bool = True) -> bool:
    """向指定设备发送命令"""
    try:
        url = f"{BACKEND_BASE_URL}/{command}"
        payload = {"target_device_id": device_id}
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            if verbose:
                logger.info(f"✓ 设备 {device_id}: {command}")
            return True
        else:
            if verbose:
                logger.error(f"✗ 设备 {device_id}: {command} 失败")
            return False
            
    except requests.exceptions.RequestException as e:
        if verbose:
            logger.error(f"✗ 设备 {device_id}: {command} 错误 - {e}")
        return False

def interactive_mode():
    """交互模式 - 显示设备信息并确认后发送命令"""
    logger.info("=== 交互模式 ===")
    
    # 获取所有设备
    devices = get_all_devices()
    if not devices:
        logger.error("无法获取设备列表，退出")
        return
    
    # 过滤掉系统设备（如superuser001）
    user_devices = [device for device in devices if not device['deviceId'].startswith('superuser')]
    
    if not user_devices:
        logger.warning("没有找到用户设备")
        return
    
    logger.info(f"找到 {len(user_devices)} 个用户设备")
    
    # 显示设备信息
    for device in user_devices:
        device_id = device['deviceId']
        device_name = device.get('deviceName', '未知设备')
        logger.info(f"设备: {device_id} ({device_name})")
    
    # 确认是否继续
    confirm = input(f"\n是否向这 {len(user_devices)} 个设备发送关闭命令？(y/N): ")
    if confirm.lower() != 'y':
        logger.info("用户取消操作")
        return
    
    # 发送关闭采集命令
    logger.info("\n=== 发送关闭采集命令 ===")
    end_sample_success = 0
    for device in user_devices:
        device_id = device['deviceId']
        if send_command_to_device(device_id, "end_sample"):
            end_sample_success += 1
        time.sleep(0.1)  # 避免请求过于频繁
    
    # 发送关闭定位命令
    logger.info("\n=== 发送关闭定位命令 ===")
    end_inference_success = 0
    for device in user_devices:
        device_id = device['deviceId']
        if send_command_to_device(device_id, "end_inference"):
            end_inference_success += 1
        time.sleep(0.1)  # 避免请求过于频繁
    
    # 显示结果统计
    logger.info("\n=== 命令发送结果 ===")
    logger.info(f"关闭采集命令: {end_sample_success}/{len(user_devices)} 成功")
    logger.info(f"关闭定位命令: {end_inference_success}/{len(user_devices)} 成功")
    
    if end_sample_success == len(user_devices) and end_inference_success == len(user_devices):
        logger.info("所有命令发送成功！")
    else:
        logger.warning("部分命令发送失败，请检查后端服务状态")

def quick_mode():
    """快速模式 - 直接发送命令，无需确认"""
    logger.info("=== 快速模式 ===")
    
    # 获取所有设备
    devices = get_all_devices()
    if not devices:
        logger.error("无法获取设备列表")
        return
    
    # 过滤掉系统设备
    user_devices = [device for device in devices if not device['deviceId'].startswith('superuser')]
    
    if not user_devices:
        logger.warning("没有找到用户设备")
        return
    
    logger.info(f"找到 {len(user_devices)} 个设备，开始发送关闭命令...")
    
    # 发送关闭采集命令
    logger.info("\n--- 发送关闭采集命令 ---")
    end_sample_success = 0
    for device in user_devices:
        device_id = device['deviceId']
        if send_command_to_device(device_id, "end_sample", verbose=False):
            end_sample_success += 1
        time.sleep(0.05)  # 短暂延迟
    
    # 发送关闭定位命令
    logger.info("\n--- 发送关闭定位命令 ---")
    end_inference_success = 0
    for device in user_devices:
        device_id = device['deviceId']
        if send_command_to_device(device_id, "end_inference", verbose=False):
            end_inference_success += 1
        time.sleep(0.05)  # 短暂延迟
    
    # 显示结果
    logger.info(f"\n=== 完成 ===")
    logger.info(f"关闭采集: {end_sample_success}/{len(user_devices)}")
    logger.info(f"关闭定位: {end_inference_success}/{len(user_devices)}")

def check_backend_service():
    """检查后端服务是否可用"""
    try:
        health_url = f"{BACKEND_BASE_URL}/device_status"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            logger.info("后端服务正常")
            return True
        else:
            logger.error("后端服务不可用")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接到后端服务 {BACKEND_BASE_URL}")
        logger.error("请确保 backend.py 正在运行在端口 8000")
        return False
    except Exception as e:
        logger.error(f"检查后端服务时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="向所有设备发送关闭命令")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="快速模式，直接发送命令无需确认")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="交互模式，显示设备信息并确认后发送命令")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("设备命令发送工具")
    print("=" * 50)
    
    try:
        # 检查后端服务
        if not check_backend_service():
            return
        
        # 根据参数选择模式
        if args.quick:
            quick_mode()
        elif args.interactive:
            interactive_mode()
        else:
            # 默认交互模式
            interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("\n操作被中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")

if __name__ == "__main__":
    main() 