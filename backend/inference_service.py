import os
import json
import time
import logging
import requests
import numpy as np
import torch
import math
from flask import Flask, request, jsonify
from typing import Dict, Optional, Tuple
import threading
from collections import defaultdict
from datetime import datetime
from model import *
from torch_geometric.nn import GAE

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load graph dataset
graph_dataset = torch.load("./params/graph_dataset.pt", weights_only=False).to(device)
# Initialize models
gnn = GAE(GCNEncoder(128, 128), MLPDecoder(128)).to(device)
mlp = MyMLP(128, 2).float().to(device)
model = JointModel(gnn, mlp).to(device)
model.load_state_dict(torch.load('./params/fine_tuned_model.pt'))
model.eval()

# Load normalization parameters
norm_params = torch.load('./params/norm_params.pt')
pos_range = norm_params['pos_range'].to(device)
pos_min = norm_params['pos_min'].to(device)

# Load AP mapping
with open("./params/ap_union.json", 'r') as f:
    ap_mapping = json.load(f)

# 设备粒子滤波器管理
device_pfs: Dict[str, TorchParticleFilter] = {}
device_last_update: Dict[str, float] = {}
PF_TIMEOUT = 10.0  # 10秒超时
lock = threading.Lock()

def wifi_inference(data):
    """WiFi推理函数"""
    bssid_rssi_map = defaultdict(list)
    bssi_band_map = {}
    
    for entry in data['wifi_entries']:
        bssid = entry['bssid']
        freq = entry['frequency']
        rssi = entry['rssi']
        
        # 根据频率判断频段
        band = '5G' if freq >= 5000 else '2.4G'
        
        if bssid in ap_mapping and rssi > -75:
            bssid_rssi_map[ap_mapping[bssid]].append((rssi, band))
            bssi_band_map[ap_mapping[bssid]] = band

    if bssid_rssi_map:
        # Create input weights tensor
        input_weights = torch.zeros((graph_dataset.num_nodes,), dtype=torch.float32).to(device)
        
        for union_id, rssi_band_list in bssid_rssi_map.items():
            # 计算平均RSSI
            avg_rssi = sum(rssi for rssi, _ in rssi_band_list) / len(rssi_band_list)
            # 使用对应频段的LDPL参数
            band = bssi_band_map[union_id]
            weight = 1 / (1 + LDPL(avg_rssi, band=band, mode=LDPL_MODE))
            input_weights[union_id] = weight
            logging.debug(f"{union_id}, {avg_rssi}, {band}, {weight}")
        
        # Normalize weights
        input_weights = input_weights.unsqueeze(0)  # Add batch dimension
        input_weights = input_weights / input_weights.sum()
        
        # Make prediction
        with torch.no_grad():
            predict_coors, _ = model(graph_dataset, input_weights)
            embs = model.gen_emb(graph_dataset, input_weights)
            predict_coors = predict_coors * pos_range + pos_min
            predict = predict_coors.cpu().tolist()[0]  # Remove batch dimension
        return predict
    else:
        return None

def get_or_create_pf(device_id: str) -> Tuple[TorchParticleFilter, float]:
    """获取或创建设备的粒子滤波器"""
    with lock:
        if device_id not in device_pfs:
            device_pfs[device_id] = TorchParticleFilter(num_particles=1000, device=device)
            device_last_update[device_id] = time.time()
            logger.info(f"Created new particle filter for device {device_id}")
        
        return device_pfs[device_id], device_last_update[device_id]

def update_device_pf_time(device_id: str):
    """更新设备的粒子滤波器时间"""
    with lock:
        device_last_update[device_id] = time.time()

def inference_with_pf(data):
    """带粒子滤波器的推理函数"""
    device_id = data.get('device_id', 'unknown')
    
    # 1. WiFi推理
    predict = wifi_inference(data)
    if predict is None:
        return {"error": "No valid WiFi data"}
    
    # 2. 获取设备的粒子滤波器
    pf, last_update_time = get_or_create_pf(device_id)
    current_time = time.time()
    
    # 3. 检查粒子滤波器是否超时
    if current_time - last_update_time > PF_TIMEOUT:
        logger.info(f"Particle filter timeout for device {device_id}, resetting...")
        pf.reset(predict, data.get('obs_noise', 3.0))
        update_device_pf_time(device_id)
        return {"x": predict[0], "y": predict[1]}
    
    # 4. 检查IMU数据
    if data.get('dx', 0) == 0 and data.get('dy', 0) == 0:
        # 没有IMU数据，重置粒子滤波器
        logger.info(f"No IMU data for device {device_id}, resetting particle filter")
        pf.reset(predict, data.get('obs_noise', 3.0))
        update_device_pf_time(device_id)
        return {"x": predict[0], "y": predict[1]}
    
    # 5. 更新粒子滤波器
    pf.update(
        observation=(predict[0], predict[1]), 
        system_input=(data['dx'], data['dy']), 
        system_noise_scale=data.get('system_noise', 1.0), 
        obs_noise_scale=data.get('obs_noise', 3.0)
    )
    
    # 6. 获取估计结果
    estimate = pf.estimate.cpu().numpy().tolist()
    update_device_pf_time(device_id)
    
    logger.info(f"Device {device_id} - Observation: {predict}, Estimation: {estimate}")
    return {"x": estimate[0], "y": estimate[1]}

def reset_particle_filter(device_id: str, data: dict):
    """重置指定设备的粒子滤波器"""
    predict = wifi_inference(data)
    if predict is None:
        return {"error": "No valid WiFi data"}
    
    pf, _ = get_or_create_pf(device_id)
    pf.reset(predict, data.get('obs_noise', 3.0))
    update_device_pf_time(device_id)
    
    return {"x": predict[0], "y": predict[1]}

def get_particle_filter_status():
    """获取所有粒子滤波器状态"""
    with lock:
        filters_info = {}
        for device_id, pf in device_pfs.items():
            last_update = device_last_update.get(device_id, 0)
            filters_info[device_id] = {
                "is_initialized": pf.is_initialized,
                "particle_count": pf.num_particles,
                "effective_particle_size": pf.effective_particle_size(),
                "current_estimate": pf.estimate.cpu().numpy().tolist() if pf.estimate is not None else None,
                "last_update_time": last_update,
                "timeout": time.time() - last_update
            }
        
        return filters_info

def get_health_status():
    """获取健康状态"""
    with lock:
        active_devices = len(device_pfs)
        device_timeouts = {}
        for device_id, last_update in device_last_update.items():
            device_timeouts[device_id] = time.time() - last_update
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": str(device),
        "active_devices": active_devices,
        "device_timeouts": device_timeouts
    }

