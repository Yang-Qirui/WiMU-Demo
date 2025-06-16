import os
import numpy as np
import ast
import shutil
from utils import *
import json
import re
import matplotlib.pyplot as plt
from itertools import combinations
import time
from functools import wraps
import torch
from config import *
# from test import simple_optimizer

logger = init_logger("data_process_logger", "data_process.log")

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper

def process_rot(rot_file_path):
    '''
        Return N * 2 ndarray. Each row refers to a record.
        The first element of each row is timestamp, the second is yaw (in radius)
    '''
    logger.info("Processing Rotation Vector")
    with open(rot_file_path) as f:
        rot_records = f.readlines()
    parsed_rot_records = []
    for i, record in enumerate(rot_records):
        data = record.strip().split()
        if len(data) == 6:
            parsed_rot_records.append([float(num) if i != 0 else int(num) for i, num in enumerate(data) ])
        else:    
            logger.warning(f"Invalid record on line {i}: {data}")
    result = np.array(parsed_rot_records)
    yaw_data = []
    for row in result:
        timestamp = row[0]
        x, y, z, w = row[1], row[2], row[3], row[4]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)) # calculate yaw from quaternion
        yaw_data.append([timestamp, yaw])
    yaw_result = np.array(yaw_data)
    return yaw_result

def process_step_count(step_file_path):
    '''
        Return N * 2 ndarray. Each row refers to a record.
        The first element of each row is timestamp, the second is relative step count
    '''
    logger.info("Processing Step Count")
    with open(step_file_path) as f:
        step_records = f.readlines()
    parsed_step_records = []
    for i, record in enumerate(step_records):
        data = record.strip().split()
        if len(data) == 2:
            parsed_step_records.append([float(num) if i != 0 else int(num) for i, num in enumerate(data)])
        else:
            logger.warning(f"Invalid record on line {i}: {data}")
    result = np.array(parsed_step_records)
    min_step_count = np.min(result[:, 1]) # minimum step count
    rel_step_counts = result
    rel_step_counts[:, 1] = rel_step_counts[:, 1] - min_step_count
    return result

def process_step_detector(step_file_path):
    '''
    each line is a timestamp, representing the timestamp of a step
    '''
    with open(step_file_path) as f:
        step_records = f.readlines()
    parsed_step_records = []
    for i, record in enumerate(step_records):
        data = record.strip().split()
        parsed_step_records.append(int(data[0]))
    return np.array(parsed_step_records)

def process_yaws(yaw_file_path):
    logger.info("Processing yaw")
    with open(yaw_file_path) as f:
        yaw_records = f.readlines()
    parsed_yaw_records = []
    for i, record in enumerate(yaw_records):
        data = record.strip().split()
        if len(data) == 2:
            parsed_yaw_records.append([float(num) if i != 0 else int(num) for i, num in enumerate(data)])
        else:
            logger.warning(f"Invalid record on line {i}: {data}")
    result = np.array(parsed_yaw_records)
    result[:, 1] = np.deg2rad(result[:, 1])
    return result

def merge_yaw_step(yaws, step_counts, stride=0.5):
    '''
        Return N * 3 ndarray. Each row refers to a waypoint.
        The first element of each row is timestamp, the second is relative x coordinate, the third refers to relative y coordinate.
    '''
    common_timestamps = np.intersect1d(yaws[:, 0], step_counts[:, 0])
    yaws = yaws[np.isin(yaws[:, 0], common_timestamps)]
    step_counts = step_counts[np.isin(step_counts[:, 0], common_timestamps)]
    unique_values = {}
    for row in step_counts:
        timestamp, value = row
        if value not in unique_values:
            unique_values[value] = timestamp
        else:
            unique_values[value] = min(unique_values[value], timestamp)
    result = []
    for value, min_timestamp in unique_values.items():
        result.append([min_timestamp, value])
    step_counts = np.array(result)
    waypoints = [[yaws[0, 0], 0, 0]]
    start_waypoint = waypoints[0]
    for i in range(len(step_counts) - 1):
        start_time_stamp, start_step_count = step_counts[i, 0], step_counts[i, 1]
        end_time_stamp, end_step_count = step_counts[i + 1, 0], step_counts[i + 1, 1]
        start_yaw_index = np.where(yaws[:, 0] == start_time_stamp)[0]
        end_yaw_index = np.where(yaws[:, 0] == end_time_stamp)[0]
        assert len(start_yaw_index) == len(end_yaw_index) == 1
        start_yaw_index = start_yaw_index.item()
        end_yaw_index = end_yaw_index.item()
        avg_stride = stride * (end_step_count - start_step_count) / (end_yaw_index - start_yaw_index)
        next_waypoint = [start_time_stamp, start_waypoint[1], start_waypoint[2]]
        for j in range(start_yaw_index, end_yaw_index + 1):
            yaw = yaws[j, 1]
            # next_waypoint[1] += avg_stride * np.sin(yaw) 
            # next_waypoint[2] += avg_stride * np.cos(yaw)
            tmp_waypoint = [yaws[j, 0], next_waypoint[1] - avg_stride * np.cos(yaw), next_waypoint[2] - avg_stride * np.sin(yaw)]
            # print(end_step_count - start_step_count, end_yaw_index - start_yaw_index, avg_stride, next_waypoint)
            waypoints.append(tmp_waypoint)
            next_waypoint = waypoints[-1]
        start_waypoint = waypoints[-1]
    return np.array(waypoints)

def process_wifi(wifi_file_path):
    logger.info("Processing wifi records")
    
    # Create sampling counter file if it doesn't exist
    sampling_counter_path = 'output/data_process/sampling_counter.json'
   
    with open(sampling_counter_path, 'r') as f:
        sampling_counter_dict = json.load(f)
    
    if wifi_file_path in sampling_counter_dict['counted']:
        counted = True
    else:
        counted = False
        sampling_counter_dict["counted"].append(wifi_file_path) 
    
    with open(wifi_file_path) as f:
        wifi_records = f.readlines()
    wifi_record_dict = {}
    
    # Process records
    for i, record in enumerate(wifi_records):
        pattern = r"(\d+)\s+(.*?)\s+([0-9a-fA-F:]+)\s+(\d+)\s+(-\d+)"
        match = re.match(pattern, record)
        if match:
            timestamp = match.group(1)
            ssid = match.group(2)
            bssid = match.group(3)
            freq = match.group(4)
            rssi = int(match.group(5))
        else:
            logger.warning(f"Invalid record on line {i}: {record}")
            continue    
            
        if JD_MODE:
            if rssi <= FILTER_THRESHOLD or "JD" not in ssid: # TODO
                continue
        else:
            if rssi <= FILTER_THRESHOLD: # TODO
                continue
            
        if not counted:
            if bssid in sampling_counter_dict["freq"].keys():
                sampling_counter_dict["freq"][bssid][0] += 1
            else:
                band = '5G' if int(freq) > 5000 else '2.4G'
                sampling_counter_dict["freq"][bssid] = [1, band]
                
        if timestamp in wifi_record_dict.keys():
            band = '5G' if int(freq) > 5000 else '2.4G'
            wifi_record_dict[timestamp].append([bssid, rssi, band])
        else:
            wifi_record_dict[timestamp] = {}
            band = '5G' if int(freq) > 5000 else '2.4G'
            wifi_record_dict[timestamp] = [[bssid, rssi, band]]
                
    with open(sampling_counter_path, 'w') as json_file:
        json.dump(sampling_counter_dict, json_file, indent=4)
    return wifi_record_dict

def process_wifi_with_filter(wifi_file_path, valid_aps):
    """
    Process wifi data using the filtered AP list.
    
    Args:
        wifi_file_path (str): Path to wifi data file
        valid_aps (set): Set of valid AP BSSIDs
    """
    logger.info("Processing wifi records with AP filter")
    
    with open(wifi_file_path) as f:
        wifi_records = f.readlines()
    wifi_record_dict = {}
    
    for i, record in enumerate(wifi_records):
        pattern = r"(\d+)\s+(.*?)\s+([0-9a-fA-F:]+)\s+(\d+)\s+(-\d+)"
        match = re.match(pattern, record)
        if match:
            timestamp = match.group(1)
            ssid = match.group(2)
            bssid = match.group(3)
            freq = match.group(4)
            rssi = int(match.group(5))
        else:
            logger.warning(f"Invalid record on line {i}: {record}")
            continue    
            
        if JD_MODE:
            if rssi <= FILTER_THRESHOLD or bssid not in valid_aps or "JD" not in ssid: #TODO
                continue
        else:
            if rssi <= FILTER_THRESHOLD or bssid not in valid_aps: #TODO
                continue
                
        if timestamp in wifi_record_dict.keys():
            if int(freq) > 5000:
                wifi_record_dict[timestamp].append([bssid, rssi, '5G'])
            if 2000 < int(freq) < 5000:
                wifi_record_dict[timestamp].append([bssid, rssi, '2.4G'])
        else:
            wifi_record_dict[timestamp] = {}
            if int(freq) > 5000:
                wifi_record_dict[timestamp] = [[bssid, rssi, '5G']]
            if 2000 < int(freq) < 5000:
                wifi_record_dict[timestamp] = [[bssid, rssi, '2.4G']]
    
    return wifi_record_dict

def align_wifi_waypoint(wifi_records, waypoints):
    wifi_record_dict = {}
    selected_waypoints = []
    for timestamp, wifi_record in wifi_records.items():
        diff = np.abs(waypoints[:, 0] - int(timestamp))
        index = np.argmin(diff)
        waypoint = waypoints[index]
        selected_waypoints.append(waypoint)
        wifi_record_dict[tuple(waypoint[1:])] = wifi_record
    selected_waypoints = np.array(selected_waypoints)
    return wifi_record_dict
               
def read_waypoint(wifi_records):
    timestamp_records = {}
    with open("./output/data_process/ap_unions.json", 'r') as f:
        ap_mapping = json.load(f)
    
    for i, record in enumerate(wifi_records):
        pattern = r"(\d+)\s+(.*?)\s+([0-9a-fA-F:]+)\s+(\d+)\s+(-\d+)"
        match = re.match(pattern, record)
        if match:
            timestamp = match.group(1)
            ssid = match.group(2)
            bssid = match.group(3)
            freq = match.group(4)
            rssi = match.group(5)
        else:
            parts = record.split(",")
            coor = (float(parts[0]), float(parts[1]))
            logger.warning(f"Coordinate found on line {i}: {record}")
            continue
            
        if bssid not in ap_mapping.keys():
            continue
        bssid = ap_mapping[bssid]
        # 根据频率判断频段
        band = '5G' if int(freq) > 5000 else '2.4G'
        if int(rssi) > FILTER_THRESHOLD:
            if timestamp not in timestamp_records:
                timestamp_records[timestamp] = {}
            if bssid in timestamp_records[timestamp]:
                timestamp_records[timestamp][bssid].append((int(rssi), band))
            else:
                timestamp_records[timestamp][bssid] = [(int(rssi), band)]
                
    # Calculate mean RSSI for each AP at each timestamp
    new_timestamp_records = {}
    for timestamp in timestamp_records:
        new_timestamp_records[timestamp] = {}
        for bssid in timestamp_records[timestamp]:
            # 计算平均RSSI
            avg_rssi = np.mean([rssi for rssi, _ in timestamp_records[timestamp][bssid]])
            # 获取频段信息（同一个AP在同一时间戳的频段应该相同）
            band = timestamp_records[timestamp][bssid][0][1]
            new_timestamp_records[timestamp][bssid] = (avg_rssi, band)
    
    return new_timestamp_records

def merge_rssi(rssi_list):
    rssi_list = sorted(rssi_list)
    merged_rssis = [int(rssi_list[0])]
    for i in range(1, len(rssi_list)):
        if np.abs(rssi_list[i] - merged_rssis[-1]) <= 2:
            continue
        else:
            merged_rssis.append(int(rssi_list[i]))
    return merged_rssis
 
def process_euler(euler_file):
    '''
        euler_file: euler.txt
        return: euler_records
        first element is timestamp, second is yaw, third is pitch, fourth is roll
    '''
    with open(euler_file) as f:
        euler_records = f.readlines()
    euler_records = [record.strip().split(" ") for record in euler_records]
    return np.array(euler_records)

@log_execution_time
def read_file_with_step_detector(dir_path, ap_unions, AP_dist_count, AP_dist, AP_dist_sq, valid_aps, verbose=True):
    logger.info(f"Processing {dir_path}")
    
    # Process rotation and step data
    euler_file = os.path.join(dir_path, 'euler.txt')
    step_file = os.path.join(dir_path, 'single_step_rec.txt')
    wifi_records = process_wifi_with_filter(os.path.join(dir_path, 'wifi.txt'), valid_aps)
    
    # Process rotation data
    euler = process_euler(euler_file)
    step = process_step_detector(step_file)
    # Find closest matching timestamps between euler and step data
    euler_timestamps = euler[:, 0].astype(np.int64)
    step_timestamps = step.astype(np.int64)
    
    # Create arrays to store matched timestamps
    matched_euler_indices = []
    matched_step_indices = []
    
    # For each step timestamp, find closest euler timestamp
    for i, step_ts in enumerate(step_timestamps):
        # Find index of closest euler timestamp
        closest_idx = np.argmin(np.abs(euler_timestamps - step_ts))
        time_diff = abs(int(euler_timestamps[closest_idx]) - int(step_ts))
        
        # Only match if timestamps are within 100ms
        # if time_diff <= 100: // TODO: problem!
        matched_euler_indices.append(closest_idx)
        matched_step_indices.append(i)
            
    # Filter arrays to only include matched timestamps
    euler = euler[matched_euler_indices]
    step = step[matched_step_indices]
    
    # Find common timestamps efficiently
    # Initialize arrays for storing positions
    positions = np.zeros((len(step), 3))  # [timestamp, x, y]
    positions[:, 0] = step  # Set timestamps
    
    curr_x, curr_y = 0, 0
    stride_length = 0.5  # Average stride length in meters
    for i in range(len(step)):
        yaw_angle = float(euler[i, 1])
        yaw = yaw_angle * np.pi / 180.0
        dx = stride_length * np.cos(yaw)
        dy = stride_length * np.sin(yaw)
        curr_x -= dx
        curr_y -= dy
        positions[i, 1] = curr_x
        positions[i, 2] = curr_y
    valid_mask = ~np.isnan(positions).any(axis=1)
    positions = positions[valid_mask]
    # # Create directory if it doesn't exist
    # os.makedirs('data/fig/test', exist_ok=True)
    
    # # Plot positions
    # plt.figure(figsize=(10, 8))
    # plt.scatter(positions[:, 1], positions[:, 2], c='blue', s=20, alpha=0.6, label='Positions')
    # plt.plot(positions[:, 1], positions[:, 2], 'b-', alpha=0.3)  # Connect points with lines
    
    # # Add start/end markers
    # plt.scatter(positions[0, 1], positions[0, 2], c='green', s=100, marker='^', label='Start')
    # plt.scatter(positions[-1, 1], positions[-1, 2], c='red', s=100, marker='v', label='End')
    
    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)') 
    # plt.title('Position Trajectory')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.axis('equal')  # Equal aspect ratio
    
    # # Save figure
    # print(dir_path.split("/")[-1])
    # plt.savefig(f'data/fig/test/{dir_path.split("/")[-1]}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # positions[:, 1:] -= positions[0, 1:]
    # positions[:, 1:] = simple_optimizer(positions[:, 1:])

    wifi_records_coor = align_wifi_waypoint(wifi_records, positions)
    selected_waypoints = list(wifi_records_coor.keys())
    if verbose:
        # Check if first and last positions are at origin (0,0)
        plt.scatter(*zip(*positions[:, 1:]), color='blue', label='RPs')
        plt.scatter(*zip(*selected_waypoints), color='red', label='RP with wifi')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        file_name = dir_path.split("/")[-1]
        plt.savefig(f"./fig/unlabel_trace/langfang_xiaomi/{file_name}.png")
        plt.close()
    # Pre-calculate all waypoint pairs and their distances
    waypoint_pairs = list(combinations(selected_waypoints, 2))
    waypoint_dists = np.array([np.sqrt((wp1[0] - wp2[0])**2 + (wp1[1] - wp2[1])**2) 
                              for wp1, wp2 in waypoint_pairs])
    
    # Process each pair of waypoints efficiently
    for (wp1, wp2), dist in zip(waypoint_pairs, waypoint_dists):
        wifi1 = wifi_records_coor[wp1]
        wifi2 = wifi_records_coor[wp2]
        
        # Create arrays for vectorized operations
        ap1_bssids = [ap[0] for ap in wifi1]
        ap2_bssids = [ap[0] for ap in wifi2]
        ap1_rssis = [ap[1] for ap in wifi1]
        ap2_rssis = [ap[1] for ap in wifi2]
        ap1_bands = [ap[2] for ap in wifi1]  # 获取AP1的频带信息
        ap2_bands = [ap[2] for ap in wifi2]  # 获取AP2的频带信息
        
        # Get union IDs efficiently
        ap1_unions = np.array([ap_unions[bssid] for bssid in ap1_bssids])
        ap2_unions = np.array([ap_unions[bssid] for bssid in ap2_bssids])
        
        # Create masks for valid AP pairs
        valid_pairs = np.array([[ap_unions[b1] != ap_unions[b2] for b2 in ap2_bssids] for b1 in ap1_bssids])
        
        # Calculate distances efficiently
        for i, (ap1_union, ap1_rssi, ap1_band) in enumerate(zip(ap1_unions, ap1_rssis, ap1_bands)):
            for j, (ap2_union, ap2_rssi, ap2_band) in enumerate(zip(ap2_unions, ap2_rssis, ap2_bands)):
                if valid_pairs[i, j]:
                    # 根据AP的频带计算LDPL值
                    ldpl1 = LDPL(ap1_rssi, band=ap1_band, mode=LDPL_MODE)
                    ldpl2 = LDPL(ap2_rssi, band=ap2_band, mode=LDPL_MODE)
                    new_dist = dist + ldpl1 + ldpl2
                    # Update matrices efficiently
                    AP_dist_count[ap1_union, ap2_union] += 1
                    AP_dist_count[ap2_union, ap1_union] += 1
                    AP_dist[ap1_union, ap2_union] += new_dist
                    AP_dist[ap2_union, ap1_union] += new_dist
                    AP_dist_sq[ap1_union, ap2_union] += np.square(new_dist)
                    AP_dist_sq[ap2_union, ap1_union] += np.square(new_dist)
    
    return AP_dist_count, AP_dist, AP_dist_sq, len(selected_waypoints)

def build_consecutive_step_dataset(dir_path, ap_unions, norm_params, verbose=False):
    # Process parent directory by delegating to process_directory
    """Build dataset of consecutive steps with wifi measurements"""
    logger.info(f"Building consecutive step dataset from {dir_path}")
    
    # Process files
    euler_file = os.path.join(dir_path, 'euler.txt')
    step_file = os.path.join(dir_path, 'single_step.txt')
    wifi_records = process_wifi(os.path.join(dir_path, 'wifi.txt'))
    
    euler = process_euler(euler_file)
    step = process_step_detector(step_file)
    
    # Match timestamps and calculate positions
    euler_timestamps = euler[:, 0].astype(np.int64)
    step_timestamps = step.astype(np.int64)
    
    matched_euler_indices = []
    matched_step_indices = []
    
    for i, step_ts in enumerate(step_timestamps):
        closest_idx = np.argmin(np.abs(euler_timestamps - step_ts))
        time_diff = abs(int(euler_timestamps[closest_idx]) - int(step_ts))
        if time_diff <= 100:
            matched_euler_indices.append(closest_idx)
            matched_step_indices.append(i)
            
    euler = euler[matched_euler_indices]
    step = step[matched_step_indices]
    
    positions = np.zeros((len(step), 3))
    positions[:, 0] = step
    
    curr_x, curr_y = 0, 0
    stride_length = 0.5
    for i in range(len(step)):
        yaw = float(euler[i, 1]) * np.pi / 180.0
        dx = stride_length * np.cos(yaw)
        dy = stride_length * np.sin(yaw)
        curr_x -= dx
        curr_y -= dy
        positions[i, 1:] = [curr_x, curr_y]
        
    valid_mask = ~np.isnan(positions).any(axis=1)
    positions = positions[valid_mask]
    wifi_records_coor = align_wifi_waypoint(wifi_records, positions)
    
    # Build consecutive step pairs dataset
    wifi_inputs = []
    distance_labels = []


    pos_range = norm_params['pos_range']
    pos_min = norm_params['pos_min']
    
    waypoints = list(wifi_records_coor.keys())
    for i in range(len(waypoints)-1):
        for j in range(i+1, len(waypoints)):
            wifi1 = wifi_records_coor[waypoints[i]]
            wifi2 = wifi_records_coor[waypoints[j]]
            
            wp1, wp2 = np.array(waypoints[i]), np.array(waypoints[j])
            wp1_pos = (wp1 - pos_min) / pos_range
            wp2_pos = (wp2 - pos_min) / pos_range
            # dist = np.sqrt(np.sum((wp1_pos - wp2_pos)**2))
            dist_vector = wp2_pos - wp1_pos

            # Get wifi readings for farthest point
            # Convert wifi readings to tensors
            union_num = len(set(ap_unions.values()))
            wifi1_tensor = torch.zeros(union_num)
            wifi2_tensor = torch.zeros(union_num)

            # Handle multiple BSSIDs mapping to same union by taking mean RSSI
            union_to_rssis1 = {}
            union_to_rssis2 = {}
            
            # First wifi scan - collect all RSSIs per union
            for bssid, rssi, _ in wifi1:
                if bssid in ap_unions:
                    union_idx = ap_unions[bssid]
                    if union_idx not in union_to_rssis1:
                        union_to_rssis1[union_idx] = []
                    union_to_rssis1[union_idx].append(rssi)
                    
            # Second wifi scan - collect all RSSIs per union
            for bssid, rssi, _ in wifi2:
                if bssid in ap_unions:
                    union_idx = ap_unions[bssid]
                    if union_idx not in union_to_rssis2:
                        union_to_rssis2[union_idx] = []
                    union_to_rssis2[union_idx].append(rssi)

            
                    
                    
            # Fill tensors with mean RSSI values
            for union_idx, rssis in union_to_rssis1.items():
                mean_rssi = np.mean(rssis)
                band = '5G' if any(b[2] == '5G' for b in wifi1 if b[0] in ap_unions and ap_unions[b[0]] == union_idx) else '2.4G'
                wifi1_tensor[union_idx] = 1 / (1 + LDPL(mean_rssi, band=band, mode=LDPL_MODE))
                
            for union_idx, rssis in union_to_rssis2.items():
                mean_rssi = np.mean(rssis)
                band = '5G' if any(b[2] == '5G' for b in wifi2 if b[0] in ap_unions and ap_unions[b[0]] == union_idx) else '2.4G'
                wifi2_tensor[union_idx] = 1 / (1 + LDPL(mean_rssi, band=band, mode=LDPL_MODE))
            
            
            # Normalize tensors
            wifi1_tensor = wifi1_tensor / wifi1_tensor.sum() if wifi1_tensor.sum() > 0 else wifi1_tensor
            wifi2_tensor = wifi2_tensor / wifi2_tensor.sum() if wifi2_tensor.sum() > 0 else wifi2_tensor
            
            wifi_inputs.append((wifi1_tensor, wifi2_tensor))
            distance_labels.append(dist_vector)

    
    logger.info(f"Saved {len(wifi_inputs)} wifi input pairs, distance labels")
    return wifi_inputs, distance_labels, waypoints

@log_execution_time
def process_directory_for_steps(dir_path, ap_unions, norm_params, verbose=False):
    """
    Process all subdirectories in the given directory to build consecutive step dataset.
    
    Args:
        dir_path (str): Path to parent directory containing subdirectories to process
        ap_unions (dict): Dictionary mapping BSSIDs to their union IDs
        norm_params (dict): Dictionary containing normalization parameters
        verbose (bool): Whether to print verbose output
        
    Returns:
        tuple: (wifi_inputs, distance_labels, farthest_points) containing combined data from all subdirs
    """
    wifi_inputs = []
    distance_labels = []
    farthest_points = []
    waypoints = []
    id_mask = []
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    logger.info(f"Found {len(subdirs)} subdirectories to process")
    trace_id_map = {}
    
    for trace_id, subdir in enumerate(subdirs):
        subdir_path = os.path.join(dir_path, subdir)
        logger.info(f"Processing subdirectory: {subdir_path}")
        
        # Process each subdirectory and collect the data
        try:
            subdir_wifi_inputs, subdir_distance_labels, subdir_waypoints = build_consecutive_step_dataset(
                subdir_path, ap_unions, norm_params, verbose=verbose
            )
        except Exception as e:
            logger.error(f"Error encountered when processing {subdir_path}, {e}")
            continue
        wifi_inputs.extend(subdir_wifi_inputs)
        distance_labels.extend(subdir_distance_labels)
        waypoints.extend(subdir_waypoints)
        id_mask.extend([trace_id] * len(subdir_wifi_inputs))
        trace_id_map[trace_id] = subdir
    # Convert lists to numpy arrays
    wifi_inputs_np = np.array([(w1.numpy(), w2.numpy()) for w1, w2 in wifi_inputs], dtype=np.float32)
    distance_labels_np = np.array(distance_labels, dtype=np.float32)
    # farthest_points_np = np.array(farthest_points, dtype=np.float32)
    waypoints_np = np.array([np.array(waypoint) for waypoint in waypoints], dtype=np.float32)
    id_mask_np = np.array(id_mask, dtype=np.int32)
    # Create output directory if it doesn't exist
    os.makedirs("data/pre_training", exist_ok=True)
    
    # Save arrays
    np.save("data/pre_training/wifi_inputs.npy", wifi_inputs_np)
    np.save("data/pre_training/distance_labels.npy", distance_labels_np)
    np.save("data/pre_training/waypoints.npy", waypoints_np)
    np.save("data/pre_training/id_mask.npy", id_mask_np)
    json.dump(trace_id_map, open("data/pre_training/trace_id_map.json", 'w'))
    # np.save("data/pre_training/farthest_points.npy", farthest_points_np)
    
    logger.info(f"Saved {len(wifi_inputs)} wifi input pairs, {len(distance_labels)} distance labels, and {len(farthest_points)} farthest points")
    return wifi_inputs, distance_labels, farthest_points
    

def merge_similar_aps(aps_dict):
    """
    Merge Wi-Fi APs with similar MAC addresses based on their prefix.
    
    Args:
        aps_dict (dict): Dictionary of APs with their information
        
    Returns:
        dict: Merged dictionary of APs
    """
    logger.info("Merging similar APs based on MAC address prefix")
    
    # Create a dictionary to store merged APs
    merged_aps = {}
    
    # Sort APs by frequency to process most common ones first
    sorted_aps = dict(sorted(aps_dict.items(), key=lambda x: x[1]['count'], reverse=True))
    
    for bssid, info in sorted_aps.items():
        # Get the prefix (first 11 characters of MAC address)
        
        prefix = bssid[:16]
        
        # Check if we've seen this prefix before
        if prefix in merged_aps:
            # Merge with existing AP
            merged_aps[prefix]['count'] += info['count']
            # Keep the most common frequency and band
            if info['count'] > merged_aps[prefix]['count']:
                merged_aps[prefix]['frequency'] = info['frequency']
                merged_aps[prefix]['band'] = info['band']
            # Add the original BSSID to a list of merged BSSIDs
            if 'merged_bssids' not in merged_aps[prefix]:
                merged_aps[prefix]['merged_bssids'] = []
            merged_aps[prefix]['merged_bssids'].append(bssid)
        else:
            # Create new entry
            merged_aps[prefix] = {
                'frequency': info['frequency'],
                'band': info['band'],
                'count': info['count'],
                'merged_bssids': [bssid]
            }
    
    # Sort by count again
    merged_aps = dict(sorted(merged_aps.items(), key=lambda x: x[1]['count'], reverse=True))
    
    logger.info(f"Merged {len(aps_dict)} APs into {len(merged_aps)} unique APs")
    return merged_aps

def create_ap_unions(merged_aps):
    """
    Create unions of similar APs with consecutive IDs.
    
    Args:
        merged_aps (dict): Dictionary of merged APs
        
    Returns:
        dict: Dictionary mapping each BSSID to its union ID
    """
    logger.info("Creating AP unions with consecutive IDs")
    
    # Create a dictionary to store BSSID to union ID mapping
    ap_unions = {}
    union_id = 0
    
    # Process each merged AP group
    for prefix, info in merged_aps.items():
        # Assign the same union ID to all BSSIDs in the group
        for bssid in info['merged_bssids']:
            ap_unions[bssid] = union_id
        union_id += 1
    
    # Save the union mapping
    output_dir = "output/data_process"
    with open(os.path.join(output_dir, "ap_unions.json"), 'w') as f:
        json.dump(ap_unions, f, indent=4)
    
    logger.info(f"Created {union_id} AP unions")
    return ap_unions

def record_distinct_aps(data_dir, valid_aps):
    """
    Record all distinct Wi-Fi APs from the source data.
    
    Args:
        data_dir (str): Directory containing the Wi-Fi data files
        
    Returns:
        dict: Dictionary containing distinct APs with their frequencies and bands
    """
    logger.info("Recording distinct Wi-Fi APs")
    distinct_aps = {}
    
    # Process all subdirectories
    for dir_name in os.listdir(data_dir):
        wifi_file = os.path.join(data_dir, dir_name, 'wifi.txt')
        if not os.path.exists(wifi_file):
            continue
            
        with open(wifi_file) as f:
            wifi_records = f.readlines()
            
        for record in wifi_records:
            pattern = r"(\d+)\s+(.*?)\s+([0-9a-fA-F:]+)\s+(\d+)\s+(-\d+)"
            match = re.match(pattern, record)
            if match:
                bssid = match.group(3)
                freq = int(match.group(4))
                rssi = int(match.group(5))
                ssid = match.group(2)
                if rssi > FILTER_THRESHOLD and bssid in valid_aps:
                    if bssid not in distinct_aps:
                        distinct_aps[bssid] = {
                            'ssid': ssid,
                            'frequency': freq,
                            'band': '5G' if freq > 5000 else '2.4G',
                            'count': 1
                        }
                    else:
                        distinct_aps[bssid]['count'] += 1
    
    # Merge similar APs
    merged_aps = merge_similar_aps(distinct_aps)
    
    # Create AP unions
    ap_unions = create_ap_unions(merged_aps)
    
    # Save results to JSON files
    output_dir = "output/data_process"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original distinct APs
    with open(os.path.join(output_dir, "distinct_aps.json"), 'w') as f:
        json.dump(distinct_aps, f, indent=4)
    
    # Save merged APs
    with open(os.path.join(output_dir, "merged_aps.json"), 'w') as f:
        json.dump(merged_aps, f, indent=4)
    
    logger.info(f"Found {len(distinct_aps)} distinct APs, merged into {len(merged_aps)} unique APs")
    return merged_aps, ap_unions

@log_execution_time
def process_waypoints(is_training: bool, data_path: str):
    """Process waypoint data for either training or testing.
    
    Args:
        is_training (bool): True for training mode, False for testing mode
        data_path (str): Path to the data directory containing waypoint data
    """
    sub_dir = os.listdir(data_path)
    
    # Set output directory based on mode
    output_dir = "data/train_json" if is_training else "data/test_json"
    
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
        
    pattern = r'^\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*$'
    wifi_pattern  = r"(\d+)\s+(.*?)\s+([0-9a-fA-F:]+)\s+(\d+)\s+(-\d+)"
    training_coors = []

    for dir in sub_dir:
        logger.info(f"Processing {dir}")
        with open(f"{data_path}/{dir}/wifi.txt", 'r') as f:
            all_records = f.readlines()
            
        start_coor_id = -1
        last_timestamp = -1
        single_trace_data = {}
        
        for i, record in enumerate(all_records):
            if re.match(pattern, record):
                coordinate_list = record.strip().split(',')
                coordinate_tuple = (float(coordinate_list[0]), float(coordinate_list[1]))
            elif re.match(wifi_pattern, record):
                current_timestamp = int(record.split(' ')[0])
                if start_coor_id == -1:
                    start_coor_id = i
                    last_timestamp = current_timestamp
                if current_timestamp != last_timestamp:
                    wifi_records = all_records[start_coor_id: i]
                    try:
                        bssid_rssi = read_waypoint(wifi_records)
                        for k, v in bssid_rssi.items():
                            single_trace_data[k] = v
                    except Exception as e:
                        # Extract waypoint number from directory name
                        waypoint_num = dir.split('-')[1] if '-' in dir else dir
                        logger.error(f"Error encountered when processing waypoint #{waypoint_num}, {e}")
                    start_coor_id = i
                    last_timestamp = current_timestamp
                
        # Save processed data
        if single_trace_data:
            with open(f"{output_dir}/{coordinate_tuple}.json", 'w') as f:
                json.dump(single_trace_data, f)
            training_coors.append(coordinate_tuple)
    return torch.tensor(training_coors)

@log_execution_time
def process_all(unlabeled_dir_path):
    """
    Process all unlabeled data and construct the AP graph.
    
    Args:
        unlabeled_dir_path (str): Path to the unlabeled data directory
        
    Returns:
        int: Number of unique APs after merging
    """
    logger.info(f"Processing {unlabeled_dir_path}")
    sub_dir = os.listdir(unlabeled_dir_path)
    for dir in sub_dir:
        dir_path = os.path.join(unlabeled_dir_path, dir)
        wifi_file = os.path.join(dir_path, 'wifi.txt')
        process_wifi(wifi_file)
    sampling_counter = json.load(open("output/data_process/sampling_counter.json", 'r'))
    valid_aps = [bssid for bssid, (count, _) in sampling_counter["freq"].items() if count >= MINIMUM_SCANNING_TIMES] # minimum scanning tims
    # First, record and merge all distinct APs
    merged_aps, ap_unions = record_distinct_aps(unlabeled_dir_path, valid_aps)
    ap_num = len(merged_aps)
    
    # Initialize matrices for graph construction
    AP_dist_count = np.zeros((ap_num, ap_num))
    AP_dist = np.ones_like(AP_dist_count)
    AP_dist_sq = np.zeros_like(AP_dist_count)  # For standard deviation calculation
    waypoint_cnt = 0
    wifi_cnt = 0
    rp_nums = 0
    
    # Process each subdirectory
    for dir in sub_dir:
        dir_path = os.path.join(unlabeled_dir_path, dir)
        try:
            AP_dist_count, AP_dist, AP_dist_sq, tmp_rp = read_file_with_step_detector(dir_path, ap_unions, AP_dist_count, AP_dist, AP_dist_sq, valid_aps, verbose=True)
            rp_nums += tmp_rp
        except Exception as e:
            logger.error(f"Error encountered when processing {dir_path}, {e}")
    
    # Save the graph matrices
    output_dir = "output/data_process"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "ap_dist_count.npy"), AP_dist_count)
    np.save(os.path.join(output_dir, "ap_dist.npy"), AP_dist)
    np.save(os.path.join(output_dir, "ap_dist_sq.npy"), AP_dist_sq)
    
    AP_dist_count[AP_dist_count == 0] = 1
    mean_dist = np.divide(AP_dist, AP_dist_count)
    # Correct standard deviation calculation: sqrt(sum(x^2)/n - mean^2)
    std_dist = np.sqrt(np.divide(AP_dist_sq, AP_dist_count) - np.square(mean_dist))
    
    # Save statistics
    np.save(os.path.join(output_dir, "ap_mean.npy"), mean_dist)
    np.save(os.path.join(output_dir, "ap_std.npy"), std_dist)
    np.savetxt(os.path.join(output_dir, "ap_mean.txt"), mean_dist)
    np.savetxt(os.path.join(output_dir, "ap_std.txt"), std_dist)
    
    logger.info(f"waypoint: {waypoint_cnt}, wifi: {wifi_cnt}, unlabeled rp: {rp_nums}")
    return ap_unions

def plot_ap_mean_heatmap(output_dir="output/data_process"):
    """
    Load AP mean distances from file and generate a heatmap visualization.
    
    Args:
        output_dir (str): Directory containing ap_mean.npy and where to save the heatmap
    """
    # Load the AP mean distances
    mean_dist = np.load(os.path.join(output_dir, "ap_mean.npy"))
    
    # Generate and save heatmap
    heatmap_path = os.path.join(output_dir, "ap_mean_heatmap.png") 
    heatmap(mean_dist, heatmap_path)
    logger.info(f"Saved AP mean distance heatmap to {heatmap_path}")

if __name__ == "__main__":
    start_time = time.time()
    
    # Add the new function call
    # merged_aps, ap_unions = record_distinct_aps("data/WiMU data/unlabeled_data")
    sampling_counter_path = 'output/data_process/sampling_counter.json'
    # if not os.path.exists(sampling_counter_path):
    sampling_counter_dict = {
        "counted": [],
        "freq": {}
    }
    os.makedirs(os.path.dirname(sampling_counter_path), exist_ok=True)
    with open(sampling_counter_path, 'w') as f:
        json.dump(sampling_counter_dict, f, indent=4)
    
    ap_unions = process_all("data/JD_langfang_high/unlabeled")
    training_coors = process_waypoints(is_training=True, data_path="data/JD_langfang_high/train")
    testing_coors = process_waypoints(is_training=False, data_path="data/JD_langfang_high/test")
    training_pos_range = torch.max(training_coors, dim=0)[0] - torch.min(training_coors, dim=0)[0]
    # training_pos_range = torch.tensor([20.6, 81])
    training_pos_min = torch.min(training_coors, dim=0)[0]
    # training_pos_min = torch.tensor([8.4, -33])
    process_directory_for_steps("data/JD_langfang_high/unlabeled", ap_unions, {"pos_range": training_pos_range.numpy(), "pos_min": training_pos_min.numpy()})
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    # plot_ap_mean_heatmap()