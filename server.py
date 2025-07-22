import logging.config
import math
from flask import Flask, request, jsonify
from gnn import MyMLP, GCNEncoder, JointModel
import torch
import json
from utils import LDPL, CircularArray
import logging
from collections import defaultdict
from torch_geometric.nn import GAE
from torch_geometric.data import Data
import os
from gnn import *
from particle_filter import TorchParticleFilter
from config import *
from datetime import datetime
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

app = Flask(__name__)
datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# logging.basicConfig(level=logging.DEBUG, filename=f"./log/server_{datetime}.log")
logging.basicConfig(level=logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load graph dataset
graph_dataset = torch.load("./output/graph_dataset.pt", weights_only=False).to(device)
# Initialize models
gnn = GAE(GCNEncoder(128, 128), MLPDecoder(128)).to(device)
mlp = MyMLP(128, 2).float().to(device)
model = JointModel(gnn, mlp).to(device)
model.load_state_dict(torch.load('./output/fine_tuned_model.pt'))
model.eval()

# Load normalization parameters
norm_params = torch.load('./output/norm_params.pt')
pos_range = norm_params['pos_range'].to(device)
pos_min = norm_params['pos_min'].to(device)
# Save normalization parameters as JSON
norm_params_json = {
    'pos_range_x': pos_range.cpu().numpy().tolist()[0],
    'pos_range_y': pos_range.cpu().numpy().tolist()[1],
    'pos_min_x': pos_range.cpu().numpy().tolist()[0],
    'pos_min_y': pos_min.cpu().numpy().tolist()[1]
}


with open("./output/data_process/ap_unions.json", 'r') as f:
    ap_mapping = json.load(f)

pf = TorchParticleFilter(num_particles=1000, device=device)
step_data = CircularArray(100)
last_loc = None

def wifi_inference(data):
    bssid_rssi_map = defaultdict(list)
    bssid_band_map = {}  # 存储每个AP的频段信息
    
    for entry in data['wifiEntries']:
        bssid = entry['bssid']
        freq = entry['frequency']
        rssi = entry['rssi']
        print(f"{bssid}, {entry['ssid']}, {freq}, {rssi}")
        
        # 根据频率判断频段
        band = '5G' if freq >= 5000 else '2.4G'
        
        if bssid in ap_mapping and rssi > FILTER_THRESHOLD:
            bssid_rssi_map[ap_mapping[bssid]].append((rssi, band))
            bssid_band_map[ap_mapping[bssid]] = band

    if bssid_rssi_map:
        # Create input weights tensor
        input_weights = torch.zeros((graph_dataset.num_nodes,), dtype=torch.float32).to(device)
        
        for union_id, rssi_band_list in bssid_rssi_map.items():
            # 计算平均RSSI
            avg_rssi = sum(rssi for rssi, _ in rssi_band_list) / len(rssi_band_list)
            # 使用对应频段的LDPL参数
            band = bssid_band_map[union_id]
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
            torch.save(embs, "./output/embs.pt")
            predict_coors = predict_coors * pos_range + pos_min
            predict = predict_coors.cpu().tolist()[0]  # Remove batch dimension
        return predict
    else:
        return None

@app.route('/echo', methods=['POST'])
def echo():
    logging.debug("===============================================================")
    try:
        data = request.json
        bssid_rssi_map = defaultdict(list)
        bssid_band_map = {}  # 存储每个AP的频段信息
        
        # Process input data
        for entry in data:
            bssid = entry['bssid']
            freq = entry['frequency']
            rssi = entry['rssi']
            
            # 根据频率判断频段
            band = '5G' if freq >= 5000 else '2.4G'
            
            if bssid in ap_mapping and rssi > -70:
                logging.debug(f"{bssid}, {ap_mapping[bssid]}, {rssi}")
                bssid_rssi_map[ap_mapping[bssid]].append((rssi, band))
                bssid_band_map[ap_mapping[bssid]] = band

        if bssid_rssi_map:
            # Create input weights tensor
            input_weights = torch.zeros((graph_dataset.num_nodes,), dtype=torch.float32).to(device)
            
            for union_id, rssi_band_list in bssid_rssi_map.items():
                # 计算平均RSSI
                avg_rssi = sum(rssi for rssi, _ in rssi_band_list) / len(rssi_band_list)
                # 使用对应频段的LDPL参数
                band = bssid_band_map[union_id]
                print(f"{union_id}, {avg_rssi}, {band}")
                weight = 1 / (1 + LDPL(avg_rssi, band=band, mode=LDPL_MODE))
                input_weights[union_id] = weight
            
            # Normalize weights
            input_weights = input_weights.unsqueeze(0)  # Add batch dimension
            input_weights = input_weights / input_weights.sum()
            
            # Make prediction
            with torch.no_grad():
                predict_coors, _ = model(graph_dataset, input_weights)
                predict_coors = predict_coors * pos_range + pos_min
                predict = predict_coors.cpu().tolist()[0]  # Remove batch dimension
            
            logging.debug(f"Prediction: {predict}")
            return jsonify({"x": predict[0], "y": predict[1]})
        else:
            return None
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/inference', methods=['POST', 'GET'])
def inference():
    data = request.json
    print(f"data: {data}")
    return jsonify({"x": -100, "y": 100})
    logging.debug("=" * os.get_terminal_size().columns)
    try:
        data = request.json
        # Process input data
        predict = wifi_inference(data)
        if predict is None:
            return jsonify({"error": "No valid data"})
        if data['dx'] == 0 and data['dy'] == 0:
            print(f"No IMU, Observation: {predict}")
            pf.reset(predict, data['obs_noise_scale'])
            return jsonify({"x": predict[0], "y": predict[1]})
        pf.update(
            observation=(predict[0], predict[1]), 
            system_input=(data['dx'], data['dy']), 
            system_noise_scale=data['system_noise_scale'], 
            obs_noise_scale=data['obs_noise_scale']
        )
        estimate = pf.estimate.cpu().numpy().tolist()
        print(f"system_noise_scale: {data['system_noise_scale']}, obs_noise_scale: {data['obs_noise_scale']}")
        print(f"Observation: {predict}, Estimation: {estimate}")
        return jsonify({"x": estimate[0], "y": estimate[1]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    data = request.json
    print(f"data: {data}")
    return jsonify({"x": 100, "y": 100})	
    logging.debug("=" * os.get_terminal_size().columns)
    data = request.json
    predict = wifi_inference(data)
    print(f"predict: {predict}")
    if predict is None:
        return jsonify({"error": "No valid data"})
    pf.reset(predict, data['obs_noise_scale'])
    print(f"system_noise_scale: {data['system_noise_scale']}, obs_noise_scale: {data['obs_noise_scale']}")
    # logging.debug(f"Reset: {predict}")
    print(f"Reset: {predict}")
    return jsonify({"x": predict[0], "y": predict[1]})

@app.route('/sendimu', methods=['POST', 'GET'])
def sendimu():
    logging.debug("=" * os.get_terminal_size().columns)
    data = request.json
    print(f"IMU: {data}")
    step_data.push((data['timestamp'], data['yaw'], data['stride']))
    return jsonify({"success": True})

@app.route('/inference2', methods=['POST', 'GET'])
def inference2():
    logging.debug("=" * os.get_terminal_size().columns)
    global last_loc
    try:
        data = request.json
        # Process input data
        # print(f"data: {data}")
        predict = wifi_inference(data)
        print(f"predict: {predict}")
        if predict is None:
            return jsonify({"error": "No valid data"})
        timestamp = data['timestamp']
        closest_data = step_data.find_closest(timestamp, key_func=lambda x: x[0])
        print(f"closest_data: {closest_data}")
        if last_loc is None or closest_data[0] is None: #第一次预测/没有IMU数据
            last_loc = (timestamp, predict[0], predict[1])
            return jsonify({"x": predict[0], "y": predict[1]})
        elif abs(closest_data[0][0] - timestamp) > 1000: #距离上一次收到IMU数据时间过长，说明位置没有移动
            return jsonify({"x": last_loc[1], "y": last_loc[2]})
        else:
            # print(step_data.get_all())
            print(f"last_loc: {last_loc}")
            last_ts = last_loc[0]
            last_loc_index = step_data.find_closest(last_ts, key_func=lambda x: x[0])[2]
            curr_loc_index = closest_data[2]
            print(f"last_loc_index: {last_loc_index}, curr_loc_index: {curr_loc_index}")
            cnt = last_loc_index
            dx, dy = 0, 0
            while cnt != curr_loc_index:
                cnt = (cnt + 1) % step_data.size
                if cnt == curr_loc_index:
                    break
                yaw = step_data.get(cnt)[1] * math.pi / 180
                stride = step_data.get(cnt)[2]
                dx -= stride * math.cos(yaw)
                dy -= stride * math.sin(yaw)
            pf.update(
                observation=(predict[0], predict[1]), 
                system_input=(dx, dy), 
                system_noise_scale=data['system_noise_scale'], 
                obs_noise_scale=data['obs_noise_scale']
            )
            estimate = pf.estimate.cpu().numpy().tolist()
            print(f"system_noise_scale: {data['system_noise_scale']}, obs_noise_scale: {data['obs_noise_scale']}")
            print(f"Observation: {predict}, Estimation: {estimate}")
            last_loc = (timestamp, estimate[0], estimate[1])
            return jsonify({"x": estimate[0], "y": estimate[1]})
    except Exception as e:
        print(f"error: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=13344, help='服务器端口号')
    args = parser.parse_args()
    print(f"服务器运行在端口 {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
