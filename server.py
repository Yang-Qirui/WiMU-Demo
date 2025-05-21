import logging.config
from flask import Flask, request, jsonify
from gnn import MyMLP, GCNEncoder, JointModel
import torch
import json
from utils import LDPL
import logging
from collections import defaultdict
from torch_geometric.nn import GAE
from torch_geometric.data import Data
import os
from gnn import *
from particle_filter import TorchParticleFilter
from config import *
from datetime import datetime

app = Flask(__name__)
datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(level=logging.DEBUG, filename=f"./log/server_{datetime}.log")
# logging.basicConfig(level=logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load graph dataset
graph_dataset = torch.load("./output/graph_dataset.pt").to(device)

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

with open("./output/data_process/ap_unions.json", 'r') as f:
    ap_mapping = json.load(f)

pf = TorchParticleFilter(num_particles=1000, device=device)

def wifi_inference(data):
    bssid_rssi_map = defaultdict(list)
    bssid_band_map = {}  # 存储每个AP的频段信息
    
    for entry in data['wifiEntries']:
        bssid = entry['bssid']
        freq = entry['frequency']
        rssi = entry['rssi']
        # logging.debug(f"{bssid}, {freq}, {rssi}")
        
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
            weight = 1 / (1 + LDPL(avg_rssi, band=band))
            input_weights[union_id] = weight
            logging.debug(f"{union_id}, {avg_rssi}, {band}, {weight}")
        
        # Normalize weights
        input_weights = input_weights.unsqueeze(0)  # Add batch dimension
        input_weights = input_weights / input_weights.sum()
        
        # Make prediction
        with torch.no_grad():
            predict_coors, _ = model(graph_dataset, input_weights)
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
                logging.debug(f"{union_id}, {avg_rssi}, {band}")
                weight = 1 / (1 + LDPL(avg_rssi, band=band))
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

@app.route('/inference', methods=['POST'])
def inference():
    logging.debug("=" * os.get_terminal_size().columns)
    try:
        data = request.json
        # Process input data
        predict = wifi_inference(data)
        if predict is None:
            return jsonify({"error": "No valid data"})
        if data['dx'] == 0 and data['dy'] == 0:
            logging.debug(f"No IMU, Observation: {predict}")
            pf.reset(predict, data['obs_noise_scale'])
            return jsonify({"x": predict[0], "y": predict[1]})
        pf.update(
            observation=(predict[0], predict[1]), 
            system_input=(data['dx'], data['dy']), 
            system_noise_scale=data['system_noise_scale'], 
            obs_noise_scale=data['obs_noise_scale']
        )
        estimate = pf.estimate.cpu().numpy().tolist()
        logging.debug(f"system_noise_scale: {data['system_noise_scale']}, obs_noise_scale: {data['obs_noise_scale']}")
        logging.debug(f"Observation: {predict}, Estimation: {estimate}")
        return jsonify({"x": estimate[0], "y": estimate[1]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/reset', methods=['POST'])
def reset():
    logging.debug("=" * os.get_terminal_size().columns)
    data = request.json
    predict = wifi_inference(data)
    if predict is None:
        return jsonify({"error": "No valid data"})
    pf.reset(predict, data['obs_noise_scale'])
    logging.debug(f"system_noise_scale: {data['system_noise_scale']}, obs_noise_scale: {data['obs_noise_scale']}")
    logging.debug(f"Reset: {predict}")
    return jsonify({"x": predict[0], "y": predict[1]})

if __name__ == '__main__':
    print("Server is running on port 13344")
    app.run(host='0.0.0.0', port=13344, debug=True)