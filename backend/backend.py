import os
import random
import shutil
import requests
import secrets
import logging
import json
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify
import database
from threading import Thread
from filelock import FileLock
import inference_service
import re

# ========== 配置 ==========
# EMQX HTTP API
EMQX_HOST = "127.0.0.1"
EMQX_API_PORT = 18083
EMQX_API_USER = "2cff38a527697f0c"
EMQX_API_PASS = "9BM9CwTu5c9BTM6U6CJGQngInpJwOAwn3FLHHnGEsj9AfVE"
EMQX_API_URL = f"http://{EMQX_HOST}:{EMQX_API_PORT}"
EMQX_CREATE_USER_ENDPOINT = f"{EMQX_API_URL}/api/v5/authentication/password_based%3Abuilt_in_database/users"

# MQTT Broker
MQTT_BROKER_HOST = EMQX_HOST
MQTT_BROKER_PORT = 1883
SCRIPT_MQTT_USER = "wands"
SCRIPT_MQTT_PASS = "wands123"
PUBLISHER_CLIENT_ID = "superuser001"
PUBLISHER_CLIENT_NAME = "super_publisher"
REQUEST_TOPICS = ["devices/inference", "devices/ack"]

# Cache Folder
CACHE_FOLDER = "cache"
DEVICE_STATUS_FILE = os.path.join(CACHE_FOLDER, 'device_status.json')
DEVICE_NAME_FILE = os.path.join(CACHE_FOLDER, 'device_names.json')
DEVICE_PATHS_FOLDER = os.path.join(CACHE_FOLDER, 'device_paths')
os.makedirs(DEVICE_PATHS_FOLDER, exist_ok=True)

# ========== 日志 ==========
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 如果日志消息中包含 "GET /device_status"，则返回 False
        # record.getMessage() 会获取完整的日志字符串
        return "GET /device_status" not in record.getMessage()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HealthCheckFilter())

# ========== Flask ==========
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
app.config['CACHE_FOLDER'] = CACHE_FOLDER
BATCH_META_FOLDER = os.path.join(CACHE_FOLDER, 'batch_meta')
os.makedirs(BATCH_META_FOLDER, exist_ok=True)

# ========== MQTT 客户端 ==========
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=PUBLISHER_CLIENT_ID)
mqtt_client.username_pw_set(SCRIPT_MQTT_USER, SCRIPT_MQTT_PASS)

# 业务函数映射
def inference(device_id, wifi_list, imu_offset, sys_noise, obs_noise):
    """调用inference_service进行推理"""
    try:
        # 转换WiFi数据格式
        wifi_entries = []
        for wifi_record in wifi_list:
            # 解析WiFi记录格式: "timestamp ssid bssid channel rssi"
            regex_wifi = r"^(\d+)\s+(.*?)\s+((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})\s+(\d+)\s+(-?\d+)$"
            match_wifi = re.match(regex_wifi, wifi_record)
            if match_wifi:
                timestamp, ssid, bssid, channel, rssi = match_wifi.groups()
                # print(timestamp, ssid, bssid, channel, rssi)
                wifi_entries.append({
                    "bssid": bssid,
                    "ssid": ssid,
                    "frequency": int(channel),
                    "rssi": int(rssi)
                })
        
        # 准备推理数据
        inference_data = {
            "device_id": device_id,
            "wifi_entries": wifi_entries,
            "dx": imu_offset.get('first', 0) if imu_offset else 0,
            "dy": imu_offset.get('second', 0) if imu_offset else 0,
            "system_noise": sys_noise,
            "obs_noise": obs_noise
        }
        
        # 调用inference_service的推理函数
        result = inference_service.inference_with_pf(inference_data)
        
        if "error" in result:
            logging.error(f"Inference error: {result['error']}")
            return {"x": 0, "y": 0, "confidence": 0}
        
        # 保存设备路径
        x_coord = result["x"]
        y_coord = result["y"]
        confidence = 5  # 默认置信度
        save_device_path(device_id, x_coord, y_coord, confidence=confidence)
        logging.info(f"Saved path for device {device_id}: ({x_coord}, {y_coord})")
        
        return {
            "x": x_coord,
            "y": y_coord, 
            "confidence": confidence
        }
        
    except Exception as e:
        logging.error(f"Error in inference function: {e}")
        return {"x": 0, "y": 0, "confidence": 0}

def load_device_status():
    lock_path = DEVICE_STATUS_FILE + '.lock'
    with FileLock(lock_path):
        if not os.path.exists(DEVICE_STATUS_FILE):
            return {}
        with open(DEVICE_STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_device_status(status):
    lock_path = DEVICE_STATUS_FILE + '.lock'
    with FileLock(lock_path):
        with open(DEVICE_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

def load_device_names():
    lock_path = DEVICE_NAME_FILE + '.lock'
    with FileLock(lock_path):
        if not os.path.exists(DEVICE_NAME_FILE):
            return {}
        with open(DEVICE_NAME_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_device_names(names):
    lock_path = DEVICE_NAME_FILE + '.lock'
    with FileLock(lock_path):
        with open(DEVICE_NAME_FILE, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)

def load_device_paths(device_id):
    """加载设备的路径历史"""
    device_path_file = os.path.join(DEVICE_PATHS_FOLDER, f'{device_id}_paths.json')
    lock_path = device_path_file + '.lock'
    with FileLock(lock_path):
        if not os.path.exists(device_path_file):
            return []
        with open(device_path_file, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_device_path(device_id, x, y, timestamp=None, confidence=None):
    """保存设备的路径点"""
    import time
    if timestamp is None:
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
    
    device_path_file = os.path.join(DEVICE_PATHS_FOLDER, f'{device_id}_paths.json')
    lock_path = device_path_file + '.lock'
    
    with FileLock(lock_path):
        # 加载现有路径
        paths = []
        if os.path.exists(device_path_file):
            with open(device_path_file, 'r', encoding='utf-8') as f:
                paths = json.load(f)
        
        # 添加新路径点
        path_point = {
            "timestamp": timestamp,
            "x": float(x),
            "y": float(y),
            "confidence": confidence if confidence is not None else 0
        }
        paths.append(path_point)
        
        # 保存回文件
        with open(device_path_file, 'w', encoding='utf-8') as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logging.info("Successfully connected to MQTT Broker!")
        for topic in REQUEST_TOPICS:
            client.subscribe(topic)
            logging.info(f"Subscribed to topic: {topic}")
    else:
        logging.error(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    # TODO: Currently, we only support data related to inference to be uploaded to the backend.
    logging.info(f"Received message on topic '{msg.topic}'")
    if msg.topic == "devices/ack":
        try:
            data = json.loads(msg.payload.decode())
            # Kotlin端ack消息格式: {"deviceId": ..., "ackInfo": ...}
            device_id = data.get("deviceId")
            ack_info = data.get("ackInfo")
            if device_id is not None and ack_info is not None:
                status = load_device_status()
                if device_id not in status:
                    status[device_id] = {"is_sampling": 0, "is_inference": 0}
                if ack_info == "sample_on":
                    status[device_id]["is_sampling"] = 1
                elif ack_info == "sample_off":
                    status[device_id]["is_sampling"] = 0
                elif ack_info == "inference_on":
                    status[device_id]["is_inference"] = 1
                elif ack_info == "inference_off":
                    status[device_id]["is_inference"] = 0
                save_device_status(status)
                logging.info(f"Device {device_id} status updated: {status[device_id]}")
        except Exception as e:
            logging.error(f"Error handling ack message: {e}")
    if msg.topic == "devices/inference":
        try:
            data = json.loads(msg.payload.decode())
            # print(data)
            device_id = data.get("deviceId")
            wifi_list = data.get("wifiList")
            imu_offset = data.get("imuOffset")
            sys_noise = data.get("sysNoise")
            obs_noise = data.get("obsNoise")
            if not all([device_id, wifi_list, sys_noise, obs_noise]):
                logging.error("Message missing 'device_id', 'wifiList', 'sysNoise', or 'obsNoise'. Ignoring.")
                return
            result = inference(device_id, wifi_list, imu_offset, sys_noise, obs_noise)
            result["command"] = "inference_result"
            result["sender"] = "mqtt_manager"
            send_command_to_device(device_id, "DIRECT_PUSH", result)
            # client.publish(f"devices/{device_id}/commands", response_payload, qos=1)
        except Exception as e:
            logging.error(f"Error handling MQTT message: {e}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
names = load_device_names()
names[PUBLISHER_CLIENT_ID] = PUBLISHER_CLIENT_NAME
save_device_names(names)
status = load_device_status()
status[PUBLISHER_CLIENT_ID] = {"is_sampling": -1, "is_inference": -1} # -1 表示不支持该功能
save_device_status(status)
mqtt_client.loop_start()

# ========== 工具函数 ==========
def get_all_connected_device_ids():
    api_url = f"http://{EMQX_HOST}:{EMQX_API_PORT}/api/v5/clients"
    all_client_ids = []
    page = 1
    limit = 100
    while True:
        params = {"page": page, "limit": limit}
        try:
            response = requests.get(api_url, params=params, auth=(EMQX_API_USER, EMQX_API_PASS), timeout=10)
            if response.status_code != 200:
                logging.error(f"Failed to fetch clients. Status code: {response.status_code}")
                return None
            data = response.json()
            clients = data.get("data", [])
            if not clients:
                break
            for client in clients:
                client_id = client.get("clientid")
                if client_id:
                    all_client_ids.append(client_id)
            page += 1
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling EMQX API: {e}")
            return None
    return all_client_ids

def send_command_to_device(device_id: str, command: str, data: dict):
    topic = f"devices/{device_id}/commands"
    payload = json.dumps({"command": command, "data": data})
    try:
        logging.info(f"Publishing to topic: {topic}")
        mqtt_client.publish(topic, payload, qos=1)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

def load_batch_meta(batch_id):
    meta_path = os.path.join(BATCH_META_FOLDER, f"{batch_id}.json")
    lock_path = meta_path + ".lock"
    with FileLock(lock_path):
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_batch_meta(batch_id, meta):
    meta_path = os.path.join(BATCH_META_FOLDER, f"{batch_id}.json")
    lock_path = meta_path + ".lock"
    with FileLock(lock_path):
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

def delete_batch_meta(batch_id):
    meta_path = os.path.join(BATCH_META_FOLDER, f"{batch_id}.json")
    lock_path = meta_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(meta_path):
            os.remove(meta_path)

# ========== Flask 路由 ==========
@app.route('/register', methods=['POST'])
def register_device():
    if not request.is_json:
        logging.warning("Request is not JSON")
        return jsonify({"error": "Invalid request: Content-Type must be application/json"}), 400
    data = request.get_json()
    device_id = data.get('deviceId')
    device_name = data.get('deviceName')
    if not device_id:
        logging.warning("deviceId is missing from request")
        return jsonify({"error": "Missing 'deviceId' in request body"}), 400
    # 保存deviceId和deviceName映射
    if device_name:
        names = load_device_names()
        names[device_id] = device_name
        save_device_names(names)
    # 初始化设备状态
    status = load_device_status()
    if device_id not in status:
        status[device_id] = {"is_sampling": 0, "is_inference": 0}
        save_device_status(status)
    logging.info(f"Received registration request for deviceId: {device_id}, deviceName: {device_name}")
    mqtt_username = device_id
    mqtt_password = secrets.token_hex(16)
    user_payload = {
        "password": mqtt_password,
        "user_id": device_id,
        "is_superuser": False
    }
    try:
        response = requests.post(
            EMQX_CREATE_USER_ENDPOINT,
            json=user_payload,
            auth=(EMQX_API_USER, EMQX_API_PASS),
            timeout=5
        )
        if response.status_code == 201:
            logging.info(f"Successfully created new MQTT user: {mqtt_username}")
        elif response.status_code == 409:
            logging.warning(f"MQTT user '{mqtt_username}' already exists. A new password will be provided.")
            update_url = f"{EMQX_CREATE_USER_ENDPOINT}/{mqtt_username}"
            update_response = requests.put(
                update_url,
                json={"password": mqtt_password},
                auth=(EMQX_API_USER, EMQX_API_PASS)
            )
            if update_response.status_code != 200:
                 logging.error(f"Failed to update password for user {mqtt_username}. Status: {update_response.status_code}, Body: {update_response.text}")
                 return jsonify({"error": "Failed to update existing user credentials"}), 500
            logging.info(f"Successfully updated password for existing user: {mqtt_username}")
        else:
            logging.error(f"Failed to create MQTT user. EMQX API returned status {response.status_code}. Body: {response.text}")
            return jsonify({"error": "Failed to configure MQTT credentials"}), 500
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling EMQX API: {e}")
        return jsonify({"error": "Could not connect to MQTT management service"}), 503
    response_data = {
        "mqttUsername": mqtt_username,
        "mqttPassword": mqtt_password,
    }
    logging.info(f"Returning credentials for user: {mqtt_username}, deviceName: {device_name}")
    return jsonify(response_data), 200

@app.route('/devices', methods=['GET'])
def get_devices():
    devices = get_all_connected_device_ids()
    names = load_device_names()
    result = []
    print(devices)
    for device_id in devices:
        result.append({
            "deviceId": device_id,
            "deviceName": names.get(device_id, "")
        })
    return jsonify(result), 200

@app.route('/start_sample', methods=['POST'])
def start_sample():
    data = request.get_json()
    if not data or 'target_device_id' not in data:
        return jsonify({"error": "Missing target_device_id in request body"}), 400
    target_device_id = data['target_device_id']
    send_command_to_device(
        target_device_id, 
        "DIRECT_PUSH", {
            "sender": "mqtt_manager",
            "command": "start_sample",
        }
    )
    return jsonify({"message": "开始采样"}), 200

@app.route('/end_sample', methods=['POST'])
def end_sample():
    data = request.get_json()
    if not data or 'target_device_id' not in data:
        return jsonify({"error": "Missing target_device_id in request body"}), 400
    target_device_id = data['target_device_id']
    send_command_to_device(
        target_device_id, 
        "DIRECT_PUSH", {
            "sender": "mqtt_manager",
            "command": "end_sample",
        }
    )
    return jsonify({"message": "结束采样"}), 200

@app.route('/start_inference', methods=['POST'])
def start_inference():
    data = request.get_json()
    if not data or 'target_device_id' not in data:
        return jsonify({"error": "Missing target_device_id in request body"}), 400
    target_device_id = data['target_device_id']
    send_command_to_device(
        target_device_id, 
        "DIRECT_PUSH", {
            "sender": "mqtt_manager",
            "command": "start_inference",
        }
    )
    return jsonify({"message": "开始推理"}), 200

@app.route('/end_inference', methods=['POST'])
def end_inference():
    data = request.get_json()
    if not data or 'target_device_id' not in data:
        return jsonify({"error": "Missing target_device_id in request body"}), 400
    target_device_id = data['target_device_id']
    send_command_to_device(
        target_device_id, 
        "DIRECT_PUSH", {
            "sender": "mqtt_manager",
            "command": "end_inference",
        }
    )
    return jsonify({"message": "结束推理"}), 200

@app.route('/device_status', methods=['GET'])
def get_device_status():
    status = load_device_status()
    return jsonify(status)

@app.route('/device_paths', methods=['GET'])
def get_device_paths():
    """获取所有设备的路径历史"""
    device_id = request.args.get('device_id')
    
    if device_id:
        # 获取单个设备的路径
        paths = load_device_paths(device_id)
        return jsonify({device_id: paths})
    else:
        # 获取所有设备的路径
        all_paths = {}
        try:
            for filename in os.listdir(DEVICE_PATHS_FOLDER):
                if filename.endswith('_paths.json'):
                    device_id = filename[:-11]  # 移除 '_paths.json'
                    all_paths[device_id] = load_device_paths(device_id)
        except Exception as e:
            logging.error(f"Error loading device paths: {e}")
            return jsonify({"error": str(e)}), 500
        
        return jsonify(all_paths)

@app.route('/clear_device_paths', methods=['POST'])
def clear_device_paths():
    """清空指定设备的路径历史"""
    try:
        data = request.get_json()
        device_id = data.get('device_id') if data else None
        
        if not device_id:
            return jsonify({"error": "Missing device_id"}), 400
        
        device_path_file = os.path.join(DEVICE_PATHS_FOLDER, f'{device_id}_paths.json')
        lock_path = device_path_file + '.lock'
        
        with FileLock(lock_path):
            if os.path.exists(device_path_file):
                os.remove(device_path_file)
                logging.info(f"Cleared path history for device {device_id}")
                return jsonify({"message": f"Path history cleared for device {device_id}"})
            else:
                return jsonify({"message": f"No path history found for device {device_id}"})
                
    except Exception as e:
        logging.error(f"Error clearing device paths: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/inference_health', methods=['GET'])
def get_inference_health():
    """获取推理服务健康状态"""
    health_status = inference_service.get_health_status()
    return jsonify(health_status)

@app.route('/particle_filters', methods=['GET'])
def get_particle_filters():
    """获取所有粒子滤波器状态"""
    filters_status = inference_service.get_particle_filter_status()
    return jsonify(filters_status)

@app.route('/reset_particle_filter', methods=['POST'])
def reset_particle_filter():
    """重置指定设备的粒子滤波器"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({"error": "Missing device_id"}), 400
        
        result = inference_service.reset_particle_filter(device_id, data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in reset_particle_filter endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

def process_batch(batch_id):
    with app.app_context():
        batch_path = os.path.join(app.config['CACHE_FOLDER'], batch_id)
        upload_meta = load_batch_meta(batch_id)
        print(f"Processing batch: {batch_id}")
        files_dict = {
            "euler.txt": os.path.join(batch_path, "euler.txt"),
            "step.txt": os.path.join(batch_path, "step.txt"),
            "wifi.txt": os.path.join(batch_path, "wifi.txt")
        }
        try:
            database.process_and_save_data(upload_meta["device_id"], upload_meta["path_name"], upload_meta["data_type"], files_dict)
        except Exception as e:
            print(f"Error processing batch: {e}")
        finally:
            delete_batch_meta(batch_id)
            shutil.rmtree(batch_path)
            print(f"Batch {batch_id} processed successfully")

@app.route('/upload', methods=['POST'])
def upload_data():
    batch_id = request.form.get('batch_id')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    batch_path = os.path.join(app.config['CACHE_FOLDER'], batch_id)
    file_path = os.path.join(batch_path, file.filename)
    file.save(file_path)
    print(f"[{batch_id}] {file.filename} uploaded successfully")
    # 更新本地meta
    meta = load_batch_meta(batch_id)
    if meta is None:
        return jsonify({"error": "Batch meta not found"}), 400
    if file.filename not in meta["received_files"]:
        meta["received_files"].append(file.filename)
        save_batch_meta(batch_id, meta)
    updated_batch = meta
    if updated_batch and len(updated_batch.get("received_files", [])) == int(updated_batch.get("total_files", -1)):
        print(f"Updated batch: {updated_batch}")
        processing_thread = Thread(target=process_batch, args=(batch_id,))
        processing_thread.start()

    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/upload_meta', methods=['POST'])
def upload_meta():
    data = request.get_json()
    batch_id = data.get('batch_id')
    total_files = data.get('total_files')
    device_id = data.get('device_id')
    path_name = data.get('path_name')
    data_type = data.get('data_type')
    if not all([batch_id, total_files, device_id, path_name, data_type]):
        return jsonify({"error": "Missing required fields"}), 400
    batch_doc = {
        "batch_id": batch_id,
        "total_files": total_files,
        "device_id": device_id,
        "path_name": path_name,
        "data_type": data_type,
        "received_files": []
    }
    try:
        save_batch_meta(batch_id, batch_doc)
        print(f"Meta saved successfully. Batch ID: {batch_id}")
    except Exception as e:
        print(f"Error saving meta: {e}")
    batch_path = os.path.join(app.config['CACHE_FOLDER'], batch_id)
    os.makedirs(batch_path, exist_ok=True)
    return jsonify({"message": "Meta uploaded successfully"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
