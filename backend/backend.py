import random
import requests
import secrets
import logging
import json
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify

# ========== 配置 ==========
# EMQX HTTP API
EMQX_HOST = "limcpu1.cse.ust.hk"
EMQX_API_PORT = 18083
EMQX_API_USER = "2cff38a527697f0c"
EMQX_API_PASS = "XYG9Ajk9BpOaFTIHyC9BcDCeePDtIVFZasZMZ2ZlRp1C5M"
EMQX_API_URL = f"http://{EMQX_HOST}:7860/mqttpanel"
EMQX_CREATE_USER_ENDPOINT = f"{EMQX_API_URL}/api/v5/authentication/password_based%3Abuilt_in_database/users"

# MQTT Broker
MQTT_BROKER_HOST = EMQX_HOST
MQTT_BROKER_PORT = 1883
SCRIPT_MQTT_USER = "wands"
SCRIPT_MQTT_PASS = "wands123"
PUBLISHER_CLIENT_ID = f"python-direct-push-script-01"
REQUEST_TOPIC = "devices/inference"

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== Flask ==========
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')

# ========== MQTT 客户端 ==========
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=PUBLISHER_CLIENT_ID)
mqtt_client.username_pw_set(SCRIPT_MQTT_USER, SCRIPT_MQTT_PASS)

# 业务函数映射
def inference(wifi_list, imu_offset, sys_noise, obs_noise):
    return {"x": 100 * random.random(), "y": 200 * random.random(), "confidence": 5}

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logging.info("Successfully connected to MQTT Broker!")
        client.subscribe(REQUEST_TOPIC)
        logging.info(f"Subscribed to topic: {REQUEST_TOPIC}")
    else:
        logging.error(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    # TODO: Currently, we only support data related to inference to be uploaded to the backend.
    logging.info(f"Received message on topic '{msg.topic}'")
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
        result = inference(wifi_list, imu_offset, sys_noise, obs_noise)
        result["command"] = "inference_result"
        result["sender"] = "mqtt_manager"
        send_command_to_device(device_id, "DIRECT_PUSH", result)
        # client.publish(f"devices/{device_id}/commands", response_payload, qos=1)
    except Exception as e:
        logging.error(f"Error handling MQTT message: {e}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
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

# ========== Flask 路由 ==========
@app.route('/register', methods=['POST'])
def register_device():
    if not request.is_json:
        logging.warning("Request is not JSON")
        return jsonify({"error": "Invalid request: Content-Type must be application/json"}), 400
    data = request.get_json()
    device_id = data.get('deviceId')
    if not device_id:
        logging.warning("deviceId is missing from request")
        return jsonify({"error": "Missing 'deviceId' in request body"}), 400
    logging.info(f"Received registration request for deviceId: {device_id}")
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
        "mqttPassword": mqtt_password
    }
    logging.info(f"Returning credentials for user: {mqtt_username}")
    return jsonify(response_data), 200

@app.route('/devices', methods=['GET'])
def get_devices():
    devices = get_all_connected_device_ids()
    return jsonify(devices), 200

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

@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
