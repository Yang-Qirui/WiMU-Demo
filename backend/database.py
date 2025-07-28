import re
import datetime
import json
import os

# MongoDB相关已移除
# client = MongoClient("mongodb://localhost:27017/")
# db = client["wimu_database"]
# recordings_collection = db["recordings"]
# upload_meta_collection = db["upload_meta"]

def process_and_save_data(device_id, path_name, data_type, files_dict, save_dir="./data_json"):
    os.makedirs(f"{save_dir}/{device_id}/{data_type}", exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = now.strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{timestamp_str}.json"
    filepath = os.path.join(f"{save_dir}/{device_id}/{data_type}", filename)

    document = {
        "device_id": device_id,
        "path_name": path_name,
        "data_type": data_type,
        "create_at": now.isoformat(),
        "euler_data": [],
        "step_data": [],
        "wifi_data": []
    }
    if "euler.txt" in files_dict:
        with open(files_dict["euler.txt"], "r") as f:
            for line in f:
                timestamp, yaw, pitch, roll = line.strip().split(" ")
                document["euler_data"].append({
                    "timestamp": timestamp,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll
                })
    if "step.txt" in files_dict:
        with open(files_dict["step.txt"], "r") as f:
            for line in f:
                timestamp = line.strip().split(",")
                document["step_data"].append({
                    "timestamp": timestamp,
                })
    if "wifi.txt" in files_dict:
        regex_xy = r"^(-?\d+\.?\d*),(-?\d+\.?\d*)$"
        regex_wifi = r"^(\d+)\s+(.*?)\s+((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})\s+(\d+)\s+(-?\d+)$"
        with open(files_dict["wifi.txt"], "r") as f:
            for line in f:
                match_xy = re.match(regex_xy, line)
                match_wifi = re.match(regex_wifi, line)
                if match_xy:
                    x, y = match_xy.groups()
                    document["groundtruth"] = {
                        "x": float(x),
                        "y": float(y)
                    }
                elif match_wifi:
                    timestamp, ssid, mac, rssi, channel = match_wifi.groups()
                    document["wifi_data"].append({
                        "timestamp": timestamp,
                        "ssid": ssid,
                        "bssid": mac,
                        "channel": channel,
                        "rssi": rssi,
                    })
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        print(f"Data saved successfully. File: {filepath}")
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == "__main__":
    process_and_save_data("test_device", "test_path", "test_data", {"euler.txt": "test_euler.txt", "step.txt": "test_step.txt", "wifi.txt": "test_wifi.txt"})
                
