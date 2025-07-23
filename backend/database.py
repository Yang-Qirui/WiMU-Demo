import re
from pymongo import MongoClient
import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["wimu_database"]
recordings_collection = db["recordings"]
upload_meta_collection = db["upload_meta"]

def process_and_save_data(device_id, path_name, data_type, files_dict):

    document = {
        "device_id": device_id,
        "path_name": path_name,
        "data_type": data_type,
        "create_at": datetime.datetime.now(datetime.timezone.utc),
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
        regex = r"^(\d+)\s+(.*?)\s+((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})\s+(\d+)\s+(-?\d+)$"
        with open(files_dict["wifi.txt"], "r") as f:
            for line in f:
                match = re.match(regex, line)
                if match:
                    timestamp, ssid, mac, rssi, channel = match.groups()
                    document["wifi_data"].append({
                        "timestamp": timestamp,
                        "ssid": ssid,
                        "bssid": mac,
                        "channel": channel,
                        "rssi": rssi,
                    })
    try:
        result = recordings_collection.insert_one(document)
        print(f"Data saved successfully. ID: {result.inserted_id}")
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == "__main__":
    process_and_save_data("test_device", "test_path", "test_data", {"euler.txt": "test_euler.txt", "step.txt": "test_step.txt", "wifi.txt": "test_wifi.txt"})
                
