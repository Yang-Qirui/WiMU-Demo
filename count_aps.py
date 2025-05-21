import json

def count_aps(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return len(data)

# Count distinct APs
distinct_count = count_aps('output/data_process/distinct_aps.json')
print(f"Number of distinct APs: {distinct_count}")

# Count merged APs
merged_count = count_aps('output/data_process/merged_aps.json')
print(f"Number of merged APs: {merged_count}") 