import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import LDPL
from config import LDPL_MODE
import torch
from torch_geometric.data import Data
from gnn import GCNEncoder, JointModel, MLPDecoder, MyMLP
from torch_geometric.nn import GAE
import seaborn as sns

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_rssi(rssi_values, num_nodes):
    # Convert RSSI values to distance-like values using LDPL
    weights = torch.zeros(num_nodes)
    for i, rssi in rssi_values.items():
        weights[int(i)] = 1 / (1 + LDPL(rssi, mode=LDPL_MODE))
    weights = weights / sum(weights)
    return weights

def compute_embeddings(data, model, weights, device):
    with torch.no_grad():
        # embeddings = model.encode(data.x, data.edge_index)
        embeddings = model.gen_emb(data, weights.to(device))
    return embeddings.flatten().cpu().numpy()

def compute_similarity_matrix(embeddings_list):
    n = len(embeddings_list)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            # Compute cosine similarity
            sim = np.dot(embeddings_list[i], embeddings_list[j]) / (
                np.linalg.norm(embeddings_list[i]) * np.linalg.norm(embeddings_list[j])
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    return similarity_matrix

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    gae = GAE(GCNEncoder(128, 128), MLPDecoder(128)).to(device)
    mlp = MyMLP(128, 2).to(device)
    model = JointModel(gae, mlp).to(device)
    model.load_state_dict(torch.load('output/pre_trained_model.pt'))
    model.eval()
    
    # Load saved graph data
    graph_data = torch.load('output/graph_dataset.pt')
    num_nodes = graph_data.x.shape[0]
    
    # Load data
    data_dir = 'data/train_json'
    location_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    graph_data = torch.load('output/graph_dataset.pt')

    for location_file in location_files:
        print(f"Processing {location_file}...")
        file_path = os.path.join(data_dir, location_file)
        data = load_json_data(file_path)
        
        # Process each timestamp
        embeddings_list = []
        timestamps = []
        
        for timestamp, rssi_values in data.items():
            # Preprocess RSSI values
            weights = preprocess_rssi(rssi_values, num_nodes)
            # Compute embedding
            embedding = compute_embeddings(graph_data, model, weights, device)
            embeddings_list.append(embedding)
            timestamps.append(timestamp)
        
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(embeddings_list)
        print(np.mean(similarity_matrix))
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=timestamps,
                   yticklabels=timestamps,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f')
        plt.title(f'Embedding Similarity Matrix for {location_file}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('output/embedding_visualization', exist_ok=True)
        plt.savefig(f'output/embedding_visualization/{location_file.replace(".json", ".png")}')
        plt.close()

def compute_location_similarities():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    gae = GAE(GCNEncoder(128, 128), MLPDecoder(128)).to(device)
    mlp = MyMLP(128, 2).to(device)
    model = JointModel(gae, mlp).to(device)
    model.load_state_dict(torch.load('output/pre_trained_model.pt'))
    model.eval()
    
    # Load saved graph data
    graph_data = torch.load('output/graph_dataset.pt')
    num_nodes = graph_data.x.shape[0]
    
    # Load data
    data_dir = 'data/train_json'
    location_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    # Store embeddings and locations
    location_embeddings = {}
    locations = []
    
    for location_file in location_files:
        print(f"Processing {location_file}...")
        file_path = os.path.join(data_dir, location_file)
        data = load_json_data(file_path)
        
        # Get the first timestamp's data
        first_timestamp = next(iter(data))
        rssi_values = data[first_timestamp]
        
        # Preprocess RSSI values
        weights = preprocess_rssi(rssi_values, num_nodes)
        
        # Compute embedding
        embedding = compute_embeddings(graph_data, model, weights, device)
        
        # Extract location from filename (assuming format like "(x,y).json")
        location_str = location_file.replace('.json', '')  # Remove .json
        location_str = location_str.strip('()')  # Remove parentheses
        x, y = map(float, location_str.split(','))  # Split by comma and convert to float
        locations.append((x, y))
        location_embeddings[location_file] = embedding
    
    # Compute similarity matrix and distance matrix
    n = len(location_files)
    similarity_matrix = np.zeros((n, n))
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Compute embedding similarity
            sim = np.dot(location_embeddings[location_files[i]], 
                        location_embeddings[location_files[j]]) / (
                np.linalg.norm(location_embeddings[location_files[i]]) * 
                np.linalg.norm(location_embeddings[location_files[j]])
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            # Compute physical distance
            dist = np.sqrt((locations[i][0] - locations[j][0])**2 + 
                          (locations[i][1] - locations[j][1])**2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Flatten matrices for scatter plot
    similarities = similarity_matrix[np.triu_indices(n, k=1)]
    distances = distance_matrix[np.triu_indices(n, k=1)]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, similarities, alpha=0.6)
    plt.xlabel('Physical Distance (m)')
    plt.ylabel('Embedding Similarity')
    plt.title('Location Distance vs Embedding Similarity')
    
    # Add trend line
    z = np.polyfit(distances, similarities, 1)
    p = np.poly1d(z)
    plt.plot(distances, p(distances), "r--", alpha=0.5)
    
    # Save plot
    os.makedirs('output/embedding_visualization', exist_ok=True)
    plt.savefig('output/embedding_visualization/location_distance_vs_similarity.png')
    plt.close()
    
    # Print correlation coefficient
    correlation = np.corrcoef(distances, similarities)[0, 1]
    print(f"Correlation between distance and similarity: {correlation:.3f}")

if __name__ == '__main__':
    main()
    compute_location_similarities() 