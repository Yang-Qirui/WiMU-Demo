import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def load():
    path = "./output/embeddings"
    data = torch.load('/'.join([path, "finetune_train_embeddings.pt"]), weights_only=False)
    embs = data['embeddings']
    # Convert embeddings to numpy array if not already
    embs_np = embs.detach().cpu().numpy() if torch.is_tensor(embs) else np.array(embs)
    embs_np = embs_np.reshape(embs_np.shape[0], -1)
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embs_np)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                cmap='viridis',
                xticklabels=False, 
                yticklabels=False)
    plt.title('Embedding Similarity Heatmap')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    
    # Save the plot
    plt.savefig('./output/embeddings/similarity_heatmap.png')
    plt.close()
    
    return similarity_matrix

if __name__ == "__main__":
    load()