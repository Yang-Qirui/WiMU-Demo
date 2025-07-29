import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torch_geometric.nn import GCNConv, InnerProductDecoder, VGAE, GATConv, GCN, GraphSAGE
from collections import defaultdict
import random
from sklearn.cluster import DBSCAN


class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 512, normalize=True)
        self.conv2 = GCNConv(512, 512, normalize=True)
        # self.conv3 = GCNConv(512, 512, normalize=True)
        
        self.conv_mu = GCNConv(512, out_channels, normalize=True)
        self.conv_logvar = GCNConv(512, out_channels, normalize=True)

    def forward(self, x, edge_index, weight):
        x = self.conv1(x, edge_index, weight).relu()
        x = self.conv2(x, edge_index, weight).relu()
        # x = self.conv3(x, edge_index, weight).relu()
        return self.conv_mu(x, edge_index, weight), self.conv_logvar(x, edge_index, weight)
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layer=None):
        super(GCNEncoder, self).__init__()
        if hidden_layer == None:
            hidden_layer = 8 * in_channels
        # self.conv1 = GATConv(in_channels, hidden_layer, heads=4, dropout=0.2)
        # self.conv2 = GATConv(hidden_layer * 4, out_channels, heads=1, dropout=0.2)

        self.conv1 = GCNConv(in_channels, hidden_layer)
        # self.conv2 = GCNConv(hidden_layer, hidden_layer)
        self.conv3 = GCNConv(hidden_layer, out_channels)

    def forward(self, x, edge_index, edge_weight):
        out1 = torch.relu(self.conv1(x, edge_index, edge_weight))
        # out2 = torch.relu(self.conv2(out1, edge_index, edge_weight))
        out3 = self.conv3(out1, edge_index, edge_weight)
        return out3
    
# 改进的多层解码器
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(2*latent_dim, 256)  # 拼接节点特征
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, z):
        row = z.unsqueeze(1).repeat(1, z.size(0), 1)
        col = z.unsqueeze(0).repeat(z.size(0), 1, 1)
        pair = torch.cat([row, col], dim=-1)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(pair)))).squeeze()
        # return torch.clamp(self.fc2(torch.relu(self.fc1(pair))), min=0, max=1).squeeze()

class MyMLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, 2 * in_channel)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(2 * in_channel, 2 * in_channel)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(2 * in_channel, in_channel)
        self.fc4 = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        fc1 = F.leaky_relu(self.fc1(x))
        dropout1 = self.dropout1(fc1)
        fc2 = F.leaky_relu(self.fc2(dropout1))
        dropout2 = self.dropout2(fc2)
        fc3 = F.leaky_relu(self.fc3(dropout2))
        fc4 = self.fc4(fc3)
        return torch.sigmoid(fc4)  # Use sigmoid to ensure output in [0,1]
    
class JointModel(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(JointModel, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        # Initialize learnable weights for each AP
        # self.ap_weights = torch.nn.Parameter(torch.ones(num_aps) / num_aps)
     
    def encode(self, graph):
        return self.gnn.encode(graph.x, graph.edge_index, graph.edge_attr)
        
    def decode(self, embs):
        return self.gnn.decoder(embs)
        
    def kl_loss(self):
        return self.gnn.kl_loss()
    
    def gen_emb(self, graph, weights):
        embs = self.encode(graph)
        # Apply learnable weights to the embeddings
        # weighted_embs = embs * self.ap_weights.unsqueeze(1)
        return torch.matmul(embs.T, weights.unsqueeze(-1))
        
    def forward(self, graph, weights):
        embs = self.encode(graph)
        # Apply learnable weights to the embeddings
        # weighted_embs = embs * self.ap_weights.unsqueeze(1)
        x = torch.matmul(embs.T, weights.unsqueeze(-1))
        x = self.mlp(x.squeeze(-1))
        return x, self.decode(embs)
    
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, training_embs, training_gts):
        '''
            training_embs: N * hidden_dim
            training_gts: corresponding groundtruth: N * 2
        '''
        self.device = training_embs.device
        # gt_filtering_dict = defaultdict(list)
        # for idx, gt in enumerate(training_gts):
        #     gt_tuple = tuple(gt.tolist())
        #     gt_filtering_dict[gt_tuple].append((idx, training_embs[idx]))
        # filtered_data = []
        # for gt,  in gt_filtering_dict.items():
        #     # TODO: filter noisy embeddings
        #     idx = [v[0] for v in values]
        #     tensors = [v[-1] for v in values]
        #     X = torch.stack(tensors).cpu().numpy()
        #     dbscan = DBSCAN(eps=0.5, min_samples=5)
        #     labels = dbscan.fit_predict(X)
        #     filtered_data += [i for i, lbl in zip(idx, labels) if lbl != -1]
        # self.embs = training_embs[filtered_data]
        # self.gts = training_gts[filtered_data]
        self.embs = training_embs
        self.gts = training_gts
        self.gt_to_indices = defaultdict(list)
        for idx, gt in enumerate(self.gts):
            gt_tuple = tuple(gt.tolist())
            self.gt_to_indices[gt_tuple].append(idx)
    
    def __len__(self):
        return len(self.embs)
    
    def __getitem__(self, idx):
        anchor_emb = self.embs[idx].to(self.device)
        anchor_gt = tuple(self.gts[idx].tolist())
        
        positive_embs = self.gt_to_indices[anchor_gt].copy()
        positive_embs.remove(idx)
        if positive_embs:
            positive_idx = random.choice(positive_embs)
            positive_emb = self.embs[positive_idx]
        else:
            positive_emb = anchor_emb.clone()
            
        other_gts = [gt for gt in self.gt_to_indices if gt != anchor_gt]
        if not other_gts:
            negative_emb = anchor_emb.clone()
        else:
            negative_gt = random.choice(other_gts)
            negative_idx = random.choice(self.gt_to_indices[negative_gt])
            negative_emb = self.embs[negative_idx]
        
        return torch.tensor(anchor_gt), anchor_emb, positive_emb, negative_emb
            
class CorrelationLoss(nn.Module):
    def __init__(self, neg_coor=False):
        super(CorrelationLoss, self).__init__()
        self.neg_coor = neg_coor

    def forward(self, embs, distances, id_mask):
        """
        Args:
            embs: (batch_size, hidden_dim)
            distances: (batch_size, batch_size) distance matrix
            id_mask: (batch_size, batch_size) mask indicating which pairs are from the same path
            neg_coor: whether to use negative correlation
        """
        embs = embs.squeeze(-1)  # Remove last dimension if present
        norm_embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        sim_matrix = torch.matmul(norm_embs, norm_embs.T)  # (batch_size, batch_size)
        
        # Vectorized correlation computation
        valid_samples = (id_mask.sum(dim=1) >= 2)  # Filter samples with at least 2 points
        if not valid_samples.any():
            return torch.tensor(0.0, device=sim_matrix.device)
            
        # Only process valid samples
        sim_matrix_valid = sim_matrix[valid_samples]  # (num_valid, batch_size)
        distances_valid = -distances[valid_samples]  # (num_valid, batch_size)
        id_mask_valid = id_mask[valid_samples]  # (num_valid, batch_size)
        
        # Compute means for valid elements
        sim_means = (sim_matrix_valid * id_mask_valid).sum(dim=1, keepdim=True) / id_mask_valid.sum(dim=1, keepdim=True)
        dist_means = (distances_valid * id_mask_valid).sum(dim=1, keepdim=True) / id_mask_valid.sum(dim=1, keepdim=True)
        
        # Center the valid elements
        sim_centered = sim_matrix_valid - sim_means
        dist_centered = distances_valid - dist_means
        
        # Compute covariance for valid elements
        cov = (sim_centered * dist_centered * id_mask_valid).sum(dim=1) / id_mask_valid.sum(dim=1)
        
        # Compute standard deviations for valid elements
        sim_std = torch.sqrt(((sim_centered ** 2) * id_mask_valid).sum(dim=1) / id_mask_valid.sum(dim=1))
        dist_std = torch.sqrt(((dist_centered ** 2) * id_mask_valid).sum(dim=1) / id_mask_valid.sum(dim=1))
        
        # Compute correlation with epsilon to prevent division by zero
        eps = 1e-8
        corrs = cov / (sim_std * dist_std + eps)
        
        # Average correlation across valid samples
        avg_corr = corrs.mean()
        return 1 - avg_corr if self.neg_coor else avg_corr

class MutualInformationLoss(nn.Module):
    def __init__(self, neg_coor=False, bins=100):
        super(MutualInformationLoss, self).__init__()
        self.neg_coor = neg_coor
        self.bins = bins

    def forward(self, emb1, emb2, distances):
        """
        Args:
            emb1: (batch_size, hidden_dim) first set of embeddings
            emb2: (batch_size, hidden_dim) second set of embeddings
            distances: (batch_size,) distance labels
        """
        # Calculate cosine similarity between emb1 and emb2
        emb1_norm = torch.nn.functional.normalize(emb1.squeeze(-1), p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2.squeeze(-1), p=2, dim=1)
        similarity = torch.sum(emb1_norm * emb2_norm, dim=1)  # (batch_size,)
        
        if len(similarity) == 0:
            return torch.tensor(0.0, device=similarity.device)
        
        # Discretize similarity and distance values into bins
        sim_bins = torch.linspace(-1, 1, self.bins + 1, device=similarity.device)
        dist_bins = torch.linspace(distances.min(), distances.max(), self.bins + 1, device=distances.device)
        
        # Calculate joint probability distribution
        joint_counts = torch.zeros((self.bins, self.bins), device=similarity.device)
        for i in range(self.bins):
            for j in range(self.bins):
                sim_mask = (similarity >= sim_bins[i]) & (similarity < sim_bins[i+1])
                dist_mask = (distances >= dist_bins[j]) & (distances < dist_bins[j+1])
                joint_counts[i,j] = (sim_mask & dist_mask).sum()
        
        # Normalize to get joint probability
        joint_prob = joint_counts / joint_counts.sum()
        
        # Calculate marginal probabilities
        sim_marginal = joint_prob.sum(dim=1)
        dist_marginal = joint_prob.sum(dim=0)
        
        # Calculate mutual information
        eps = 1e-10  # Small constant to avoid log(0)
        mi = joint_prob * (torch.log(joint_prob + eps) - 
                          torch.log(sim_marginal.unsqueeze(1) + eps) - 
                          torch.log(dist_marginal.unsqueeze(0) + eps))
        mi = mi.sum()
        
        # Normalize MI to [0,1] range
        mi = mi / torch.min(-torch.log(sim_marginal + eps).sum(), -torch.log(dist_marginal + eps).sum())
        
        return 1 - mi if self.neg_coor else mi
    
def spearman_rank_loss(S, D):
    # 将矩阵展平为向量
    s = S.flatten()
    d = D.flatten()
    
    # 计算秩（使用稳定排序）
    s_rank = torch.argsort(torch.argsort(s)).float()
    d_rank = torch.argsort(torch.argsort(d)).float()
    
    # 计算秩相关系数
    cov = torch.cov(torch.stack([s_rank, d_rank]))
    spearman = cov[0,1] / (torch.std(s_rank) * torch.std(d_rank))
    
    # 最大化负相关 → 最小化正相关
    return 1.0 + spearman  # 范围[0,2]

def mutual_info_loss(S, D, bins=20):
    # Normalize inputs to [0,1] range
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    D = (D - D.min()) / (D.max() - D.min() + 1e-8)
    
    # Calculate histograms with specified range [0,1]
    s_hist = torch.histc(S, bins=bins, min=0, max=1)
    d_hist = torch.histc(D, bins=bins, min=0, max=1)
    
    # Calculate joint histogram
    joint = torch.zeros((bins, bins), device=S.device)
    s_bin = (S * (bins-1)).long()
    d_bin = (D * (bins-1)).long()
    for i in range(len(S)):
        joint[s_bin[i], d_bin[i]] += 1
    
    # Normalize to get probabilities
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    d_hist = d_hist / (d_hist.sum() + 1e-8)
    joint = joint / (joint.sum() + 1e-8)
    
    # Calculate mutual information with small epsilon to avoid log(0)
    eps = 1e-8
    mi = joint * (torch.log(joint + eps) - 
                 torch.log(s_hist.unsqueeze(1) + eps) - 
                 torch.log(d_hist.unsqueeze(0) + eps))
    mi = mi.sum()
    
    # Normalize MI to [0,1] range and ensure positive
    mi = mi / torch.min(-torch.log(s_hist + eps).sum(), -torch.log(d_hist + eps).sum())
    mi = torch.clamp(mi, min=0)  # Ensure non-negative
    
    return 1 - mi  # Convert to loss (minimize this)

class ClassificationHead(nn.Module):
    def __init__(self, in_channel, num_classes, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_channel, 2 * in_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * in_channel, in_channel)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_channel, in_channel // 2)
        self.fc4 = nn.Linear(in_channel // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x  # Return logits, will apply softmax in loss function

class JointClassificationModel(torch.nn.Module):
    def __init__(self, gnn, classification_head):
        super(JointClassificationModel, self).__init__()
        self.gnn = gnn
        self.classification_head = classification_head
     
    def encode(self, graph):
        return self.gnn.encode(graph.x, graph.edge_index, graph.edge_attr)
        
    def decode(self, embs):
        return self.gnn.decoder(embs)
        
    def kl_loss(self):
        return self.gnn.kl_loss()
    
    def gen_emb(self, graph, weights):
        embs = self.encode(graph)
        return torch.matmul(embs.T, weights.unsqueeze(-1))
        
    def forward(self, graph, batch_weights):
        """
        Args:
            graph: Graph dataset
            batch_weights: [batch_size, num_aps] - WiFi weights for each sample in the batch
        
        Returns:
            class_logits: [batch_size, num_classes]
            recon_A: [num_aps, num_aps] - reconstructed adjacency matrix
        """
        embs = self.encode(graph)  # [num_aps, fp_dim]
        
        # Handle batch processing: batch_weights is [batch_size, num_aps]
        # embs.T is [fp_dim, num_aps]
        # We want to compute weighted sum for each sample in the batch
        batch_features = torch.matmul(batch_weights, embs)  # [batch_size, fp_dim]
        
        # Apply classification head
        class_logits = self.classification_head(batch_features)  # [batch_size, num_classes]
        
        # Reconstruction (independent of batch)
        recon_A = self.decode(embs)  # [num_aps, num_aps]
        
        return class_logits, recon_A

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, coordinates=None):
        """
        Args:
            features: N * num_aps (WiFi features)
            labels: N (classification labels)
            coordinates: N * 2 (optional coordinates)
        """
        self.features = features
        self.labels = labels
        self.coordinates = coordinates
        self.device = features.device
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.coordinates is not None:
            return self.features[idx], self.labels[idx], self.coordinates[idx]
        else:
            return self.features[idx], self.labels[idx]