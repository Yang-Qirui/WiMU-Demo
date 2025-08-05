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