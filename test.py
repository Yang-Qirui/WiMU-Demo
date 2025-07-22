import torch
import json

from utils import LDPL
from config import LDPL_MODE
from torch_geometric.nn import GAE
from gnn import GCNEncoder, MLPDecoder, MyMLP, JointModel

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = GAE(GCNEncoder(128, 128), MLPDecoder(128)).to(device)
mlp = MyMLP(128, 2).float().to(device)
model = JointModel(gnn, mlp).to(device)

# 加载预训练模型权重
model.load_state_dict(torch.load("./output/fine_tuned_model.pt", weights_only=False))

embs1 = torch.load("./output/embs.pt").flatten()
print(embs1.shape)
file = "data/train_json/(-4.306595, -15.354298).json"
with open(file, 'r') as f:
    data = json.load(f)
    data = data["1750659835640"]
graph_dataset = torch.load("./output/graph_dataset.pt", weights_only=False).to(device)
weights = torch.zeros(graph_dataset.num_nodes).to(device)
for key, value in data.items():
    weight = 1 / (1 + LDPL(value[0], band = value[1], mode = LDPL_MODE))
    weights[int(key)] = weight
weights = weights / weights.sum()
print(weights)

# 使用模型计算嵌入
model.eval()
with torch.no_grad():
    embs2 = model.gen_emb(graph_dataset, weights).flatten()

print(embs2.shape)
print(embs2)
# 计算余弦相似度
embs1_norm = torch.nn.functional.normalize(embs1, p=2, dim=0)
embs2_norm = torch.nn.functional.normalize(embs2, p=2, dim=0)
similarity = torch.sum(embs1_norm * embs2_norm)
print("余弦相似度:", similarity)
a = model.mlp(embs1.unsqueeze(0))
b = model(graph_dataset, weights)
print(a, b[0])
pos_range = torch.load("./output/norm_params.pt")['pos_range'].to(device)
pos_min = torch.load("./output/norm_params.pt")['pos_min'].to(device)
print(a * pos_range + pos_min)
print(b[0] * pos_range + pos_min)
