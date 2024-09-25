import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_FGGP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN_FGGP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        if num_layers > 1:
            self.layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        else:
            self.layers.append(GCNConv(input_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        features = x.clone()

        x = self.classifier(features)
        out = F.log_softmax(x, dim=1)  # 类别的对数概率
        confidence = F.softmax(x, dim=1)  # 计算置信度

        return out, features, confidence  # 返回置信度
