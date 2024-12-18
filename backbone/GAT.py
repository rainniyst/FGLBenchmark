import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        if num_layers > 1:
            self.layers.append(GATConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_dim, hidden_dim))
            self.layers.append(GATConv(hidden_dim, output_dim))
        else:
            self.layers.append(GATConv(input_dim, output_dim))

    def forward(self, data):
        x,  edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.layers[-1](x, edge_index)
        return x, F.log_softmax(out, dim=1)


