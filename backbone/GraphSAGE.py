import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(SAGEConv(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hid_dim, hid_dim))
            self.layers.append(SAGEConv(hid_dim, output_dim))
        else:
            self.layers.append(SAGEConv(input_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.layers[-1](x, edge_index)
        return x, F.log_softmax(out, dim=1)
